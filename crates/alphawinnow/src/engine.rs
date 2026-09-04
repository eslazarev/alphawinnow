use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use rayon::prelude::*;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    RejectionReason, SearchSpec, StructuralDescriptor, StructuralScoreComponents,
    StructuralScoreConfig, StructuralScoreWeights, analyze_expression,
    artifact::{
        ArtifactError, CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA, CandidateRecord, CheckpointDraft,
        MANIFEST_SCHEMA, RunCheckpoint, RunManifest, candidate_content, read_candidates,
        read_checkpoint, write_checkpoint_atomic,
    },
    canonical::{canonical, digest, fingerprint},
    deduplicate_with_catalog, describe_with_catalog, descriptor_distance,
    operators::Catalog,
    parse_expression_with_catalog,
    transform::{
        Provenance, crossover_with_catalog, generate_with_catalog, mutate_with_catalog,
        validate_transformed_with_catalog,
    },
};

const INITIAL_POPULATION: usize = 64;
const OFFSPRING_BATCH: usize = 128;

#[derive(Debug)]
pub struct SearchResult {
    pub candidates: Vec<CandidateRecord>,
    pub manifest: RunManifest,
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid search configuration: {0}")]
    Spec(#[from] crate::SpecError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    #[error("invalid archive candidate: {0}")]
    Archive(#[from] crate::DedupError),
    #[error("cannot initialize bounded worker pool: {0}")]
    WorkerPool(#[from] rayon::ThreadPoolBuildError),
    #[error("cannot serialize search configuration: {0}")]
    Config(#[from] serde_json::Error),
    #[error("checkpoint is incompatible with this run: {0}")]
    Checkpoint(String),
    #[error("invalid operator catalog: {0}")]
    Catalog(String),
}

#[derive(Debug, Clone, Default)]
pub struct SearchRunOptions {
    pub checkpoint_path: Option<PathBuf>,
    pub resume_from: Option<PathBuf>,
    pub pause_after_generations: Option<u64>,
}

#[derive(Debug, Clone)]
struct Draft {
    expression: crate::Expr,
    provenance: Provenance,
    semantic_fingerprint: String,
    descriptor: StructuralDescriptor,
}

#[derive(Debug, Clone)]
struct ScoredDraft {
    draft: Draft,
    structural_score: f64,
    components: StructuralScoreComponents,
    weights: StructuralScoreWeights,
}

#[derive(Debug)]
struct Prepared {
    ordinal: u64,
    outcome: PreparedOutcome,
}

#[derive(Debug)]
enum PreparedOutcome {
    Invalid,
    Trivial(RejectionReason),
    Candidate(Box<Draft>),
}

#[derive(Debug, Clone)]
struct SemanticArchive {
    capacity: usize,
    entries: HashSet<String>,
    insertion_order: VecDeque<String>,
}

impl SemanticArchive {
    fn new(capacity: usize, initial: impl IntoIterator<Item = String>) -> Self {
        let mut archive = Self {
            capacity,
            entries: HashSet::with_capacity(capacity.min(16_384)),
            insertion_order: VecDeque::with_capacity(capacity.min(16_384)),
        };
        for identity in initial {
            archive.insert(identity);
        }
        archive
    }

    fn insert(&mut self, identity: String) -> bool {
        if self.entries.contains(&identity) {
            return false;
        }
        if self.entries.len() == self.capacity
            && let Some(expired) = self.insertion_order.pop_front()
        {
            self.entries.remove(&expired);
        }
        self.insertion_order.push_back(identity.clone());
        self.entries.insert(identity)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn snapshot(&self) -> Vec<String> {
        self.insertion_order.iter().cloned().collect()
    }
}

impl From<CheckpointDraft> for Draft {
    fn from(value: CheckpointDraft) -> Self {
        Self {
            expression: crate::parse_expression(&value.expression)
                .expect("checkpoint expressions were validated before publication"),
            provenance: value.provenance,
            semantic_fingerprint: value.semantic_fingerprint,
            descriptor: value.descriptor,
        }
    }
}

impl From<&Draft> for CheckpointDraft {
    fn from(value: &Draft) -> Self {
        Self {
            expression: canonical(&value.expression),
            provenance: value.provenance.clone(),
            semantic_fingerprint: value.semantic_fingerprint.clone(),
            descriptor: value.descriptor.clone(),
        }
    }
}

fn draft_from_checkpoint(value: CheckpointDraft, catalog: &Catalog) -> Result<Draft, SearchError> {
    let expression =
        parse_expression_with_catalog(&value.expression, catalog).map_err(|error| {
            SearchError::Checkpoint(format!("checkpoint expression is invalid: {error}"))
        })?;
    Ok(Draft {
        expression,
        provenance: value.provenance,
        semantic_fingerprint: value.semantic_fingerprint,
        descriptor: value.descriptor,
    })
}

/// Run a bounded, offline structural search. No market metric is computed.
///
/// # Errors
/// Returns an error for invalid configuration, archive input, serialization, or
/// worker-pool initialization.
pub fn run_search(spec: &SearchSpec, archive: Option<&Path>) -> Result<SearchResult, SearchError> {
    run_search_with_options(spec, archive, &SearchRunOptions::default())
}

/// Run bounded search against an explicitly supplied public catalog.
///
/// # Errors
/// Returns an error for invalid configuration, catalog, archive input, or
/// worker-pool initialization.
pub fn run_search_with_catalog(
    spec: &SearchSpec,
    archive: Option<&Path>,
    catalog: &Catalog,
) -> Result<SearchResult, SearchError> {
    run_search_with_options_and_catalog(spec, archive, &SearchRunOptions::default(), catalog)
}

/// Run search with optional deterministic checkpointing and resume.
///
/// # Errors
/// Returns the same failures as [`run_search`] plus incompatible checkpoint
/// metadata.
#[allow(clippy::too_many_lines)]
pub fn run_search_with_options(
    spec: &SearchSpec,
    archive: Option<&Path>,
    options: &SearchRunOptions,
) -> Result<SearchResult, SearchError> {
    run_search_with_options_and_catalog(spec, archive, options, crate::operators::builtin_catalog())
}

/// Run checkpointable search against an explicitly supplied public catalog.
///
/// # Errors
/// Returns the same failures as [`run_search_with_catalog`] plus incompatible
/// checkpoint metadata.
#[allow(clippy::too_many_lines)]
pub fn run_search_with_options_and_catalog(
    spec: &SearchSpec,
    archive: Option<&Path>,
    options: &SearchRunOptions,
    catalog: &Catalog,
) -> Result<SearchResult, SearchError> {
    spec.validate()?;
    catalog.validate().map_err(SearchError::Catalog)?;
    validate_search_catalog(catalog)?;
    let config = serde_json::to_string(spec)?;
    let config_checksum = digest(&config);
    let mut candidate_identity_spec = spec.clone();
    candidate_identity_spec.threads = 0;
    let candidate_identity_checksum = digest(&serde_json::to_string(&candidate_identity_spec)?);
    let operator_catalog_checksum = catalog.checksum();
    let started = Instant::now();
    let deadline = Duration::from_secs(spec.duration_seconds);
    let finalization_reserve = (deadline / 4).min(Duration::from_millis(
        spec.resources.finalization_reserve_milliseconds,
    ));
    let generation_deadline = deadline.saturating_sub(finalization_reserve);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.threads)
        .build()?;
    let raw_archive_records = archive
        .map(read_candidates)
        .transpose()?
        .unwrap_or_default();
    let archive_record_count = raw_archive_records.len();
    let archive_partition = deduplicate_with_catalog(raw_archive_records, catalog)?;
    let archive_content_checksum = archive_state_checksum(&archive_partition.accepted);
    let run_id = digest(&format!(
        "alphawinnow:{candidate_identity_checksum}:{archive_content_checksum}"
    ));
    let archive_ids: Vec<_> = archive_partition
        .accepted
        .iter()
        .map(|record| record.semantic_fingerprint.clone())
        .collect();
    let mut archive_descriptors: Vec<_> = archive_partition
        .accepted
        .iter()
        .filter_map(|record| {
            parse_expression_with_catalog(&record.expression, catalog)
                .ok()
                .map(|expression| describe_with_catalog(&expression, &record.provenance, catalog))
        })
        .collect();
    archive_descriptors.truncate(spec.resources.descriptor_archive_capacity);

    let mut attempted = 0_u64;
    let mut rejected_invalid = 0_u64;
    let mut duplicates = 0_u64;
    let mut valid_candidates = 0_u64;
    let mut accepted_transform_counts = BTreeMap::new();
    let mut rejection_reasons = BTreeMap::new();
    let mut seen = SemanticArchive::new(spec.resources.semantic_archive_capacity, archive_ids);
    let mut drafts = Vec::new();
    let mut population = Vec::new();
    let mut peak_population = 0_usize;
    let mut peak_elite_archive = 0_usize;
    let mut peak_semantic_archive = seen.len();
    let mut generations_completed = 0_u64;
    let mut deadline_reached = false;
    let mut checkpoint_paused = false;
    let mut generation = 1_u64;
    let mut restored_parents = None;

    let resumed = if let Some(path) = &options.resume_from {
        let checkpoint = read_checkpoint(path)?;
        validate_checkpoint(
            &checkpoint,
            &run_id,
            &config_checksum,
            &operator_catalog_checksum,
        )?;
        attempted = checkpoint.attempted_candidates;
        rejected_invalid = checkpoint.rejected_invalid_candidates;
        duplicates = checkpoint.duplicate_candidates;
        valid_candidates = checkpoint.valid_candidates;
        accepted_transform_counts = checkpoint.accepted_transform_counts;
        rejection_reasons = checkpoint.rejection_reasons;
        seen = SemanticArchive::new(
            spec.resources.semantic_archive_capacity,
            checkpoint.semantic_archive,
        );
        drafts = checkpoint
            .elite_archive
            .into_iter()
            .map(|value| draft_from_checkpoint(value, catalog))
            .collect::<Result<Vec<_>, _>>()?;
        population = checkpoint
            .population
            .into_iter()
            .map(|value| draft_from_checkpoint(value, catalog))
            .collect::<Result<Vec<_>, _>>()?;
        restored_parents = Some(
            checkpoint
                .parent_pool
                .into_iter()
                .map(|value| draft_from_checkpoint(value, catalog))
                .collect::<Result<Vec<_>, _>>()?,
        );
        peak_population = checkpoint.peak_population;
        peak_elite_archive = checkpoint.peak_elite_archive;
        peak_semantic_archive = checkpoint.peak_semantic_archive;
        generations_completed = checkpoint.generations_completed;
        generation = checkpoint.next_generation;
        true
    } else {
        let initial_count = usize::try_from(spec.max_candidates.min(INITIAL_POPULATION as u64))
            .unwrap_or(INITIAL_POPULATION);
        let mut initial = pool.install(|| {
            (0..initial_count)
                .into_par_iter()
                .map(|slot| {
                    let ordinal = u64::try_from(slot).unwrap_or(u64::MAX);
                    let (expression, provenance) =
                        generate_with_catalog(spec.seed, ordinal, &spec.limits, catalog);
                    prepare(ordinal, &expression, provenance, spec, catalog)
                })
                .collect::<Vec<_>>()
        });
        merge_prepared(
            &mut initial,
            &mut attempted,
            &mut rejected_invalid,
            &mut duplicates,
            &mut valid_candidates,
            &mut accepted_transform_counts,
            &mut rejection_reasons,
            &mut seen,
            &mut drafts,
            &mut population,
        );
        generations_completed += 1;
        false
    };
    if drafts.len() > spec.resources.elite_archive_capacity {
        drafts = bounded_elite_archive(drafts, spec.resources.elite_archive_capacity);
    }
    peak_elite_archive = peak_elite_archive.max(drafts.len());
    peak_semantic_archive = peak_semantic_archive.max(seen.len());
    if !resumed {
        let scored_population =
            score_drafts(&pool, &population, &archive_descriptors, &spec.scoring);
        population = diverse_elites(scored_population, spec.resources.population_capacity);
    }
    let mut parents: Vec<_> = restored_parents.unwrap_or_else(|| {
        population
            .iter()
            .take(spec.resources.parent_pool_capacity)
            .cloned()
            .collect()
    });
    peak_population = peak_population.max(population.len());
    persist_checkpoint(
        options.checkpoint_path.as_deref(),
        &run_id,
        &config_checksum,
        &operator_catalog_checksum,
        generation,
        generations_completed,
        attempted,
        rejected_invalid,
        duplicates,
        valid_candidates,
        &rejection_reasons,
        &accepted_transform_counts,
        &seen,
        &drafts,
        &population,
        &parents,
        peak_population,
        peak_elite_archive,
        peak_semantic_archive,
        "running",
    )?;
    checkpoint_paused |= options
        .pause_after_generations
        .is_some_and(|limit| generations_completed >= limit);

    while attempted < spec.max_candidates && !checkpoint_paused {
        if started.elapsed() >= generation_deadline {
            deadline_reached = true;
            break;
        }
        let remaining = spec.max_candidates - attempted;
        let batch_size =
            usize::try_from(remaining.min(OFFSPRING_BATCH as u64)).unwrap_or(OFFSPRING_BATCH);
        let batch_start = attempted;
        let mut prepared = pool.install(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|slot| {
                    let slot = u64::try_from(slot).unwrap_or(u64::MAX);
                    let ordinal = batch_start + slot;
                    let (expression, provenance) =
                        offspring(spec, generation, slot, ordinal, &parents, catalog);
                    prepare(ordinal, &expression, provenance, spec, catalog)
                })
                .collect::<Vec<_>>()
        });
        let mut accepted = Vec::new();
        merge_prepared(
            &mut prepared,
            &mut attempted,
            &mut rejected_invalid,
            &mut duplicates,
            &mut valid_candidates,
            &mut accepted_transform_counts,
            &mut rejection_reasons,
            &mut seen,
            &mut drafts,
            &mut accepted,
        );
        population.extend(accepted);
        if drafts.len() > spec.resources.elite_archive_capacity {
            drafts = bounded_elite_archive(drafts, spec.resources.elite_archive_capacity);
        }
        if started.elapsed() >= generation_deadline {
            population.truncate(spec.resources.population_capacity);
            peak_population = peak_population.max(population.len());
            generations_completed += 1;
            generation += 1;
            deadline_reached = true;
            break;
        }
        let scored_population =
            score_drafts(&pool, &population, &archive_descriptors, &spec.scoring);
        population = diverse_elites(scored_population, spec.resources.population_capacity);
        parents = population
            .iter()
            .take(spec.resources.parent_pool_capacity)
            .cloned()
            .collect();
        peak_population = peak_population.max(population.len());
        peak_elite_archive = peak_elite_archive.max(drafts.len());
        peak_semantic_archive = peak_semantic_archive.max(seen.len());
        generations_completed += 1;
        generation += 1;
        persist_checkpoint(
            options.checkpoint_path.as_deref(),
            &run_id,
            &config_checksum,
            &operator_catalog_checksum,
            generation,
            generations_completed,
            attempted,
            rejected_invalid,
            duplicates,
            valid_candidates,
            &rejection_reasons,
            &accepted_transform_counts,
            &seen,
            &drafts,
            &population,
            &parents,
            peak_population,
            peak_elite_archive,
            peak_semantic_archive,
            "running",
        )?;
        checkpoint_paused |= options
            .pause_after_generations
            .is_some_and(|limit| generations_completed >= limit);
    }

    let final_drafts = if deadline_reached {
        &population[..population.len().min(spec.shortlist_size.saturating_mul(2))]
    } else {
        drafts.as_slice()
    };
    let scored = score_drafts(&pool, final_drafts, &archive_descriptors, &spec.scoring);
    let mut records = pool.install(|| {
        scored
            .par_iter()
            .map(|candidate| candidate_record(candidate, &run_id))
            .collect::<Vec<_>>()
    });
    sort_records(&mut records);
    let shortlist_deadline = started + deadline.saturating_sub(Duration::from_millis(25));
    let candidates = diverse_shortlist(records, spec.shortlist_size, Some(shortlist_deadline));
    let retained_transform_counts =
        transform_counts(candidates.iter().map(|candidate| candidate.provenance.kind));
    let content = candidate_content(&candidates)?;
    let manifest = RunManifest {
        schema: MANIFEST_SCHEMA,
        run_id: run_id.clone(),
        candidate_schema: CANDIDATE_SCHEMA,
        operator_catalog_checksum: operator_catalog_checksum.clone(),
        tool_version: env!("CARGO_PKG_VERSION").to_owned(),
        search_spec: spec.clone(),
        config_checksum: config_checksum.clone(),
        archive_records: u64::try_from(archive_record_count).unwrap_or(u64::MAX),
        archive_accepted_records: archive_partition.accepted.len() as u64,
        archive_duplicate_records: archive_partition.duplicates.len() as u64,
        archive_rejected_records: archive_partition.rejected.len() as u64,
        archive_rejection_reasons: archive_partition.rejection_counts(),
        attempted_candidates: attempted,
        rejected_invalid_candidates: rejected_invalid,
        valid_candidates,
        rejected_trivial_candidates: rejection_reasons.values().sum(),
        rejection_reasons: rejection_reasons.clone(),
        duplicate_candidates: duplicates,
        retained_candidates: candidates.len() as u64,
        accepted_transform_counts: accepted_transform_counts.clone(),
        retained_transform_counts,
        generations_completed,
        population_capacity: spec.resources.population_capacity,
        peak_population,
        elite_archive_capacity: spec.resources.elite_archive_capacity,
        peak_elite_archive,
        semantic_archive_capacity: spec.resources.semantic_archive_capacity,
        peak_semantic_archive,
        descriptor_archive_capacity: spec.resources.descriptor_archive_capacity,
        peak_descriptor_archive: archive_descriptors.len(),
        elapsed_milliseconds: started.elapsed().as_millis(),
        termination_reason: if checkpoint_paused {
            "checkpoint_pause"
        } else if deadline_reached {
            "duration_deadline"
        } else {
            "max_candidates"
        }
        .to_owned(),
        candidate_content_checksum: digest(&content),
    };
    persist_checkpoint(
        options.checkpoint_path.as_deref(),
        &run_id,
        &config_checksum,
        &operator_catalog_checksum,
        generation,
        generations_completed,
        attempted,
        rejected_invalid,
        duplicates,
        valid_candidates,
        &rejection_reasons,
        &accepted_transform_counts,
        &seen,
        &drafts,
        &population,
        &parents,
        peak_population,
        peak_elite_archive,
        peak_semantic_archive,
        &manifest.termination_reason,
    )?;
    Ok(SearchResult {
        candidates,
        manifest,
    })
}

fn archive_state_checksum(records: &[CandidateRecord]) -> String {
    let mut identities: Vec<_> = records
        .iter()
        .map(|record| record.semantic_fingerprint.as_str())
        .collect();
    identities.sort_unstable();
    digest(&identities.join("\n"))
}

fn validate_search_catalog(catalog: &Catalog) -> Result<(), SearchError> {
    let builtin = crate::operators::builtin_catalog();
    for operator in &catalog.operators {
        let Some(reference) = builtin.lookup(&operator.name) else {
            return Err(SearchError::Catalog(format!(
                "search operator `{}` is not implemented by this binary",
                operator.name
            )));
        };
        if operator.arity != reference.arity
            || !inputs_compatible(&operator.inputs, &reference.inputs)
            || operator.required_input_kinds != reference.required_input_kinds
            || operator.output != reference.output
            || operator.commutative != reference.commutative
            || operator.associative != reference.associative
            || operator.keywords != reference.keywords
            || operator.keyword_constraints != reference.keyword_constraints
        {
            return Err(SearchError::Catalog(format!(
                "search operator `{}` changes its implemented grammar",
                operator.name
            )));
        }
    }
    Ok(())
}

fn inputs_compatible(
    configured: &[crate::operators::ArgumentSpec],
    reference: &[crate::operators::ArgumentSpec],
) -> bool {
    configured.len() == reference.len()
        && configured
            .iter()
            .zip(reference)
            .all(|(configured, reference)| {
                configured.kinds == reference.kinds
                    && match (&configured.domain, &reference.domain) {
                        (left, right) if left == right => true,
                        (
                            crate::ValueDomain::WindowSet { values },
                            crate::ValueDomain::Window { min, max },
                        ) => values.iter().all(|value| (*min..=*max).contains(value)),
                        _ => false,
                    }
            })
}

fn validate_checkpoint(
    checkpoint: &RunCheckpoint,
    run_id: &str,
    config_checksum: &str,
    operator_catalog_checksum: &str,
) -> Result<(), SearchError> {
    if checkpoint.schema != CHECKPOINT_SCHEMA {
        return Err(SearchError::Checkpoint(format!(
            "unsupported schema {}",
            checkpoint.schema
        )));
    }
    if checkpoint.run_id != run_id
        || checkpoint.config_checksum != config_checksum
        || checkpoint.operator_catalog_checksum != operator_catalog_checksum
    {
        return Err(SearchError::Checkpoint(
            "run, configuration, or operator catalog checksum changed".to_owned(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn persist_checkpoint(
    path: Option<&Path>,
    run_id: &str,
    config_checksum: &str,
    operator_catalog_checksum: &str,
    next_generation: u64,
    generations_completed: u64,
    attempted_candidates: u64,
    rejected_invalid_candidates: u64,
    duplicate_candidates: u64,
    valid_candidates: u64,
    rejection_reasons: &BTreeMap<RejectionReason, u64>,
    accepted_transform_counts: &BTreeMap<crate::TransformKind, u64>,
    semantic_archive: &SemanticArchive,
    elite_archive: &[Draft],
    population: &[Draft],
    parent_pool: &[Draft],
    peak_population: usize,
    peak_elite_archive: usize,
    peak_semantic_archive: usize,
    termination_state: &str,
) -> Result<(), SearchError> {
    let Some(path) = path else {
        return Ok(());
    };
    let checkpoint = RunCheckpoint {
        schema: CHECKPOINT_SCHEMA,
        run_id: run_id.to_owned(),
        config_checksum: config_checksum.to_owned(),
        operator_catalog_checksum: operator_catalog_checksum.to_owned(),
        next_generation,
        generations_completed,
        attempted_candidates,
        rejected_invalid_candidates,
        duplicate_candidates,
        valid_candidates,
        rejection_reasons: rejection_reasons.clone(),
        accepted_transform_counts: accepted_transform_counts.clone(),
        semantic_archive: semantic_archive.snapshot(),
        elite_archive: elite_archive.iter().map(CheckpointDraft::from).collect(),
        population: population.iter().map(CheckpointDraft::from).collect(),
        parent_pool: parent_pool.iter().map(CheckpointDraft::from).collect(),
        peak_population,
        peak_elite_archive,
        peak_semantic_archive,
        termination_state: termination_state.to_owned(),
    };
    write_checkpoint_atomic(path, &checkpoint)?;
    Ok(())
}

fn prepare(
    ordinal: u64,
    expression: &crate::Expr,
    provenance: Provenance,
    spec: &SearchSpec,
    catalog: &Catalog,
) -> Prepared {
    let Ok(expression) = parse_expression_with_catalog(&canonical(expression), catalog) else {
        return Prepared {
            ordinal,
            outcome: PreparedOutcome::Invalid,
        };
    };
    if validate_transformed_with_catalog(&expression, &spec.limits, catalog).is_err() {
        return Prepared {
            ordinal,
            outcome: PreparedOutcome::Invalid,
        };
    }
    let analysis = analyze_expression(&expression);
    let outcome = if let Some(reason) = analysis.rejection_reason {
        PreparedOutcome::Trivial(reason)
    } else {
        let descriptor = describe_with_catalog(&expression, &provenance, catalog);
        PreparedOutcome::Candidate(Box::new(Draft {
            expression,
            provenance,
            semantic_fingerprint: analysis.semantic_fingerprint,
            descriptor,
        }))
    };
    Prepared { ordinal, outcome }
}

#[allow(clippy::too_many_arguments)]
fn merge_prepared(
    prepared: &mut [Prepared],
    attempted: &mut u64,
    rejected_invalid: &mut u64,
    duplicates: &mut u64,
    valid_candidates: &mut u64,
    accepted_transform_counts: &mut BTreeMap<crate::TransformKind, u64>,
    rejection_reasons: &mut BTreeMap<RejectionReason, u64>,
    seen: &mut SemanticArchive,
    drafts: &mut Vec<Draft>,
    accepted: &mut Vec<Draft>,
) {
    prepared.sort_by_key(|item| item.ordinal);
    for item in prepared {
        *attempted += 1;
        match &item.outcome {
            PreparedOutcome::Invalid => *rejected_invalid += 1,
            PreparedOutcome::Trivial(reason) => {
                *rejection_reasons.entry(*reason).or_insert(0) += 1;
            }
            PreparedOutcome::Candidate(draft) => {
                if seen.insert(draft.semantic_fingerprint.clone()) {
                    *valid_candidates += 1;
                    *accepted_transform_counts
                        .entry(draft.provenance.kind)
                        .or_insert(0) += 1;
                    drafts.push(draft.as_ref().clone());
                    accepted.push(draft.as_ref().clone());
                } else {
                    *duplicates += 1;
                }
            }
        }
    }
}

fn offspring(
    spec: &SearchSpec,
    generation: u64,
    slot: u64,
    ordinal: u64,
    parents: &[Draft],
    catalog: &Catalog,
) -> (crate::Expr, Provenance) {
    if parents.is_empty() {
        let seed = offspring_seed(spec.seed, generation, slot, &[], "independent");
        return generate_with_catalog(seed, ordinal, &spec.limits, catalog);
    }
    if slot.is_multiple_of(3) || parents.len() == 1 {
        let index = parent_index(slot, 17, generation, parents.len());
        let parent = &parents[index];
        let seed = offspring_seed(
            spec.seed,
            generation,
            slot,
            std::slice::from_ref(&parent.semantic_fingerprint),
            "mutation",
        );
        mutate_with_catalog(&parent.expression, seed, ordinal, &spec.limits, catalog)
    } else {
        let left_index = parent_index(slot, 31, generation, parents.len());
        let mut right_index = parent_index(slot, 47, generation + 1, parents.len());
        if parents.len() > 1 && right_index == left_index {
            right_index = (right_index + 1) % parents.len();
        }
        let left = &parents[left_index];
        let right = &parents[right_index];
        let parent_ids = [
            left.semantic_fingerprint.clone(),
            right.semantic_fingerprint.clone(),
        ];
        let seed = offspring_seed(spec.seed, generation, slot, &parent_ids, "crossover");
        crossover_with_catalog(
            &left.expression,
            &right.expression,
            seed,
            ordinal,
            &spec.limits,
            catalog,
        )
    }
}

fn offspring_seed(
    run_seed: u64,
    generation: u64,
    slot: u64,
    parent_fingerprints: &[String],
    domain: &str,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(run_seed.to_le_bytes());
    hasher.update(generation.to_le_bytes());
    hasher.update(slot.to_le_bytes());
    hasher.update((parent_fingerprints.len() as u64).to_le_bytes());
    for parent in parent_fingerprints {
        hasher.update((parent.len() as u64).to_le_bytes());
        hasher.update(parent.as_bytes());
    }
    hasher.update((domain.len() as u64).to_le_bytes());
    hasher.update(domain.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn score_drafts(
    pool: &rayon::ThreadPool,
    drafts: &[Draft],
    archive_descriptors: &[StructuralDescriptor],
    scoring: &StructuralScoreConfig,
) -> Vec<ScoredDraft> {
    let mut family_counts = HashMap::<String, usize>::new();
    let mut transform_counts = BTreeMap::<crate::TransformKind, usize>::new();
    for draft in drafts {
        *family_counts
            .entry(root_family(&draft.expression))
            .or_default() += 1;
        *transform_counts.entry(draft.provenance.kind).or_default() += 1;
    }
    let weights = scoring.normalized();
    let mut scored = pool.install(|| {
        drafts
            .par_iter()
            .enumerate()
            .map(|(index, draft)| {
                let family = root_family(&draft.expression);
                let family_count = usize_as_f64(family_counts[&family]);
                let simplicity = 1.0
                    / (1.0
                        + usize_as_f64(draft.expression.operator_count())
                        + usize_as_f64(draft.expression.node_count()) / 10.0);
                let archive_novelty = nearest_descriptor_distance(
                    &draft.descriptor,
                    archive_descriptors.iter().take(64),
                );
                let family_diversity = 1.0 / family_count.sqrt();
                let distinct_parents = draft
                    .provenance
                    .parent_fingerprints
                    .iter()
                    .collect::<HashSet<_>>()
                    .len();
                let lineage_diversity = (usize_as_f64(distinct_parents) / 2.0).min(1.0);
                let structural_novelty = peer_novelty(index, drafts);
                let transform_count = transform_counts[&draft.provenance.kind];
                let transform_diversity = 1.0 / usize_as_f64(transform_count).sqrt();
                let components = StructuralScoreComponents {
                    simplicity,
                    archive_novelty,
                    family_diversity,
                    lineage_diversity,
                    structural_novelty,
                    transform_diversity,
                };
                let structural_score = weights.simplicity * components.simplicity
                    + weights.archive_novelty * components.archive_novelty
                    + weights.family_diversity * components.family_diversity
                    + weights.lineage_diversity * components.lineage_diversity
                    + weights.structural_novelty * components.structural_novelty
                    + weights.transform_diversity * components.transform_diversity;
                ScoredDraft {
                    draft: draft.clone(),
                    structural_score,
                    components,
                    weights,
                }
            })
            .collect::<Vec<_>>()
    });
    scored.sort_by(|left, right| {
        right
            .structural_score
            .total_cmp(&left.structural_score)
            .then_with(|| {
                left.draft
                    .expression
                    .node_count()
                    .cmp(&right.draft.expression.node_count())
            })
            .then_with(|| {
                left.draft
                    .semantic_fingerprint
                    .cmp(&right.draft.semantic_fingerprint)
            })
    });
    scored
}

fn nearest_descriptor_distance<'a>(
    descriptor: &StructuralDescriptor,
    others: impl Iterator<Item = &'a StructuralDescriptor>,
) -> f64 {
    others
        .map(|other| descriptor_distance(descriptor, other))
        .reduce(f64::min)
        .unwrap_or(1.0)
}

fn peer_novelty(index: usize, drafts: &[Draft]) -> f64 {
    if drafts.len() <= 1 {
        return 1.0;
    }
    let stride = (drafts.len() / 16).max(1);
    nearest_descriptor_distance(
        &drafts[index].descriptor,
        drafts
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .step_by(stride)
            .take(16)
            .map(|(_, draft)| &draft.descriptor),
    )
}

fn diverse_elites(scored: Vec<ScoredDraft>, limit: usize) -> Vec<Draft> {
    let mut families = BTreeMap::<(crate::TransformKind, String, u8), VecDeque<ScoredDraft>>::new();
    for candidate in scored {
        families
            .entry((
                candidate.draft.provenance.kind,
                root_family(&candidate.draft.expression),
                node_bucket(candidate.draft.expression.node_count()),
            ))
            .or_default()
            .push_back(candidate);
    }
    let mut queues: Vec<_> = families.into_values().collect();
    queues.sort_by(|left, right| {
        let left_score = left
            .front()
            .map_or(f64::NEG_INFINITY, |item| item.structural_score);
        let right_score = right
            .front()
            .map_or(f64::NEG_INFINITY, |item| item.structural_score);
        right_score.total_cmp(&left_score)
    });
    let mut result = Vec::new();
    while result.len() < limit {
        let mut progressed = false;
        for queue in &mut queues {
            if let Some(candidate) = queue.pop_front()
                && result.len() < limit
            {
                result.push(candidate.draft);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    result
}

fn bounded_elite_archive(drafts: Vec<Draft>, limit: usize) -> Vec<Draft> {
    let mut families = BTreeMap::<(crate::TransformKind, String, u8), Vec<Draft>>::new();
    for draft in drafts {
        families
            .entry((
                draft.provenance.kind,
                root_family(&draft.expression),
                node_bucket(draft.expression.node_count()),
            ))
            .or_default()
            .push(draft);
    }
    for family in families.values_mut() {
        family.sort_by(|left, right| {
            retention_value(right)
                .cmp(&retention_value(left))
                .then_with(|| {
                    left.expression
                        .node_count()
                        .cmp(&right.expression.node_count())
                })
                .then_with(|| left.semantic_fingerprint.cmp(&right.semantic_fingerprint))
        });
    }
    let mut queues: Vec<_> = families.into_values().map(VecDeque::from).collect();
    let mut result = Vec::with_capacity(limit);
    while result.len() < limit {
        let mut progressed = false;
        for queue in &mut queues {
            if let Some(draft) = queue.pop_front()
                && result.len() < limit
            {
                result.push(draft);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    result
}

fn retention_value(draft: &Draft) -> usize {
    let descriptor = &draft.descriptor;
    descriptor.operator_shingles.len() * 8
        + descriptor.operator_histogram.len() * 5
        + descriptor.field_families.len() * 7
        + descriptor.fields.len() * 3
        + descriptor.window_buckets.len() * 4
        + descriptor.groups.len() * 5
        + usize::from(descriptor.nonlinear) * 3
        + usize::from(descriptor.conditional) * 5
        + descriptor.lineage_parents.len() * 2
}

fn candidate_record(scored: &ScoredDraft, run_id: &str) -> CandidateRecord {
    CandidateRecord {
        schema: CANDIDATE_SCHEMA,
        run_id: run_id.to_owned(),
        expression: canonical(&scored.draft.expression),
        fingerprint: fingerprint(&scored.draft.expression),
        semantic_fingerprint: scored.draft.semantic_fingerprint.clone(),
        structural_score: scored.structural_score,
        score_schema: 2,
        score_components: scored.components.clone(),
        normalized_weights: scored.weights,
        structural_descriptor: scored.draft.descriptor.clone(),
        root_family: root_family(&scored.draft.expression),
        nodes: scored.draft.expression.node_count(),
        depth: scored.draft.expression.depth(),
        provenance: scored.draft.provenance.clone(),
    }
}

fn root_family(expression: &crate::Expr) -> String {
    expression.operator().unwrap_or("field").to_owned()
}

fn node_bucket(nodes: usize) -> u8 {
    match nodes {
        0..=5 => 0,
        6..=12 => 1,
        13..=24 => 2,
        25..=48 => 3,
        49..=96 => 4,
        _ => 5,
    }
}

fn parent_index(ordinal: u64, multiplier: u64, offset: u64, length: usize) -> usize {
    let length = u64::try_from(length).expect("parent pool length fits in u64");
    usize::try_from(ordinal.wrapping_mul(multiplier).wrapping_add(offset) % length)
        .expect("index is smaller than the original usize length")
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn transform_counts(
    kinds: impl IntoIterator<Item = crate::TransformKind>,
) -> BTreeMap<crate::TransformKind, u64> {
    let mut counts = BTreeMap::new();
    for kind in kinds {
        *counts.entry(kind).or_insert(0) += 1;
    }
    counts
}

fn sort_records(records: &mut [CandidateRecord]) {
    records.sort_by(|left, right| {
        right
            .structural_score
            .total_cmp(&left.structural_score)
            .then_with(|| left.semantic_fingerprint.cmp(&right.semantic_fingerprint))
    });
}

fn diverse_shortlist(
    records: Vec<CandidateRecord>,
    limit: usize,
    deadline: Option<Instant>,
) -> Vec<CandidateRecord> {
    let mut families = BTreeMap::<(crate::TransformKind, u8), Vec<CandidateRecord>>::new();
    for record in records {
        families
            .entry((record.provenance.kind, node_bucket(record.nodes)))
            .or_default()
            .push(record);
    }
    let mut result = Vec::new();
    let mut budget_exhausted = false;
    while result.len() < limit {
        if deadline.is_some_and(|value| Instant::now() >= value) {
            budget_exhausted = true;
            break;
        }
        let mut progressed = false;
        for records in families.values_mut() {
            if !records.is_empty() && result.len() < limit {
                let index = descriptor_aware_choice(records, &result);
                result.push(records.remove(index));
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    if budget_exhausted {
        fill_from_score_order(&mut result, families, limit);
    }
    result
}

/// Tops a partially selected shortlist up from the score-ordered remainder.
///
/// The finalization deadline bounds how much diversity-aware selection is
/// affordable, never whether completed work is published at all. Without this
/// fallback a run whose finalization reserve is consumed by a loaded machine
/// publishes an empty shortlist while holding scored, admissible candidates,
/// which contradicts the documented hard-upper-bound-with-graceful-results
/// contract. The remainder is re-sorted with the same total order the caller
/// applied, so the fallback stays deterministic for a fixed record set.
fn fill_from_score_order(
    result: &mut Vec<CandidateRecord>,
    families: BTreeMap<(crate::TransformKind, u8), Vec<CandidateRecord>>,
    limit: usize,
) {
    if result.len() >= limit {
        return;
    }
    let mut remaining = families.into_values().flatten().collect::<Vec<_>>();
    sort_records(&mut remaining);
    result.extend(remaining.into_iter().take(limit - result.len()));
}

fn descriptor_aware_choice(records: &[CandidateRecord], selected: &[CandidateRecord]) -> usize {
    let selected_stride = (selected.len() / 32).max(1);
    records
        .iter()
        .take(32)
        .enumerate()
        .map(|(index, record)| {
            let novelty = nearest_descriptor_distance(
                &record.structural_descriptor,
                selected
                    .iter()
                    .step_by(selected_stride)
                    .take(32)
                    .map(|candidate| &candidate.structural_descriptor),
            );
            (index, record.structural_score + 0.35 * novelty)
        })
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Limits, SEARCH_SPEC_SCHEMA, StructuralScoreConfig, parse_expression};

    fn spec(threads: usize) -> SearchSpec {
        SearchSpec {
            schema: SEARCH_SPEC_SCHEMA,
            seed: 42,
            threads,
            max_candidates: 500,
            duration_seconds: 10,
            limits: Limits::default(),
            scoring: StructuralScoreConfig::default(),
            shortlist_size: 40,
            resources: crate::ResourceLimits::default(),
        }
    }

    fn draft(expression: &str) -> Draft {
        let expression = parse_expression(expression).unwrap();
        let semantic_fingerprint = analyze_expression(&expression).semantic_fingerprint;
        let (_, provenance) = generate_with_catalog(
            1,
            1,
            &Limits::default(),
            crate::operators::builtin_catalog(),
        );
        let descriptor = crate::describe(&expression, &provenance);
        Draft {
            expression,
            provenance,
            semantic_fingerprint,
            descriptor,
        }
    }

    #[test]
    fn fixed_configuration_has_deterministic_candidates() {
        let left = run_search(&spec(2), None).unwrap();
        let right = run_search(&spec(2), None).unwrap();
        assert_eq!(left.candidates, right.candidates);
        assert_eq!(
            left.manifest.candidate_content_checksum,
            right.manifest.candidate_content_checksum
        );
    }

    #[test]
    fn candidate_identity_is_deterministic_across_thread_counts() {
        let left = run_search(&spec(1), None).unwrap();
        let right = run_search(&spec(4), None).unwrap();
        assert_eq!(left.candidates, right.candidates);
        assert_eq!(
            left.manifest.candidate_content_checksum,
            right.manifest.candidate_content_checksum
        );
    }

    #[test]
    fn high_ranked_candidates_receive_selectable_descendants() {
        let candidates = vec![draft("close"), draft("rank(ts_mean(add(open, high), 20))")];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let scored = score_drafts(&pool, &candidates, &[], &StructuralScoreConfig::default());
        let parents = diverse_elites(scored, 2);
        let highest = parents[0].semantic_fingerprint.clone();
        let mut found_descendant = false;
        for slot in 0..32 {
            let (_, provenance) = offspring(
                &spec(1),
                1,
                slot,
                100 + slot,
                &parents,
                crate::operators::builtin_catalog(),
            );
            if provenance.parent_fingerprints.contains(&highest) {
                found_descendant = true;
                break;
            }
        }
        assert!(found_descendant);
    }

    #[test]
    fn archive_novelty_orders_near_and_distant_descriptors() {
        let archive = draft("rank(ts_mean(close, 20))").descriptor;
        let candidates = vec![
            draft("rank(ts_mean(open, 20))"),
            draft("group_rank(volume, group(\"industry\"))"),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let scored = score_drafts(
            &pool,
            &candidates,
            &[archive],
            &StructuralScoreConfig::default(),
        );
        let near = scored
            .iter()
            .find(|candidate| candidate.draft.expression.operator() == Some("rank"))
            .unwrap();
        let distant = scored
            .iter()
            .find(|candidate| candidate.draft.expression.operator() == Some("group_rank"))
            .unwrap();
        assert!(near.components.archive_novelty < distant.components.archive_novelty);
    }

    #[test]
    fn shortlist_balances_competitive_transform_kinds_and_descriptors() {
        let result = run_search(&spec(2), None).unwrap();
        assert!(result.manifest.retained_transform_counts.len() >= 3);
        assert!(
            result
                .manifest
                .retained_transform_counts
                .values()
                .all(|count| *count <= 20)
        );
        let descriptor_signatures: HashSet<_> = result
            .candidates
            .iter()
            .map(|candidate| candidate.structural_descriptor.signature())
            .collect();
        assert!(descriptor_signatures.len() > result.candidates.len() / 2);
    }

    #[test]
    fn shortlist_preserves_competitive_node_buckets_within_a_transform() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let seed = draft("close");
        let scored = score_drafts(&pool, &[seed], &[], &StructuralScoreConfig::default());
        let base = candidate_record(&scored[0], "node-bucket-test");
        let records = [2, 8, 18, 35, 70, 120]
            .into_iter()
            .enumerate()
            .map(|(ordinal, nodes)| {
                let mut record = base.clone();
                record.nodes = nodes;
                record.structural_score -= usize_as_f64(ordinal) * 0.01;
                record.fingerprint = format!("fingerprint-{ordinal}");
                record.semantic_fingerprint = format!("semantic-{ordinal}");
                record
            })
            .collect();

        let retained = diverse_shortlist(records, 6, None);
        let buckets: HashSet<_> = retained
            .iter()
            .map(|record| node_bucket(record.nodes))
            .collect();
        assert_eq!(buckets, HashSet::from([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn an_expired_finalization_budget_still_publishes_score_ordered_work() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let seed = draft("close");
        let scored = score_drafts(&pool, &[seed], &[], &StructuralScoreConfig::default());
        let base = candidate_record(&scored[0], "expired-budget-test");
        let records: Vec<_> = (0..12_usize)
            .map(|ordinal| {
                let mut record = base.clone();
                record.nodes = 2 + ordinal;
                record.structural_score -= usize_as_f64(ordinal) * 0.01;
                record.fingerprint = format!("fingerprint-{ordinal}");
                record.semantic_fingerprint = format!("semantic-{ordinal}");
                record
            })
            .collect();

        // An already-expired deadline is the state a loaded machine reaches
        // when generations consume the finalization reserve. Selection must
        // degrade from diversity-aware to score-ordered, never to nothing.
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);
        let retained = diverse_shortlist(records.clone(), 5, Some(expired));

        assert_eq!(retained.len(), 5);
        let mut expected = records;
        sort_records(&mut expected);
        assert_eq!(
            retained
                .iter()
                .map(|record| &record.semantic_fingerprint)
                .collect::<Vec<_>>(),
            expected
                .iter()
                .take(5)
                .map(|record| &record.semantic_fingerprint)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            diverse_shortlist(Vec::new(), 5, Some(expired)),
            Vec::<CandidateRecord>::new()
        );
    }

    #[test]
    fn population_and_parent_pool_are_bounded() {
        let configured = spec(2);
        let capacity = configured.resources.population_capacity;
        let result = run_search(&configured, None).unwrap();
        assert_eq!(result.manifest.population_capacity, capacity);
        assert!(result.manifest.peak_population <= capacity);
        assert!(result.manifest.generations_completed > 1);
    }

    #[test]
    fn manifest_records_the_exact_active_catalog_checksum() {
        let result = run_search(&spec(1), None).unwrap();
        assert_eq!(
            result.manifest.operator_catalog_checksum,
            crate::builtin_catalog().checksum()
        );
    }

    #[test]
    fn resident_archives_remain_bounded_as_attempts_grow() {
        let mut configured = spec(2);
        configured.max_candidates = 5_000;
        configured.resources.population_capacity = 64;
        configured.resources.parent_pool_capacity = 16;
        configured.resources.elite_archive_capacity = 100;
        configured.resources.semantic_archive_capacity = 200;
        configured.resources.descriptor_archive_capacity = 32;
        let result = run_search(&configured, None).unwrap();
        assert!(result.manifest.valid_candidates > 100);
        assert!(result.manifest.peak_population <= 64);
        assert!(result.manifest.peak_elite_archive <= 100);
        assert!(result.manifest.peak_semantic_archive <= 200);
        assert!(result.manifest.peak_descriptor_archive <= 32);
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_fixed_budget() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("search.checkpoint.json");
        let mut configured = spec(2);
        configured.max_candidates = 1_000;
        configured.resources.elite_archive_capacity = 200;
        let uninterrupted = run_search(&configured, None).unwrap();
        let paused = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                checkpoint_path: Some(checkpoint.clone()),
                resume_from: None,
                pause_after_generations: Some(3),
            },
        )
        .unwrap();
        assert_eq!(paused.manifest.termination_reason, "checkpoint_pause");
        let resumed = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                checkpoint_path: Some(checkpoint.clone()),
                resume_from: Some(checkpoint),
                pause_after_generations: None,
            },
        )
        .unwrap();
        assert_eq!(resumed.candidates, uninterrupted.candidates);
        assert_eq!(
            resumed.manifest.candidate_content_checksum,
            uninterrupted.manifest.candidate_content_checksum
        );
    }

    #[test]
    fn interrupted_compatibility_aliases_leave_previous_pair_readable() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("candidates.jsonl");
        let result = run_search(&spec(1), None).unwrap();
        crate::write_run_transactional(&output, &result.candidates, &result.manifest).unwrap();
        std::fs::write(&output, "partial compatibility alias\n").unwrap();
        std::fs::write(crate::artifact::manifest_path(&output), "stale alias\n").unwrap();
        let (records, manifest) = crate::read_published_run(&output).unwrap();
        assert_eq!(records.len(), result.candidates.len());
        assert_eq!(
            records
                .iter()
                .map(|record| &record.semantic_fingerprint)
                .collect::<Vec<_>>(),
            result
                .candidates
                .iter()
                .map(|record| &record.semantic_fingerprint)
                .collect::<Vec<_>>()
        );
        assert_eq!(manifest.run_id, result.manifest.run_id);
    }

    #[test]
    fn deadline_is_a_hard_upper_bound_with_graceful_results() {
        // Both constants are deliberately generous. A busy shared runner can
        // starve the single worker for an entire scheduling quantum, so a
        // one-second budget is long enough to expire before any candidate is
        // admitted, and a one-second margin is short enough to expire while
        // the search is shutting down. This test must fail when the deadline
        // is not enforced, not when the machine happens to be loaded.
        const BUDGET: Duration = Duration::from_secs(3);
        const TOLERATED_OVERSHOOT: Duration = Duration::from_secs(2);

        let mut bounded = spec(1);
        bounded.max_candidates = crate::MAX_CANDIDATES;
        bounded.duration_seconds = BUDGET.as_secs();
        let started = Instant::now();
        let result = run_search(&bounded, None).unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed < BUDGET + TOLERATED_OVERSHOOT,
            "search took {elapsed:?}, overshooting the {BUDGET:?} deadline by more than {TOLERATED_OVERSHOOT:?}"
        );
        assert_eq!(result.manifest.termination_reason, "duration_deadline");
        assert!(
            !result.candidates.is_empty(),
            "a deadline-terminated search must still publish the work it finished"
        );
        assert_eq!(
            result.manifest.attempted_candidates,
            result.manifest.valid_candidates
                + result.manifest.duplicate_candidates
                + result.manifest.rejected_trivial_candidates
                + result.manifest.rejected_invalid_candidates
        );
    }

    #[test]
    fn shortlist_never_contains_a_provably_trivial_signal() {
        let result = run_search(&spec(2), None).unwrap();
        for candidate in result.candidates {
            let expression = crate::parse_expression(&candidate.expression).unwrap();
            assert_eq!(analyze_expression(&expression).rejection_reason, None);
        }
    }
}
