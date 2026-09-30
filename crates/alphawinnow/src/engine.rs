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
        FeedbackProvenance, MANIFEST_SCHEMA, OperatorPolicyProvenance, RunCheckpoint, RunManifest,
        candidate_content, read_candidates, read_checkpoint, write_checkpoint_atomic,
    },
    canonical::{canonical, digest, fingerprint},
    deduplicate_with_catalog, describe_with_catalog, descriptor_distance,
    feedback::{
        FeedbackConfig, FeedbackDataset, FeedbackError, GuidanceEstimate, OperatorPolicyConfig,
        estimate_candidate_guidance_prevalidated, reweight_operator_catalog,
    },
    operators::Catalog,
    parse_expression_with_catalog,
    transform::{
        CatalogScaffoldPolicy, MutationClass, Provenance, crossover_with_catalog,
        generate_with_catalog, generate_with_catalog_scaffold_policy,
        is_catalog_scaffold_operation, mutate_with_catalog, mutate_with_class_and_catalog,
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
    #[error("invalid measured feedback: {0}")]
    Feedback(#[from] FeedbackError),
    #[error("cannot seed the parent population from an empty accepted archive")]
    EmptySeedArchive,
    #[error("frozen seed parents require both archive seeding and measured feedback")]
    FrozenSeedParentsRequireGuidedArchive,
    #[error("feedback elite parents require measured feedback")]
    FeedbackEliteParentsRequireGuidance,
    #[error("feedback elite parent count exceeds the configured parent pool capacity")]
    FeedbackEliteParentCountExceedsCapacity,
    #[error("outcome-aware operator policy requires measured feedback")]
    OperatorPolicyRequiresGuidance,
    #[error("a separate action-evidence archive requires an outcome-aware operator policy")]
    ActionArchiveRequiresOperatorPolicy,
    #[error("cannot restore an accepted archive candidate: {0}")]
    SeedArchive(String),
}

#[derive(Debug, Clone)]
/// Immutable measured evidence and deterministic configuration used to guide search.
pub struct SearchGuidance {
    /// Versioned local measurements used by the guidance estimator.
    pub dataset: FeedbackDataset,
    /// Neighbor selection, distance, and confidence settings for guidance.
    pub config: FeedbackConfig,
}

#[derive(Debug, Clone, Default)]
pub struct SearchRunOptions {
    pub checkpoint_path: Option<PathBuf>,
    pub resume_from: Option<PathBuf>,
    pub pause_after_generations: Option<u64>,
    pub guidance: Option<SearchGuidance>,
    pub seed_from_archive: bool,
    pub freeze_seed_parents: bool,
    pub feedback_elite_parent_count: usize,
    pub operator_policy: Option<OperatorPolicyConfig>,
    pub operator_policy_action_archive: Option<PathBuf>,
    pub catalog_scaffold_policy: CatalogScaffoldPolicy,
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
    guidance: Option<GuidanceEstimate>,
}

struct GuidanceState<'a> {
    dataset: &'a FeedbackDataset,
    config: &'a FeedbackConfig,
    checksum: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
struct TransformActionPolicy {
    mutation_weight: u16,
    crossover_weight: u16,
    mutation_class_weights: [u16; 8],
}

impl TransformActionPolicy {
    fn audit_weights(&self) -> BTreeMap<String, u16> {
        let mut weights = BTreeMap::from([
            ("kind:crossover".to_owned(), self.crossover_weight),
            ("kind:mutation".to_owned(), self.mutation_weight),
        ]);
        for (class, weight) in MutationClass::ALL
            .into_iter()
            .zip(self.mutation_class_weights)
        {
            weights.insert(format!("mutation:{}", class.operation()), weight);
        }
        weights
    }
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

fn draft_from_candidate(value: &CandidateRecord, catalog: &Catalog) -> Result<Draft, SearchError> {
    let expression = parse_expression_with_catalog(&value.expression, catalog)
        .map_err(|error| SearchError::SeedArchive(error.to_string()))?;
    let analysis = analyze_expression(&expression);
    Ok(Draft {
        descriptor: describe_with_catalog(&expression, &value.provenance, catalog),
        expression,
        provenance: value.provenance.clone(),
        semantic_fingerprint: analysis.semantic_fingerprint,
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
    if options.freeze_seed_parents && (!options.seed_from_archive || options.guidance.is_none()) {
        return Err(SearchError::FrozenSeedParentsRequireGuidedArchive);
    }
    if options.feedback_elite_parent_count > 0 && options.guidance.is_none() {
        return Err(SearchError::FeedbackEliteParentsRequireGuidance);
    }
    if options.feedback_elite_parent_count > spec.resources.parent_pool_capacity {
        return Err(SearchError::FeedbackEliteParentCountExceedsCapacity);
    }
    if options.operator_policy.is_some() && options.guidance.is_none() {
        return Err(SearchError::OperatorPolicyRequiresGuidance);
    }
    if options.operator_policy_action_archive.is_some() && options.operator_policy.is_none() {
        return Err(SearchError::ActionArchiveRequiresOperatorPolicy);
    }
    let (effective_catalog, operator_policy_report) = if let Some(policy) = &options.operator_policy
    {
        let guidance = options
            .guidance
            .as_ref()
            .ok_or(SearchError::OperatorPolicyRequiresGuidance)?;
        let (adjusted, report) = reweight_operator_catalog(catalog, &guidance.dataset, policy)?;
        (adjusted, Some(report))
    } else {
        (catalog.clone(), None)
    };
    let catalog = &effective_catalog;
    catalog.validate().map_err(SearchError::Catalog)?;
    validate_search_catalog(catalog)?;
    let feedback_checksum = options
        .guidance
        .as_ref()
        .map(|guidance| {
            guidance.dataset.validate()?;
            guidance.config.validate()?;
            Ok::<_, FeedbackError>(guidance.dataset.checksum())
        })
        .transpose()?;
    let measured_feedback = options
        .guidance
        .as_ref()
        .zip(feedback_checksum.as_ref())
        .map(|(guidance, checksum)| FeedbackProvenance {
            dataset_id: guidance.dataset.dataset_id.clone(),
            context_checksum: guidance.dataset.context_checksum.clone(),
            feedback_checksum: checksum.clone(),
            configuration_checksum: digest(
                &serde_json::to_string(&guidance.config).unwrap_or_default(),
            ),
            outcome_label: guidance.dataset.outcome_label.clone(),
        });
    let guidance_state = options
        .guidance
        .as_ref()
        .zip(feedback_checksum.as_deref())
        .map(|(guidance, checksum)| GuidanceState {
            dataset: &guidance.dataset,
            config: &guidance.config,
            checksum,
        });
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
    let external_action_records = options
        .operator_policy_action_archive
        .as_deref()
        .map(read_candidates)
        .transpose()?;
    if let Some(records) = &external_action_records {
        if records.is_empty() {
            return Err(SearchError::EmptySeedArchive);
        }
        for record in records {
            draft_from_candidate(record, catalog)?;
        }
    }
    let action_records = external_action_records
        .as_deref()
        .unwrap_or(raw_archive_records.as_slice());
    let action_archive_checksum = external_action_records
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?
        .map(|content| digest(&content));
    let transform_action_policy = options
        .operator_policy
        .as_ref()
        .zip(options.guidance.as_ref())
        .map(|(config, guidance)| {
            transform_action_policy(action_records, &guidance.dataset, config)
        });
    let archive_record_count = raw_archive_records.len();
    let archive_partition = deduplicate_with_catalog(raw_archive_records, catalog)?;
    let archive_content_checksum = archive_state_checksum(&archive_partition.accepted);
    let archive_seed_drafts = if options.seed_from_archive {
        let seeds = archive_partition
            .accepted
            .iter()
            .map(|record| draft_from_candidate(record, catalog))
            .collect::<Result<Vec<_>, _>>()?;
        if seeds.is_empty() {
            return Err(SearchError::EmptySeedArchive);
        }
        seeds
    } else {
        Vec::new()
    };
    let mut feedback_identity = measured_feedback.as_ref().map_or_else(
        || "unguided".to_owned(),
        |provenance| digest(&serde_json::to_string(provenance).unwrap_or_default()),
    );
    if let Some(policy) = &transform_action_policy {
        feedback_identity.push_str(":actions=");
        feedback_identity.push_str(&digest(&serde_json::to_string(policy)?));
        if let Some(checksum) = &action_archive_checksum {
            feedback_identity.push_str(":action_archive=");
            feedback_identity.push_str(checksum);
        }
    }
    let run_identity = if options.feedback_elite_parent_count == 0 {
        format!(
            "alphawinnow:{candidate_identity_checksum}:{archive_content_checksum}:{feedback_identity}:seeded={}:frozen_seed_parents={}:scaffold={:?}",
            options.seed_from_archive, options.freeze_seed_parents, options.catalog_scaffold_policy,
        )
    } else {
        format!(
            "alphawinnow:{candidate_identity_checksum}:{archive_content_checksum}:{feedback_identity}:seeded={}:frozen_seed_parents={}:feedback_elite_parents={}:scaffold={:?}",
            options.seed_from_archive,
            options.freeze_seed_parents,
            options.feedback_elite_parent_count,
            options.catalog_scaffold_policy,
        )
    };
    let run_id = digest(&run_identity);
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
            measured_feedback.as_ref(),
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
    } else if options.seed_from_archive {
        population = archive_seed_drafts;
        false
    } else {
        let initial_count = usize::try_from(spec.max_candidates.min(INITIAL_POPULATION as u64))
            .unwrap_or(INITIAL_POPULATION);
        let mut initial = pool.install(|| {
            (0..initial_count)
                .into_par_iter()
                .map(|slot| {
                    let ordinal = u64::try_from(slot).unwrap_or(u64::MAX);
                    let (expression, provenance) = generate_with_catalog_scaffold_policy(
                        spec.seed,
                        ordinal,
                        &spec.limits,
                        catalog,
                        options.catalog_scaffold_policy,
                    );
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
    let mut initial_feedback_elites = Vec::new();
    if !resumed {
        let scored_population = score_drafts(
            &pool,
            &population,
            &archive_descriptors,
            &spec.scoring,
            guidance_state.as_ref(),
        );
        initial_feedback_elites =
            feedback_elite_drafts(&scored_population, options.feedback_elite_parent_count);
        population = diverse_elites(scored_population, spec.resources.population_capacity);
    }
    let mut parents: Vec<_> = restored_parents.unwrap_or_else(|| {
        select_parent_pool(
            &initial_feedback_elites,
            &population,
            spec.resources.parent_pool_capacity,
        )
    });
    peak_population = peak_population.max(population.len());
    persist_checkpoint(
        options.checkpoint_path.as_deref(),
        &run_id,
        &config_checksum,
        &operator_catalog_checksum,
        measured_feedback.as_ref(),
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
                    let (expression, provenance) = offspring(
                        spec,
                        generation,
                        slot,
                        ordinal,
                        &parents,
                        catalog,
                        transform_action_policy.as_ref(),
                    );
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
        let scored_population = score_drafts(
            &pool,
            &population,
            &archive_descriptors,
            &spec.scoring,
            guidance_state.as_ref(),
        );
        let feedback_elites =
            feedback_elite_drafts(&scored_population, options.feedback_elite_parent_count);
        population = diverse_elites(scored_population, spec.resources.population_capacity);
        if !options.freeze_seed_parents {
            parents = select_parent_pool(
                &feedback_elites,
                &population,
                spec.resources.parent_pool_capacity,
            );
        }
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
            measured_feedback.as_ref(),
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
    let scored = score_drafts(
        &pool,
        final_drafts,
        &archive_descriptors,
        &spec.scoring,
        guidance_state.as_ref(),
    );
    let guided_population_candidates = scored
        .iter()
        .filter(|candidate| candidate.guidance.is_some())
        .count() as u64;
    let guidance_scores = scored
        .iter()
        .filter_map(|candidate| {
            candidate.guidance.as_ref().map(|guidance| {
                (
                    candidate.draft.semantic_fingerprint.clone(),
                    guidance.conservative_outcome,
                )
            })
        })
        .collect::<HashMap<_, _>>();
    let mut records = pool.install(|| {
        scored
            .par_iter()
            .map(|candidate| candidate_record(candidate, &run_id))
            .collect::<Vec<_>>()
    });
    sort_records_with_guidance(&mut records, &guidance_scores);
    let shortlist_deadline = started + deadline.saturating_sub(Duration::from_millis(25));
    let candidates = diverse_shortlist(
        records,
        spec.shortlist_size,
        Some(shortlist_deadline),
        &guidance_scores,
    );
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
        measured_feedback: measured_feedback.clone(),
        operator_policy: operator_policy_report
            .as_ref()
            .map(|report| OperatorPolicyProvenance {
                configuration_checksum: digest(
                    &serde_json::to_string(options.operator_policy.as_ref().unwrap_or_else(|| {
                        unreachable!("a report exists only for a configured operator policy")
                    }))
                    .unwrap_or_default(),
                ),
                base_catalog_checksum: report.base_catalog_checksum.clone(),
                adjusted_catalog_checksum: report.adjusted_catalog_checksum.clone(),
                generation_weights: report.generation_weights.clone(),
                transform_action_weights: transform_action_policy
                    .as_ref()
                    .map_or_else(BTreeMap::new, TransformActionPolicy::audit_weights),
                transform_action_source: if action_archive_checksum.is_some() {
                    "external_action_archive".to_owned()
                } else {
                    "parent_archive".to_owned()
                },
                transform_action_archive_checksum: action_archive_checksum.clone(),
            }),
        guided_population_candidates,
        seeded_from_archive: options.seed_from_archive,
        frozen_seed_parents: options.freeze_seed_parents,
        feedback_elite_parent_count: options.feedback_elite_parent_count,
        catalog_scaffold_policy: options.catalog_scaffold_policy,
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
        measured_feedback.as_ref(),
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
        if !arity_and_inputs_compatible(operator, reference)
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

fn arity_and_inputs_compatible(
    configured: &crate::operators::OperatorSpec,
    reference: &crate::operators::OperatorSpec,
) -> bool {
    if configured.arity == reference.arity {
        return inputs_compatible(&configured.inputs, &reference.inputs);
    }
    let configured_count = match configured.arity {
        crate::Arity::Binary => 2,
        crate::Arity::Exact { count } => count,
        crate::Arity::Unary | crate::Arity::Variadic { .. } => return false,
    };
    let crate::Arity::Variadic { min, max } = reference.arity else {
        return false;
    };
    (min..=max).contains(&configured_count)
        && reference.inputs.len() == 1
        && configured.inputs.len() == configured_count
        && configured
            .inputs
            .iter()
            .all(|input| input_compatible(input, &reference.inputs[0]))
}

fn inputs_compatible(
    configured: &[crate::operators::ArgumentSpec],
    reference: &[crate::operators::ArgumentSpec],
) -> bool {
    configured.len() == reference.len()
        && configured
            .iter()
            .zip(reference)
            .all(|(configured, reference)| input_compatible(configured, reference))
}

fn input_compatible(
    configured: &crate::operators::ArgumentSpec,
    reference: &crate::operators::ArgumentSpec,
) -> bool {
    configured.kinds == reference.kinds
        && match (&configured.domain, &reference.domain) {
            (left, right) if left == right => true,
            (crate::ValueDomain::WindowSet { values }, crate::ValueDomain::Window { min, max }) => {
                values.iter().all(|value| (*min..=*max).contains(value))
            }
            _ => false,
        }
}

fn validate_checkpoint(
    checkpoint: &RunCheckpoint,
    run_id: &str,
    config_checksum: &str,
    operator_catalog_checksum: &str,
    measured_feedback: Option<&FeedbackProvenance>,
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
        || checkpoint.measured_feedback.as_ref() != measured_feedback
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
    measured_feedback: Option<&FeedbackProvenance>,
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
        measured_feedback: measured_feedback.cloned(),
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

fn transform_action_policy(
    archive: &[CandidateRecord],
    feedback: &FeedbackDataset,
    config: &OperatorPolicyConfig,
) -> TransformActionPolicy {
    let evidence = feedback
        .records
        .iter()
        .filter_map(|record| {
            record
                .semantic_fingerprint
                .as_ref()
                .map(|fingerprint| (fingerprint.as_str(), (record.outcome, record.confidence)))
        })
        .collect::<HashMap<_, _>>();
    let global_weight = evidence
        .values()
        .map(|(_, confidence)| confidence)
        .sum::<f64>();
    let global_outcome = evidence
        .values()
        .map(|(outcome, confidence)| outcome * confidence)
        .sum::<f64>()
        / global_weight;
    let adjusted = |name: &str, base: u16, matches: &dyn Fn(&CandidateRecord) -> bool| {
        let measured = archive
            .iter()
            .filter(|record| matches(record))
            .filter_map(|record| evidence.get(record.semantic_fingerprint.as_str()))
            .collect::<Vec<_>>();
        let centered = if measured.len() >= config.minimum_support {
            let denominator = measured
                .iter()
                .map(|(_, confidence)| confidence)
                .sum::<f64>();
            measured
                .iter()
                .map(|(outcome, confidence)| outcome * confidence)
                .sum::<f64>()
                / denominator
                - global_outcome
        } else {
            0.0
        };
        let normalized = (centered * 20.0).clamp(-1.0, 1.0);
        let multiplier = (1.0 + config.strength * normalized)
            .clamp(config.exploration_floor, config.maximum_multiplier);
        let bounded = (f64::from(base) * f64::from(config.weight_scale) * multiplier)
            .round()
            .clamp(1.0, f64::from(u16::MAX));
        format!("{bounded:.0}")
            .parse::<u16>()
            .unwrap_or_else(|_| panic!("bounded action weight for {name} must fit u16"))
    };
    let mutation_weight = adjusted("mutation", 1, &|record| {
        record.provenance.kind == crate::TransformKind::Mutation
    });
    let crossover_weight = adjusted("crossover", 2, &|record| {
        record.provenance.kind == crate::TransformKind::Crossover
    });
    let mutation_class_weights = MutationClass::ALL.map(|class| {
        adjusted(class.operation(), 1, &|record| {
            record.provenance.kind == crate::TransformKind::Mutation
                && record.provenance.operation == class.operation()
        })
    });
    TransformActionPolicy {
        mutation_weight,
        crossover_weight,
        mutation_class_weights,
    }
}

fn weighted_index(seed: u64, weights: &[u16]) -> usize {
    let total = weights.iter().map(|weight| u64::from(*weight)).sum::<u64>();
    let mut draw = seed % total;
    weights
        .iter()
        .position(|weight| {
            if draw < u64::from(*weight) {
                true
            } else {
                draw -= u64::from(*weight);
                false
            }
        })
        .unwrap_or(0)
}

fn offspring(
    spec: &SearchSpec,
    generation: u64,
    slot: u64,
    ordinal: u64,
    parents: &[Draft],
    catalog: &Catalog,
    action_policy: Option<&TransformActionPolicy>,
) -> (crate::Expr, Provenance) {
    if parents.is_empty() {
        let seed = offspring_seed(spec.seed, generation, slot, &[], "independent");
        return generate_with_catalog(seed, ordinal, &spec.limits, catalog);
    }
    let mutation = action_policy.map_or_else(
        || slot.is_multiple_of(3),
        |policy| {
            parents.len() == 1
                || weighted_index(
                    offspring_seed(spec.seed, generation, slot, &[], "action-kind"),
                    &[policy.mutation_weight, policy.crossover_weight],
                ) == 0
        },
    );
    if mutation || parents.len() == 1 {
        let index = parent_index(slot, 17, generation, parents.len());
        let parent = &parents[index];
        let seed = offspring_seed(
            spec.seed,
            generation,
            slot,
            std::slice::from_ref(&parent.semantic_fingerprint),
            "mutation",
        );
        if let Some(policy) = action_policy {
            let class_index = weighted_index(
                offspring_seed(spec.seed, generation, slot, &[], "mutation-class"),
                &policy.mutation_class_weights,
            );
            mutate_with_class_and_catalog(
                &parent.expression,
                MutationClass::ALL[class_index],
                seed,
                ordinal,
                &spec.limits,
                catalog,
            )
        } else {
            mutate_with_catalog(&parent.expression, seed, ordinal, &spec.limits, catalog)
        }
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
    guidance: Option<&GuidanceState<'_>>,
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
                let mut scored = ScoredDraft {
                    draft: draft.clone(),
                    structural_score,
                    components,
                    weights,
                    guidance: None,
                };
                if let Some(guidance) = guidance {
                    let candidate = candidate_record(&scored, "guidance-estimate");
                    scored.guidance = estimate_candidate_guidance_prevalidated(
                        &candidate,
                        guidance.dataset,
                        guidance.config,
                        guidance.checksum,
                    );
                }
                scored
            })
            .collect::<Vec<_>>()
    });
    scored.sort_by(compare_scored);
    scored
}

fn compare_scored(left: &ScoredDraft, right: &ScoredDraft) -> std::cmp::Ordering {
    match (&left.guidance, &right.guidance) {
        (Some(left), Some(right)) => right
            .conservative_outcome
            .total_cmp(&left.conservative_outcome)
            .then_with(|| right.weighted_outcome.total_cmp(&left.weighted_outcome)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
    .then_with(|| right.structural_score.total_cmp(&left.structural_score))
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
    queues.sort_by(|left, right| match (left.front(), right.front()) {
        (Some(left), Some(right)) => compare_scored(left, right),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
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

fn feedback_elite_drafts(scored: &[ScoredDraft], count: usize) -> Vec<Draft> {
    scored
        .iter()
        .filter(|candidate| candidate.guidance.is_some())
        .take(count)
        .map(|candidate| candidate.draft.clone())
        .collect()
}

fn select_parent_pool(
    feedback_elites: &[Draft],
    diverse_population: &[Draft],
    capacity: usize,
) -> Vec<Draft> {
    let mut selected = Vec::with_capacity(capacity);
    let mut seen = HashSet::with_capacity(capacity);
    for draft in feedback_elites.iter().chain(diverse_population) {
        if selected.len() == capacity {
            break;
        }
        if seen.insert(draft.semantic_fingerprint.clone()) {
            selected.push(draft.clone());
        }
    }
    selected
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

fn sort_records_with_guidance(
    records: &mut [CandidateRecord],
    guidance_scores: &HashMap<String, f64>,
) {
    records.sort_by(|left, right| {
        match (
            guidance_scores.get(&left.semantic_fingerprint),
            guidance_scores.get(&right.semantic_fingerprint),
        ) {
            (Some(left), Some(right)) => right.total_cmp(left),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| right.structural_score.total_cmp(&left.structural_score))
        .then_with(|| left.semantic_fingerprint.cmp(&right.semantic_fingerprint))
    });
}

fn diverse_shortlist(
    records: Vec<CandidateRecord>,
    limit: usize,
    deadline: Option<Instant>,
    guidance_scores: &HashMap<String, f64>,
) -> Vec<CandidateRecord> {
    let (mut scaffold, remaining): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|record| is_catalog_scaffold_operation(&record.provenance.operation));
    sort_records_with_guidance(&mut scaffold, guidance_scores);
    let scaffold_reserve = limit.div_ceil(2).min(scaffold.len());
    let mut result = scaffold.drain(..scaffold_reserve).collect::<Vec<_>>();
    let mut families = BTreeMap::<(crate::TransformKind, u8), Vec<CandidateRecord>>::new();
    // The reserved cover is a fixed evaluator-budget slice.  Do not let its
    // unused tail re-enter the diversity rotation and crowd out learned or
    // transformed candidates when the requested shortlist is small.
    for record in remaining {
        families
            .entry((record.provenance.kind, node_bucket(record.nodes)))
            .or_default()
            .push(record);
    }
    let mut budget_exhausted = false;
    while result.len() < limit {
        if deadline.is_some_and(|value| Instant::now() >= value) {
            budget_exhausted = true;
            break;
        }
        let mut progressed = false;
        for records in families.values_mut() {
            if !records.is_empty() && result.len() < limit {
                let index = descriptor_aware_choice(records, &result, guidance_scores);
                result.push(records.remove(index));
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    if budget_exhausted {
        fill_from_score_order(&mut result, families, limit, guidance_scores);
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
    guidance_scores: &HashMap<String, f64>,
) {
    if result.len() >= limit {
        return;
    }
    let mut remaining = families.into_values().flatten().collect::<Vec<_>>();
    sort_records_with_guidance(&mut remaining, guidance_scores);
    result.extend(remaining.into_iter().take(limit - result.len()));
}

fn descriptor_aware_choice(
    records: &[CandidateRecord],
    selected: &[CandidateRecord],
    guidance_scores: &HashMap<String, f64>,
) -> usize {
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
            let measured = guidance_scores.get(&record.semantic_fingerprint).copied();
            let score = measured.map_or_else(
                || record.structural_score + 0.35 * novelty,
                |outcome| outcome + 0.05 * novelty,
            );
            (index, measured.is_some(), score)
        })
        .max_by(
            |(left_index, left_guided, left), (right_index, right_guided, right)| {
                left_guided
                    .cmp(right_guided)
                    .then_with(|| left.total_cmp(right))
                    .then_with(|| right_index.cmp(left_index))
            },
        )
        .map_or(0, |(index, _, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FeedbackRecord, Limits, SEARCH_SPEC_SCHEMA, StructuralScoreConfig, parse_expression,
    };

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
            1_000,
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

    fn search_guidance() -> SearchGuidance {
        SearchGuidance {
            dataset: FeedbackDataset {
                schema: crate::FEEDBACK_DATASET_SCHEMA,
                dataset_id: "synthetic-search-guidance-v1".to_owned(),
                context_checksum: "fixture-context".to_owned(),
                outcome_label: "synthetic utility".to_owned(),
                feature_scales: BTreeMap::from([
                    ("depth".to_owned(), 10.0),
                    ("nodes".to_owned(), 40.0),
                ]),
                records: vec![
                    FeedbackRecord {
                        record_id: "small".to_owned(),
                        semantic_fingerprint: None,
                        features: BTreeMap::from([
                            ("depth".to_owned(), 1.0),
                            ("nodes".to_owned(), 1.0),
                        ]),
                        outcome: -0.5,
                        confidence: 1.0,
                    },
                    FeedbackRecord {
                        record_id: "large".to_owned(),
                        semantic_fingerprint: None,
                        features: BTreeMap::from([
                            ("depth".to_owned(), 6.0),
                            ("nodes".to_owned(), 30.0),
                        ]),
                        outcome: 0.5,
                        confidence: 1.0,
                    },
                ],
            },
            config: FeedbackConfig {
                schema: 1,
                neighbors: 1,
                minimum_neighbors: 1,
                maximum_distance: 1.0,
            },
        }
    }

    fn operator_policy_guidance() -> SearchGuidance {
        let rows = [
            ("mean-a", 0.0, 1.0, 0.8),
            ("mean-b", 0.0, 1.0, 0.7),
            ("mean-c", 0.0, 2.0, 0.6),
            ("rank-a", 1.0, 0.0, -0.8),
            ("rank-b", 1.0, 0.0, -0.7),
            ("rank-c", 2.0, 0.0, -0.6),
        ];
        SearchGuidance {
            dataset: FeedbackDataset {
                schema: crate::FEEDBACK_DATASET_SCHEMA,
                dataset_id: "synthetic-operator-policy-v1".to_owned(),
                context_checksum: "operator-policy-context".to_owned(),
                outcome_label: "synthetic operator reward".to_owned(),
                feature_scales: BTreeMap::from([
                    ("operator_rank_count".to_owned(), 1.0),
                    ("operator_ts_mean_count".to_owned(), 1.0),
                ]),
                records: rows
                    .into_iter()
                    .map(|(id, rank, mean, outcome)| FeedbackRecord {
                        record_id: id.to_owned(),
                        semantic_fingerprint: None,
                        features: BTreeMap::from([
                            ("operator_rank_count".to_owned(), rank),
                            ("operator_ts_mean_count".to_owned(), mean),
                        ]),
                        outcome,
                        confidence: 1.0,
                    })
                    .collect(),
            },
            config: FeedbackConfig::default(),
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
    fn guided_search_is_deterministic_and_records_feedback_identity() {
        let options = SearchRunOptions {
            guidance: Some(search_guidance()),
            ..SearchRunOptions::default()
        };
        let left = run_search_with_options(&spec(1), None, &options).unwrap();
        let right = run_search_with_options(&spec(4), None, &options).unwrap();

        assert_eq!(left.candidates, right.candidates);
        assert_eq!(
            left.manifest.candidate_content_checksum,
            right.manifest.candidate_content_checksum
        );
        let provenance = left.manifest.measured_feedback.unwrap();
        assert_eq!(provenance.dataset_id, "synthetic-search-guidance-v1");
        assert_eq!(
            provenance.feedback_checksum,
            search_guidance().dataset.checksum()
        );
        assert!(left.manifest.guided_population_candidates > 0);
    }

    #[test]
    fn outcome_aware_operator_policy_is_opt_in_deterministic_and_audited() {
        let guidance = operator_policy_guidance();
        let baseline = run_search_with_options(
            &spec(1),
            None,
            &SearchRunOptions {
                guidance: Some(guidance.clone()),
                ..SearchRunOptions::default()
            },
        )
        .unwrap();
        let options = SearchRunOptions {
            guidance: Some(guidance),
            operator_policy: Some(OperatorPolicyConfig::default()),
            ..SearchRunOptions::default()
        };
        let one_thread = run_search_with_options(&spec(1), None, &options).unwrap();
        let four_threads = run_search_with_options(&spec(4), None, &options).unwrap();

        assert_eq!(one_thread.candidates, four_threads.candidates);
        assert_ne!(
            baseline.manifest.candidate_content_checksum,
            one_thread.manifest.candidate_content_checksum
        );
        let provenance = one_thread.manifest.operator_policy.unwrap();
        assert_ne!(
            provenance.base_catalog_checksum,
            provenance.adjusted_catalog_checksum
        );
        assert_eq!(
            provenance.adjusted_catalog_checksum,
            one_thread.manifest.operator_catalog_checksum
        );
    }

    #[test]
    fn outcome_aware_operator_policy_requires_feedback() {
        let error = run_search_with_options(
            &spec(1),
            None,
            &SearchRunOptions {
                operator_policy: Some(OperatorPolicyConfig::default()),
                ..SearchRunOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, SearchError::OperatorPolicyRequiresGuidance));
    }

    #[test]
    fn seeded_catalog_scaffold_is_opt_in_and_recorded() {
        let options = SearchRunOptions {
            catalog_scaffold_policy: CatalogScaffoldPolicy::SeededDiverse,
            ..SearchRunOptions::default()
        };
        let result = run_search_with_options(&spec(2), None, &options).unwrap();
        assert_eq!(
            result.manifest.catalog_scaffold_policy,
            CatalogScaffoldPolicy::SeededDiverse
        );
        assert!(result.candidates.iter().any(|candidate| {
            candidate.provenance.operation == crate::SEEDED_CATALOG_SCAFFOLD_OPERATION
        }));
    }

    #[test]
    fn high_ranked_candidates_receive_selectable_descendants() {
        let candidates = vec![draft("close"), draft("rank(ts_mean(add(open, high), 20))")];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let scored = score_drafts(
            &pool,
            &candidates,
            &[],
            &StructuralScoreConfig::default(),
            None,
        );
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
                None,
            );
            if provenance.parent_fingerprints.contains(&highest) {
                found_descendant = true;
                break;
            }
        }
        assert!(found_descendant);
    }

    #[test]
    fn measured_feedback_overrides_structural_parent_order_without_changing_structural_scores() {
        let candidates = vec![draft("close"), draft("rank(ts_mean(add(open, high), 20))")];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let structural = score_drafts(
            &pool,
            &candidates,
            &[],
            &StructuralScoreConfig::default(),
            None,
        );
        assert_eq!(structural[0].draft.expression.operator(), None);
        let structural_scores = structural
            .iter()
            .map(|candidate| {
                (
                    candidate.draft.semantic_fingerprint.clone(),
                    candidate.structural_score,
                )
            })
            .collect::<HashMap<_, _>>();
        let feedback_features = |candidate: &ScoredDraft| {
            let record = candidate_record(candidate, "feedback-fixture");
            let features = crate::candidate_features(&record);
            BTreeMap::from([
                ("depth".to_owned(), features["depth"]),
                ("nodes".to_owned(), features["nodes"]),
            ])
        };
        let simple = structural
            .iter()
            .find(|candidate| candidate.draft.expression.operator().is_none())
            .unwrap();
        let rich = structural
            .iter()
            .find(|candidate| candidate.draft.expression.operator() == Some("rank"))
            .unwrap();
        let dataset = FeedbackDataset {
            schema: crate::FEEDBACK_DATASET_SCHEMA,
            dataset_id: "synthetic-parent-guidance-v1".to_owned(),
            context_checksum: "fixture-context".to_owned(),
            outcome_label: "synthetic utility".to_owned(),
            feature_scales: BTreeMap::from([
                ("depth".to_owned(), 10.0),
                ("nodes".to_owned(), 40.0),
            ]),
            records: vec![
                FeedbackRecord {
                    record_id: "simple".to_owned(),
                    semantic_fingerprint: None,
                    features: feedback_features(simple),
                    outcome: -0.8,
                    confidence: 1.0,
                },
                FeedbackRecord {
                    record_id: "rich".to_owned(),
                    semantic_fingerprint: None,
                    features: feedback_features(rich),
                    outcome: 0.8,
                    confidence: 1.0,
                },
            ],
        };
        let config = FeedbackConfig {
            schema: 1,
            neighbors: 1,
            minimum_neighbors: 1,
            maximum_distance: 0.0,
        };
        dataset.validate().unwrap();
        config.validate().unwrap();
        let checksum = dataset.checksum();
        let guidance = GuidanceState {
            dataset: &dataset,
            config: &config,
            checksum: &checksum,
        };
        let guided = score_drafts(
            &pool,
            &candidates,
            &[],
            &StructuralScoreConfig::default(),
            Some(&guidance),
        );

        assert_eq!(guided[0].draft.expression.operator(), Some("rank"));
        assert!((guided[0].guidance.as_ref().unwrap().weighted_outcome - 0.8).abs() < f64::EPSILON);
        assert!(guided.iter().all(|candidate| {
            (candidate.structural_score - structural_scores[&candidate.draft.semantic_fingerprint])
                .abs()
                < f64::EPSILON
        }));
        let parents = diverse_elites(guided, 2);
        assert_eq!(parents[0].expression.operator(), Some("rank"));
    }

    #[test]
    fn feedback_elite_parent_reservation_precedes_diverse_fill_without_duplicates() {
        let close = draft("close");
        let open = draft("open");
        let high = draft("high");
        let low = draft("low");
        let parents = select_parent_pool(
            &[high.clone(), low.clone()],
            &[close.clone(), high, open.clone()],
            4,
        );
        assert_eq!(
            parents
                .iter()
                .map(|parent| canonical(&parent.expression))
                .collect::<Vec<_>>(),
            vec!["high", "low", "close", "open"]
        );
    }

    #[test]
    fn feedback_elite_parent_reservation_requires_guidance_and_fits_capacity() {
        let mut configured = spec(1);
        configured.resources.parent_pool_capacity = 2;
        let without_guidance = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                feedback_elite_parent_count: 1,
                ..SearchRunOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            without_guidance,
            SearchError::FeedbackEliteParentsRequireGuidance
        ));

        let over_capacity = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                guidance: Some(search_guidance()),
                feedback_elite_parent_count: 3,
                ..SearchRunOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            over_capacity,
            SearchError::FeedbackEliteParentCountExceedsCapacity
        ));
    }

    #[test]
    fn guided_search_records_feedback_elite_parent_reservation() {
        let options = SearchRunOptions {
            guidance: Some(search_guidance()),
            feedback_elite_parent_count: 4,
            ..SearchRunOptions::default()
        };
        let result = run_search_with_options(&spec(1), None, &options).unwrap();
        assert_eq!(result.manifest.feedback_elite_parent_count, 4);
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
            None,
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
    fn shortlist_reserves_half_its_budget_for_catalog_scaffolds() {
        let mut configured = spec(2);
        configured.shortlist_size = 128;
        let result = run_search(&configured, None).unwrap();
        let scaffold = result
            .candidates
            .iter()
            .filter(|candidate| candidate.provenance.operation == crate::CATALOG_SCAFFOLD_OPERATION)
            .collect::<Vec<_>>();
        assert_eq!(scaffold.len(), 64);
        let expressions = scaffold
            .iter()
            .map(|candidate| candidate.expression.as_str())
            .collect::<HashSet<_>>();
        for expression in [
            "close",
            "open",
            "multiply(close, close)",
            "multiply(open, open)",
        ] {
            assert!(expressions.contains(expression), "missing {expression}");
        }
    }

    #[test]
    fn shortlist_preserves_competitive_node_buckets_within_a_transform() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let seed = draft("close");
        let scored = score_drafts(&pool, &[seed], &[], &StructuralScoreConfig::default(), None);
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

        let retained = diverse_shortlist(records, 6, None, &HashMap::new());
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
        let scored = score_drafts(&pool, &[seed], &[], &StructuralScoreConfig::default(), None);
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
        let retained = diverse_shortlist(records.clone(), 5, Some(expired), &HashMap::new());

        assert_eq!(retained.len(), 5);
        let mut expected = records;
        sort_records_with_guidance(&mut expected, &HashMap::new());
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
            diverse_shortlist(Vec::new(), 5, Some(expired), &HashMap::new()),
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
    fn external_catalog_may_narrow_a_variadic_operator_to_fixed_arity() {
        let mut catalog = crate::builtin_catalog().clone();
        for name in ["add", "multiply"] {
            let operator = catalog
                .operators
                .iter_mut()
                .find(|operator| operator.name == name)
                .unwrap();
            let input = operator.inputs[0].clone();
            operator.arity = crate::Arity::Binary;
            operator.inputs = vec![input; 2];
        }

        validate_search_catalog(&catalog).unwrap();
        let result = run_search_with_catalog(&spec(1), None, &catalog).unwrap();
        assert!(result.candidates.iter().all(|candidate| {
            parse_expression_with_catalog(&candidate.expression, &catalog).is_ok()
        }));

        let add = catalog
            .operators
            .iter_mut()
            .find(|operator| operator.name == "add")
            .unwrap();
        add.arity = crate::Arity::Exact { count: 9 };
        add.inputs = vec![add.inputs[0].clone(); 9];
        assert!(validate_search_catalog(&catalog).is_err());
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
    fn archive_seeding_generates_new_descendants_from_measured_parent_candidates() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("archive.jsonl");
        let initial = run_search(&spec(1), None).unwrap();
        crate::write_jsonl_atomic(&archive_path, &initial.candidates).unwrap();
        let archive_ids = initial
            .candidates
            .iter()
            .map(|candidate| candidate.semantic_fingerprint.clone())
            .collect::<HashSet<_>>();
        let mut continuation = spec(1);
        continuation.max_candidates = 64;
        let result = run_search_with_options(
            &continuation,
            Some(&archive_path),
            &SearchRunOptions {
                seed_from_archive: true,
                ..SearchRunOptions::default()
            },
        )
        .unwrap();

        assert!(result.manifest.seeded_from_archive);
        assert_eq!(result.manifest.attempted_candidates, 64);
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| !archive_ids.contains(&candidate.semantic_fingerprint))
        );
        assert!(result.candidates.iter().any(|candidate| {
            candidate
                .provenance
                .parent_fingerprints
                .iter()
                .any(|parent| archive_ids.contains(parent))
        }));
    }

    #[test]
    fn frozen_seed_parents_keep_every_descendant_attached_to_the_measured_archive() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("archive.jsonl");
        let initial = run_search(&spec(1), None).unwrap();
        crate::write_jsonl_atomic(&archive_path, &initial.candidates).unwrap();
        let archive_fingerprints = initial
            .candidates
            .iter()
            .map(|candidate| candidate.semantic_fingerprint.clone())
            .collect::<HashSet<_>>();
        let mut continuation = spec(1);
        continuation.max_candidates = 512;
        let result = run_search_with_options(
            &continuation,
            Some(&archive_path),
            &SearchRunOptions {
                guidance: Some(search_guidance()),
                seed_from_archive: true,
                freeze_seed_parents: true,
                ..SearchRunOptions::default()
            },
        )
        .unwrap();

        assert!(result.manifest.seeded_from_archive);
        assert!(result.manifest.frozen_seed_parents);
        assert!(
            result
                .candidates
                .iter()
                .any(|candidate| { !candidate.provenance.parent_fingerprints.is_empty() })
        );
        assert!(result.candidates.iter().all(|candidate| {
            candidate
                .provenance
                .parent_fingerprints
                .iter()
                .all(|parent| archive_fingerprints.contains(parent))
        }));
    }

    #[test]
    fn frozen_seed_parents_fail_closed_without_a_guided_archive() {
        let error = run_search_with_options(
            &spec(1),
            None,
            &SearchRunOptions {
                freeze_seed_parents: true,
                ..SearchRunOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            SearchError::FrozenSeedParentsRequireGuidedArchive
        ));
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_fixed_budget() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("search.checkpoint.json");
        let mut configured = spec(2);
        configured.max_candidates = 1_000;
        configured.resources.elite_archive_capacity = 200;
        let guidance = search_guidance();
        let uninterrupted = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                guidance: Some(guidance.clone()),
                ..SearchRunOptions::default()
            },
        )
        .unwrap();
        let paused = run_search_with_options(
            &configured,
            None,
            &SearchRunOptions {
                checkpoint_path: Some(checkpoint.clone()),
                resume_from: None,
                pause_after_generations: Some(3),
                guidance: Some(guidance.clone()),
                seed_from_archive: false,
                freeze_seed_parents: false,
                feedback_elite_parent_count: 0,
                operator_policy: None,
                operator_policy_action_archive: None,
                catalog_scaffold_policy: CatalogScaffoldPolicy::Fixed,
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
                guidance: Some(guidance),
                seed_from_archive: false,
                freeze_seed_parents: false,
                feedback_elite_parent_count: 0,
                operator_policy: None,
                operator_policy_action_archive: None,
                catalog_scaffold_policy: CatalogScaffoldPolicy::Fixed,
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
