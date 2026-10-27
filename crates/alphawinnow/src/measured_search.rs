//! Bounded ask/tell evolution driven only by externally measured outcomes.
//!
//! Each `tell` supplies a complete score snapshot for the retained parents and
//! pending batch under one evaluator and reference-pool context. This prevents
//! stale marginal contributions from competing with scores for a different pool.
//! Opt-in `LatestBatchScoreV1` instead scores only the pending cohort and retires
//! all incumbents: callers may rank historical admission events within a batch
//! without treating their gains as current-pool marginal contributions.
//! `LatestBatchUniformV1` is its score-blind, same-cohort retention control.
//! Indexed random streams make batch order and checkpoints deterministic.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::Arc,
};

use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Catalog, ExprKind, Limits, Provenance, analyze_expression, canonical, crossover_with_catalog,
    generate_with_catalog, mutate_applicable_with_catalog, mutate_with_catalog,
    parse_expression_with_catalog, sample_with_catalog, semantic_fingerprint,
    validate_transformed_with_catalog,
};

/// Schema for direct measured-search checkpoints.
pub const MEASURED_SEARCH_SCHEMA: u32 = 1;

/// Checkpoints with measured bootstrap parents, exclusions or screened batches.
pub const BOOTSTRAPPED_SEARCH_SCHEMA: u32 = 2;

/// Envelope for an opt-in checkpoint referring to separately persisted history.
pub const REFERENCED_SEARCH_SCHEMA: u32 = 3;

/// Immutable, validated exclusion archive, shared cheaply between restored sessions.
/// Persist `to_json()` once; its checksum hashes the canonical sorted JSON array.
/// The digest detects accidental archive mismatch, not malicious tampering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedHistory {
    entries: Arc<BTreeSet<String>>,
    checksum: String,
}

impl ExcludedHistory {
    /// Validate up to 100,000 lowercase SHA-256 semantic identities.
    ///
    /// # Errors
    /// Rejects invalid fingerprints or an excessive archive.
    pub fn new(entries: BTreeSet<String>) -> Result<Self, MeasuredSearchError> {
        ensure(
            entries.len() <= 100_000
                && entries.iter().all(|s| {
                    s.len() == 64
                        && s.bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                }),
            "invalid historical semantic fingerprints",
        )?;
        let checksum = crate::canonical::digest(&serde_json::to_string(&entries)?);
        Ok(Self {
            entries: Arc::new(entries),
            checksum,
        })
    }

    /// Read a bounded canonicalizable JSON array; validate/hash it once on load.
    ///
    /// # Errors
    /// Rejects invalid JSON, fingerprints, or input over 8 MiB.
    pub fn from_json(content: &str) -> Result<Self, MeasuredSearchError> {
        ensure(content.len() <= 8 * 1024 * 1024, "history exceeds 8 MiB")?;
        Ok(serde_json::from_str(content)?)
    }

    /// Serialize the standalone history array for caller-owned durable storage.
    ///
    /// # Errors
    /// Returns a JSON serialization error.
    pub fn to_json(&self) -> Result<String, MeasuredSearchError> {
        Ok(serde_json::to_string(self)?)
    }

    #[must_use]
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
}

impl Serialize for ExcludedHistory {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.entries.as_ref().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ExcludedHistory {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries = BTreeSet::<String>::deserialize(deserializer)?;
        Self::new(entries).map_err(serde::de::Error::custom)
    }
}

/// Data-independent policy used for fresh exploration proposals.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasuredExploration {
    /// Historical generator, including its fixed catalog-scaffold prefix.
    #[default]
    FixedCover,
    /// Seeded typed-grammar sampling from the first proposal; no cover prefix.
    Grammar,
}

/// Explicit mutation policy; missing checkpoint fields retain legacy behavior.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasuredMutation {
    #[default]
    LegacyUniform,
    Applicable,
}

impl MeasuredMutation {
    // serde's skip_serializing_if callback receives a reference.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_legacy(&self) -> bool {
        *self == Self::LegacyUniform
    }
}

/// Parent retention policy. Uniform retention is a score-blind research control.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasuredParentSelection {
    #[default]
    Score,
    /// Uniform subset of incumbents plus pending proposals, ordered by ordinal.
    Uniform,
    /// Preserve the top half by score, then prefer nonnegative, structurally
    /// distant parents (distance >= 0.15); score-ordered fallback fills capacity.
    /// Lineage is ignored. Fixed v1 parameters are part of checkpoint identity.
    ScoreDiverseV1,
    /// Rank only the latest pending batch, dropping every incumbent. Outcomes
    /// may describe sequential admission events; no old score is carried over.
    LatestBatchScoreV1,
    /// Uniform subset of the latest pending batch only, ordered by ordinal.
    /// Scores (including finite rejection sentinels) do not affect retention.
    LatestBatchUniformV1,
}

impl MeasuredParentSelection {
    fn supports_screened_snapshot(self) -> bool {
        matches!(self, Self::Score | Self::ScoreDiverseV1)
    }

    fn is_latest_batch(self) -> bool {
        matches!(self, Self::LatestBatchScoreV1 | Self::LatestBatchUniformV1)
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_score(&self) -> bool {
        *self == Self::Score
    }
}

fn score_order(parents: &mut [Parent]) {
    parents.sort_by(|a, b| {
        b.outcome
            .total_cmp(&a.outcome)
            .then(a.proposal.ordinal.cmp(&b.proposal.ordinal))
    });
}

fn diverse_parents(
    parents: &mut Vec<Parent>,
    capacity: usize,
    catalog: &Catalog,
) -> Result<(), MeasuredSearchError> {
    score_order(parents);
    if parents.len() <= capacity {
        return Ok(());
    }
    let descriptors = parents
        .iter()
        .map(|p| {
            let expr = parse_expression_with_catalog(&p.proposal.expression, catalog)
                .map_err(|e| MeasuredSearchError::Invalid(e.to_string()))?;
            let mut descriptor =
                crate::novelty::describe_with_catalog(&expr, &p.proposal.provenance, catalog);
            descriptor.lineage_parents.clear();
            Ok(descriptor)
        })
        .collect::<Result<Vec<_>, MeasuredSearchError>>()?;
    let elite_count = capacity.div_ceil(2);
    let mut chosen: Vec<usize> = (0..elite_count).collect();
    for i in elite_count..parents.len() {
        if chosen.len() == capacity {
            break;
        }
        if parents[i].outcome >= 0.
            && chosen.iter().all(|j| {
                crate::novelty::descriptor_distance(&descriptors[i], &descriptors[*j]) >= 0.15
            })
        {
            chosen.push(i);
        }
    }
    for i in elite_count..parents.len() {
        if chosen.len() == capacity {
            break;
        }
        if !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    // Parent-index assignment remains score-ordered, including fallback slots.
    chosen.sort_unstable();
    *parents = chosen.into_iter().map(|i| parents[i].clone()).collect();
    Ok(())
}

/// Hard limits and identity for a deterministic direct-measurement session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredSearchConfig {
    pub seed: u64,
    pub limits: Limits,
    /// Hard ceiling on emitted, semantically unique NEW PROPOSALS (1..=100,000),
    /// NOT on measurements or external evaluator calls. The historical name is
    /// retained for configuration/checkpoint compatibility.
    ///
    /// Every proposal emitted by `ask` spends this budget, including proposals
    /// omitted by `tell_screened`. Imported seeds and incumbent rescoring do not.
    /// For example, 32 proposals in batches of 8, measuring 2 per batch, yields
    /// 8 new measurements and 24 screened-out proposals, then exhausts the budget.
    /// Keep a separate external-call budget in the caller.
    pub max_evaluations: usize,
    /// Total attempted transforms, including invalid and duplicate proposals.
    pub max_attempts: u64,
    /// Maximum proposals per request (at most 256).
    pub batch_size: usize,
    /// Maximum retained measured parents (at most 256).
    pub parent_capacity: usize,
    /// Every Nth transform is fresh exploration; 1 disables measured-parent use.
    pub exploration_every: u64,
    /// Explicit exploration policy. Omission preserves historical fixed-cover behavior.
    #[serde(default)]
    pub exploration_policy: MeasuredExploration,
    /// Opt-in applicability filtering. Legacy serialization stays byte-compatible.
    #[serde(default, skip_serializing_if = "MeasuredMutation::is_legacy")]
    pub mutation_policy: MeasuredMutation,
    /// Missing fields preserve score ranking and legacy checkpoint bytes.
    #[serde(default, skip_serializing_if = "MeasuredParentSelection::is_score")]
    pub parent_selection: MeasuredParentSelection,
    /// Nonempty caller-owned identity of data, objective, splits and evaluator.
    pub evaluator_context: String,
}

impl MeasuredSearchConfig {
    fn validate(&self) -> Result<(), MeasuredSearchError> {
        self.limits
            .validate()
            .map_err(MeasuredSearchError::Invalid)?;
        ensure(
            self.max_evaluations > 0 && self.max_evaluations <= 100_000,
            "new-proposal budget (max_evaluations) must be in 1..=100000",
        )?;
        ensure(
            self.max_attempts >= self.max_evaluations as u64 && self.max_attempts <= 10_000_000,
            "attempt budget must cover new proposals and be <= 10000000",
        )?;
        ensure(
            (1..=256).contains(&self.batch_size) && (1..=256).contains(&self.parent_capacity),
            "batch and parent capacities must be in 1..=256",
        )?;
        ensure(
            self.exploration_every > 0,
            "exploration period must be positive",
        )?;
        ensure(
            !self.evaluator_context.trim().is_empty() && self.evaluator_context.len() <= 4096,
            "evaluator context must be nonempty and bounded",
        )
    }
}

/// One unique formula emitted for measurement, with reproducible lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredProposal {
    pub ordinal: usize,
    pub expression: String,
    pub semantic_fingerprint: String,
    pub provenance: Provenance,
}

/// Finite external measurement; larger outcomes are preferred.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredOutcome {
    pub semantic_fingerprint: String,
    pub outcome: f64,
}

/// A caller-audited measurement in the session's exact evaluator context.
/// These records initialize parents; they do not consume new-proposal budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredSeed {
    pub record_id: String,
    pub expression: String,
    pub outcome: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Bootstrap {
    seeds: Vec<MeasuredSeed>,
    pool_context: String,
    excluded: ExcludedHistory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Parent {
    proposal: MeasuredProposal,
    outcome: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct State {
    schema: u32,
    config: MeasuredSearchConfig,
    catalog_checksum: String,
    attempted: u64,
    emitted: usize,
    measured: usize,
    pool_context: Option<String>,
    seen: BTreeSet<String>,
    parents: Vec<Parent>,
    pending: Vec<MeasuredProposal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bootstrap: Option<Bootstrap>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    screened_out: BTreeSet<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferencedCheckpoint {
    schema: u32,
    excluded_sha256: String,
    state: State,
}

struct ValidatedSeeds {
    seeds: Vec<MeasuredSeed>,
    parents: Vec<Parent>,
    identities: BTreeSet<String>,
}

fn validate_seeds(
    config: &MeasuredSearchConfig,
    catalog: &Catalog,
    pool_context: &str,
    seeds: &[MeasuredSeed],
) -> Result<ValidatedSeeds, MeasuredSearchError> {
    ensure(
        !pool_context.trim().is_empty() && pool_context.len() <= 4096,
        "pool context must be nonempty and bounded",
    )?;
    ensure(
        config.parent_selection.supports_screened_snapshot(),
        "bootstrap requires a snapshot score policy",
    )?;
    ensure(
        !seeds.is_empty() && seeds.len() <= config.parent_capacity,
        "bootstrap seeds must fit parent capacity",
    )?;
    let mut records = BTreeSet::new();
    let mut identities = BTreeSet::new();
    let mut normalized = Vec::new();
    for seed in seeds {
        ensure(
            !seed.record_id.trim().is_empty()
                && seed.record_id.len() <= 4096
                && records.insert(&seed.record_id)
                && seed.outcome.is_finite()
                && seed.expression.len() <= 65_536,
            "invalid or duplicate measured seed record",
        )?;
        let expr = parse_expression_with_catalog(&seed.expression, catalog)
            .map_err(|e| MeasuredSearchError::Invalid(e.to_string()))?;
        ensure(
            expr.kind() == ExprKind::Signal
                && validate_transformed_with_catalog(&expr, &config.limits, catalog).is_ok()
                && analyze_expression(&expr).rejection_reason.is_none(),
            "invalid measured seed expression",
        )?;
        let identity = semantic_fingerprint(&expr);
        ensure(
            identities.insert(identity.clone()),
            "duplicate semantic seed",
        )?;
        normalized.push((
            MeasuredSeed {
                record_id: seed.record_id.clone(),
                expression: canonical(&expr),
                outcome: seed.outcome,
            },
            identity,
        ));
    }
    normalized.sort_by(|a, b| a.0.record_id.cmp(&b.0.record_id));
    let mut parents: Vec<_> = normalized
        .iter()
        .enumerate()
        .map(|(i, (seed, identity))| Parent {
            proposal: MeasuredProposal {
                ordinal: i + 1,
                expression: seed.expression.clone(),
                semantic_fingerprint: identity.clone(),
                provenance: Provenance {
                    kind: crate::TransformKind::Initial,
                    operation: format!("measured_bootstrap:{}", seed.record_id),
                    parent_fingerprints: Vec::new(),
                    requested_operation: None,
                    affected_path: None,
                    old_subtree_fingerprint: None,
                    new_subtree_fingerprint: None,
                    retry_count: 0,
                },
            },
            outcome: seed.outcome,
        })
        .collect();
    score_order(&mut parents);
    Ok(ValidatedSeeds {
        seeds: normalized.into_iter().map(|(s, _)| s).collect(),
        parents,
        identities,
    })
}

impl State {
    fn imported(&self) -> usize {
        self.bootstrap.as_ref().map_or(0, |b| b.seeds.len())
    }

    fn excluded(&self, identity: &str) -> bool {
        self.bootstrap
            .as_ref()
            .is_some_and(|b| b.excluded.entries.contains(identity))
    }
}

fn validate_parent_cohort(state: &State) -> Result<(), MeasuredSearchError> {
    if state.config.parent_selection.is_latest_batch() && state.measured > 0 {
        let start = (state.measured - 1) / state.config.batch_size * state.config.batch_size;
        ensure(
            state.parents.len() == state.config.parent_capacity.min(state.measured - start)
                && state.parents.iter().all(|p| p.proposal.ordinal > start),
            "latest-batch checkpoint contains stale or missing parents",
        )?;
    } else {
        ensure(
            state.parents.len()
                == state
                    .config
                    .parent_capacity
                    .min(state.measured - state.screened_out.len()),
            "snapshot checkpoint contains missing parents",
        )?;
    }
    Ok(())
}

/// Session with measured parent selection and no structural fitness surrogate.
#[derive(Debug, Clone)]
pub struct MeasuredSearchSession {
    catalog: Catalog,
    state: State,
}

/// Invalid configuration, incomplete evidence, or checkpoint I/O failure.
#[derive(Debug, Error)]
pub enum MeasuredSearchError {
    #[error("invalid measured search: {0}")]
    Invalid(String),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Artifact(#[from] crate::ArtifactError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

fn ensure(condition: bool, message: &str) -> Result<(), MeasuredSearchError> {
    if condition {
        Ok(())
    } else {
        Err(MeasuredSearchError::Invalid(message.to_owned()))
    }
}

impl MeasuredSearchSession {
    /// Start an empty session with an explicit catalog and evaluation identity.
    ///
    /// # Errors
    /// Rejects invalid or unbounded configurations and invalid catalogs.
    pub fn new(
        config: MeasuredSearchConfig,
        catalog: Catalog,
    ) -> Result<Self, MeasuredSearchError> {
        config.validate()?;
        catalog.validate().map_err(MeasuredSearchError::Invalid)?;
        let state = State {
            schema: MEASURED_SEARCH_SCHEMA,
            config,
            catalog_checksum: catalog.checksum(),
            attempted: 0,
            emitted: 0,
            measured: 0,
            pool_context: None,
            seen: BTreeSet::new(),
            parents: Vec::new(),
            pending: Vec::new(),
            bootstrap: None,
            screened_out: BTreeSet::new(),
        };
        Ok(Self { catalog, state })
    }

    /// Start from measured parents and an all-history semantic exclusion set.
    ///
    /// The caller must audit labels, catalog/operator compatibility and settings.
    /// Context strings assert identity, not proof of measurement. Use one common
    /// objective/reference pool; do not mix stale marginal scores across pools.
    /// Select at most `parent_capacity` seeds externally. Both positive and
    /// negative historical attempts should be excluded using fingerprints from
    /// this library, not another engine's hashes. Exclusions are not labels.
    ///
    /// Only snapshot score policies are supported: every subsequent `tell`
    /// re-scores incumbents with the completed batch, permitting new winners to
    /// replace seeds. Reuse cached measurements only when their context is valid.
    /// No network requests or external evaluations are performed here.
    ///
    /// # Errors
    /// Rejects context mismatch, invalid/duplicate seeds, incompatible policies,
    /// non-finite outcomes and malformed or excessive historical fingerprints.
    pub fn from_measured_parents(
        config: MeasuredSearchConfig,
        catalog: Catalog,
        evaluator_context: &str,
        pool_context: &str,
        seeds: &[MeasuredSeed],
        excluded: &BTreeSet<String>,
    ) -> Result<Self, MeasuredSearchError> {
        let mut session = Self::new(config, catalog)?;
        ensure(
            evaluator_context == session.state.config.evaluator_context,
            "evaluator context differs",
        )?;
        let initial = validate_seeds(&session.state.config, &session.catalog, pool_context, seeds)?;
        session.state.seen = initial.identities;
        session.state.parents = initial.parents;
        session.state.schema = BOOTSTRAPPED_SEARCH_SCHEMA;
        session.state.emitted = seeds.len();
        session.state.measured = seeds.len();
        session.state.pool_context = Some(pool_context.to_owned());
        session.state.bootstrap = Some(Bootstrap {
            seeds: initial.seeds,
            pool_context: pool_context.to_owned(),
            excluded: ExcludedHistory::new(excluded.clone())?,
        });
        Ok(session)
    }

    /// Return the current batch, or propose a new one from measured parents.
    /// Repeated calls before `tell` return exactly the same pending batch.
    /// An empty batch means the new-proposal or transform budget is exhausted.
    /// Each transform is bounded by the configured expression limits.
    ///
    /// # Errors
    /// Reports an internal parent parse failure without promoting a candidate.
    pub fn ask(&mut self) -> Result<Vec<MeasuredProposal>, MeasuredSearchError> {
        if !self.state.pending.is_empty() {
            return Ok(self.state.pending.clone());
        }
        let config = &self.state.config;
        let target = config
            .batch_size
            .min(config.max_evaluations - (self.state.emitted - self.state.imported()));
        let parents = self
            .state
            .parents
            .iter()
            .map(|parent| {
                parse_expression_with_catalog(&parent.proposal.expression, &self.catalog)
                    .map_err(|error| MeasuredSearchError::Invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        while self.state.pending.len() < target && self.state.attempted < config.max_attempts {
            let index = self.state.attempted;
            self.state.attempted += 1;
            let mut rng =
                ChaCha8Rng::seed_from_u64(config.seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let (expression, provenance) =
                if parents.is_empty() || index.is_multiple_of(config.exploration_every) {
                    match config.exploration_policy {
                        MeasuredExploration::FixedCover => {
                            generate_with_catalog(config.seed, index, &config.limits, &self.catalog)
                        }
                        MeasuredExploration::Grammar => {
                            sample_with_catalog(config.seed, index, &config.limits, &self.catalog)
                        }
                    }
                } else {
                    let left = rng.random_range(0..parents.len());
                    if parents.len() > 1 && rng.random_bool(0.5) {
                        let right = (left + rng.random_range(1..parents.len())) % parents.len();
                        crossover_with_catalog(
                            &parents[left],
                            &parents[right],
                            config.seed,
                            index,
                            &config.limits,
                            &self.catalog,
                        )
                    } else {
                        let mutate = match config.mutation_policy {
                            MeasuredMutation::LegacyUniform => mutate_with_catalog,
                            MeasuredMutation::Applicable => mutate_applicable_with_catalog,
                        };
                        mutate(
                            &parents[left],
                            config.seed,
                            index,
                            &config.limits,
                            &self.catalog,
                        )
                    }
                };
            if expression.kind() != ExprKind::Signal
                || validate_transformed_with_catalog(&expression, &config.limits, &self.catalog)
                    .is_err()
                || analyze_expression(&expression).rejection_reason.is_some()
            {
                continue;
            }
            let identity = semantic_fingerprint(&expression);
            if self.state.excluded(&identity) || !self.state.seen.insert(identity.clone()) {
                continue;
            }
            self.state.emitted += 1;
            self.state.pending.push(MeasuredProposal {
                ordinal: self.state.emitted,
                expression: canonical(&expression),
                semantic_fingerprint: identity,
                provenance,
            });
        }
        Ok(self.state.pending.clone())
    }

    /// Candidates that must receive a score in the next `tell` snapshot.
    /// Includes retained parents for snapshot policies. Latest-batch policies
    /// request only pending proposals and discard all incumbents at tell.
    #[must_use]
    pub fn score_requests(&self) -> Vec<MeasuredProposal> {
        if self.state.config.parent_selection.is_latest_batch() {
            return self.state.pending.clone();
        }
        self.state
            .parents
            .iter()
            .map(|parent| parent.proposal.clone())
            .chain(self.state.pending.iter().cloned())
            .collect()
    }

    /// Atomically accept a complete measurement snapshot for one pool context.
    /// For latest-batch policies the context identifies the batch's measurement
    /// events instead; outcomes need not be simultaneous marginal estimates.
    /// No state changes on duplicate, missing, unexpected or non-finite scores.
    /// Outcome order does not affect selection. Score ties retain earlier proposals.
    /// Uniform selection ignores score values, including finite rejection sentinels.
    ///
    /// # Errors
    /// Rejects wrong evaluator identities, empty contexts and incomplete scores.
    pub fn tell(
        &mut self,
        evaluator_context: &str,
        pool_context: &str,
        outcomes: &[MeasuredOutcome],
    ) -> Result<(), MeasuredSearchError> {
        self.apply_snapshot(evaluator_context, pool_context, outcomes, false)
    }

    /// Resolve a screened pool with actual measurements only for its selected subset.
    ///
    /// Supply every incumbent and at least one pending proposal. Pending proposals
    /// absent from `outcomes` are permanently screened out, not assigned invented
    /// negative labels. They remain deduplicated and consume proposal/attempt
    /// budget, but not `measured()`. Use only after the selected batch is complete;
    /// missing or uncertain external responses are NOT screening decisions.
    /// Selection and surrogate fitting belong to the caller, not this method.
    ///
    /// # Errors
    /// Rejects non-snapshot policies, missing incumbent scores, empty selections,
    /// wrong contexts, unexpected identities, duplicate or non-finite outcomes.
    /// Errors leave the session unchanged.
    pub fn tell_screened(
        &mut self,
        evaluator_context: &str,
        pool_context: &str,
        outcomes: &[MeasuredOutcome],
    ) -> Result<(), MeasuredSearchError> {
        ensure(
            self.state
                .config
                .parent_selection
                .supports_screened_snapshot(),
            "screened feedback requires a snapshot score policy",
        )?;
        self.apply_snapshot(evaluator_context, pool_context, outcomes, true)
    }

    fn apply_snapshot(
        &mut self,
        evaluator_context: &str,
        pool_context: &str,
        outcomes: &[MeasuredOutcome],
        screened: bool,
    ) -> Result<(), MeasuredSearchError> {
        ensure(
            evaluator_context == self.state.config.evaluator_context,
            "evaluator context differs",
        )?;
        ensure(
            !pool_context.trim().is_empty() && pool_context.len() <= 4096,
            "pool context must be nonempty and bounded",
        )?;
        ensure(
            !self.state.pending.is_empty(),
            "no pending measurement batch",
        )?;
        let requests = self.score_requests();
        let mut scores = BTreeMap::new();
        for result in outcomes {
            ensure(result.outcome.is_finite(), "non-finite outcome")?;
            ensure(
                scores
                    .insert(&result.semantic_fingerprint, result.outcome)
                    .is_none(),
                "duplicate outcome",
            )?;
        }
        let omitted: BTreeSet<_> = if screened {
            ensure(
                scores.len() > self.state.parents.len()
                    && scores.len() <= requests.len()
                    && self
                        .state
                        .parents
                        .iter()
                        .all(|p| scores.contains_key(&p.proposal.semantic_fingerprint))
                    && scores
                        .keys()
                        .all(|id| requests.iter().any(|p| &p.semantic_fingerprint == *id)),
                "screened snapshot needs all incumbents and measured pending proposals",
            )?;
            self.state
                .pending
                .iter()
                .filter(|p| !scores.contains_key(&p.semantic_fingerprint))
                .map(|p| p.ordinal)
                .collect()
        } else {
            ensure(scores.len() == requests.len(), "incomplete score snapshot")?;
            BTreeSet::new()
        };
        let mut parents = requests
            .into_iter()
            .filter(|p| !omitted.contains(&p.ordinal))
            .map(|proposal| {
                let outcome = scores.get(&proposal.semantic_fingerprint).ok_or_else(|| {
                    MeasuredSearchError::Invalid("missing or unexpected score identity".to_owned())
                })?;
                Ok(Parent {
                    proposal,
                    outcome: *outcome,
                })
            })
            .collect::<Result<Vec<_>, MeasuredSearchError>>()?;
        match self.state.config.parent_selection {
            MeasuredParentSelection::Score | MeasuredParentSelection::LatestBatchScoreV1 => {
                score_order(&mut parents);
            }
            MeasuredParentSelection::ScoreDiverseV1 => diverse_parents(
                &mut parents,
                self.state.config.parent_capacity,
                &self.catalog,
            )?,
            MeasuredParentSelection::Uniform | MeasuredParentSelection::LatestBatchUniformV1 => {
                // Separate indexed stream: neither proposal RNG nor scores/contexts
                // feed retention. Canonical input order avoids inherited score order.
                parents.sort_by_key(|p| p.proposal.ordinal);
                let domain = if self.state.config.parent_selection
                    == MeasuredParentSelection::LatestBatchUniformV1
                {
                    0x6c61_7465_7374_7631
                } else {
                    0x7061_7265_6e74_7631
                };
                let seed = self.state.config.seed
                    ^ domain
                    ^ (self.state.emitted as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                parents.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
                parents.truncate(self.state.config.parent_capacity);
                parents.sort_by_key(|p| p.proposal.ordinal);
            }
        }
        parents.truncate(self.state.config.parent_capacity);
        self.state.parents = parents;
        if !omitted.is_empty() {
            self.state.schema = BOOTSTRAPPED_SEARCH_SCHEMA;
            self.state.screened_out.extend(omitted);
        }
        self.state.measured += self.state.pending.len();
        self.state.pending.clear();
        self.state.pool_context = Some(pool_context.to_owned());
        Ok(())
    }

    /// Retained parents: score-descending except ordinal-ascending for uniform policies.
    #[must_use]
    pub fn parents(&self) -> Vec<MeasuredProposal> {
        self.state
            .parents
            .iter()
            .map(|parent| parent.proposal.clone())
            .collect()
    }

    /// Number of attempted transforms, including rejected duplicates.
    #[must_use]
    pub fn attempted(&self) -> u64 {
        self.state.attempted
    }

    /// Number of emitted proposals with accepted measurements.
    #[must_use]
    pub fn measured(&self) -> usize {
        self.state.measured - self.state.imported() - self.state.screened_out.len()
    }

    /// Proposals screened out without measurement, never treated as negative labels.
    #[must_use]
    pub fn screened_out(&self) -> usize {
        self.state.screened_out.len()
    }

    /// Number of imported measurements, separate from newly measured proposals.
    #[must_use]
    pub fn imported(&self) -> usize {
        self.state.imported()
    }

    /// Immutable archive to persist once before using referenced checkpoints.
    #[must_use]
    pub fn excluded_history(&self) -> Option<&ExcludedHistory> {
        self.state.bootstrap.as_ref().map(|b| &b.excluded)
    }

    /// Serialize a schema-3 envelope without repeating the immutable exclusion set.
    /// Seeds, live seen identities, parents and pending proposals remain inline.
    /// Persist `excluded_history().to_json()` separately FIRST, using its checksum
    /// as the artifact key. This format is opt-in; `checkpoint_json` stays standalone.
    ///
    /// # Errors
    /// Rejects a session without bootstrap history or serialization failure.
    pub fn checkpoint_with_history_json(&self) -> Result<String, MeasuredSearchError> {
        let mut state = self.state.clone(); // Exclusion storage is Arc-shared, not copied.
        let bootstrap = state.bootstrap.as_mut().ok_or_else(|| {
            MeasuredSearchError::Invalid("referenced checkpoint requires bootstrap".into())
        })?;
        let excluded_sha256 = bootstrap.excluded.checksum().to_owned();
        bootstrap.excluded = ExcludedHistory::new(BTreeSet::new())?;
        Ok(serde_json::to_string(&ReferencedCheckpoint {
            schema: REFERENCED_SEARCH_SCHEMA,
            excluded_sha256,
            state,
        })?)
    }

    /// Restore a referenced checkpoint against a preloaded, checksum-bound archive.
    /// Loading/validating history once and reusing this value avoids reparsing or
    /// cloning 100k exclusions on every restore. The original metadata/proposal
    /// validators still run; the checkpoint cannot supply replacement history.
    ///
    /// # Errors
    /// Rejects missing/mismatched history, embedded history, bad schemas or state.
    pub fn from_checkpoint_with_history_json(
        content: &str,
        config: MeasuredSearchConfig,
        catalog: Catalog,
        history: &ExcludedHistory,
    ) -> Result<Self, MeasuredSearchError> {
        ensure(
            content.len() <= 64 * 1024 * 1024,
            "checkpoint exceeds 64 MiB",
        )?;
        let mut referenced: ReferencedCheckpoint = serde_json::from_str(content)?;
        ensure(
            referenced.schema == REFERENCED_SEARCH_SCHEMA
                && referenced.excluded_sha256 == history.checksum(),
            "referenced history mismatch",
        )?;
        let bootstrap = referenced.state.bootstrap.as_mut().ok_or_else(|| {
            MeasuredSearchError::Invalid("referenced checkpoint requires bootstrap".into())
        })?;
        ensure(
            bootstrap.excluded.entries.is_empty(),
            "referenced checkpoint embeds history",
        )?;
        bootstrap.excluded = history.clone();
        Self::restore_state(referenced.state, config, catalog)
    }

    /// Serialize all bounded state, including pending proposals and RNG indices.
    /// Finite outcome bits (including signed zero) round-trip through the reader.
    ///
    /// # Errors
    /// Returns a JSON serialization error.
    pub fn checkpoint_json(&self) -> Result<String, MeasuredSearchError> {
        Ok(serde_json::to_string(&self.state)?)
    }

    /// Build an embeddable JSON value directly from state, without reparsing
    /// decimal floats through a string intermediary. Preserves outcome bits
    /// in memory. The string checkpoint reader also preserves finite outcome bits.
    ///
    /// # Errors
    /// Returns a JSON serialization error.
    pub fn checkpoint_value(&self) -> Result<serde_json::Value, MeasuredSearchError> {
        Ok(serde_json::to_value(&self.state)?)
    }

    /// Restore a checkpoint only under the exact expected config and catalog.
    ///
    /// # Errors
    /// Rejects malformed identities, bounds, counters and invalid expressions.
    pub fn from_checkpoint_json(
        content: &str,
        config: MeasuredSearchConfig,
        catalog: Catalog,
    ) -> Result<Self, MeasuredSearchError> {
        ensure(
            content.len() <= 64 * 1024 * 1024,
            "checkpoint exceeds 64 MiB",
        )?;
        let state: State = serde_json::from_str(content)?;
        Self::restore_state(state, config, catalog)
    }

    fn restore_state(
        state: State,
        config: MeasuredSearchConfig,
        catalog: Catalog,
    ) -> Result<Self, MeasuredSearchError> {
        let mut session = Self::new(config, catalog)?;
        let initial = session.validate_checkpoint_metadata(&state)?;
        session.validate_checkpoint_proposals(&state, initial.as_ref())?;
        session.state = state;
        Ok(session)
    }

    fn validate_checkpoint_metadata(
        &self,
        state: &State,
    ) -> Result<Option<ValidatedSeeds>, MeasuredSearchError> {
        ensure(
            state.schema
                == if state.bootstrap.is_some() || !state.screened_out.is_empty() {
                    BOOTSTRAPPED_SEARCH_SCHEMA
                } else {
                    MEASURED_SEARCH_SCHEMA
                }
                && state.config == self.state.config
                && state.catalog_checksum == self.state.catalog_checksum,
            "checkpoint configuration differs",
        )?;
        let initial = if let Some(bootstrap) = &state.bootstrap {
            let initial = validate_seeds(
                &self.state.config,
                &self.catalog,
                &bootstrap.pool_context,
                &bootstrap.seeds,
            )?;
            ensure(initial.seeds == bootstrap.seeds, "noncanonical bootstrap")?;
            Some(initial)
        } else {
            None
        };
        let imported = state.imported();
        ensure(
            state.emitted >= imported && state.measured >= imported,
            "invalid imported counters",
        )?;
        ensure(
            state
                .screened_out
                .iter()
                .all(|ordinal| *ordinal > imported && *ordinal <= state.measured)
                && (state.screened_out.is_empty()
                    || state.config.parent_selection.supports_screened_snapshot()),
            "invalid screened-out ordinals or policy",
        )?;
        ensure(
            state.emitted - imported <= state.config.max_evaluations
                && state.attempted <= state.config.max_attempts
                && state.attempted >= (state.emitted - imported) as u64
                && state.measured <= state.emitted
                && state.emitted - state.measured == state.pending.len()
                && state.pending.len() <= state.config.batch_size
                && state.parents.len() <= state.config.parent_capacity
                && state.parents.len() <= state.measured
                && state.seen.len() == state.emitted,
            "invalid checkpoint counters",
        )?;
        if let Some(initial) = &initial {
            ensure(
                initial.identities.is_subset(&state.seen),
                "missing imported identity",
            )?;
            ensure(
                state
                    .seen
                    .iter()
                    .all(|id| !state.excluded(id) || initial.identities.contains(id)),
                "historical identity emitted again",
            )?;
            if state.measured == imported {
                ensure(
                    state.parents == initial.parents
                        && state.pool_context.as_deref()
                            == state.bootstrap.as_ref().map(|b| b.pool_context.as_str()),
                    "altered bootstrap parents",
                )?;
            }
        }
        ensure(
            (state.measured == 0 && state.pool_context.is_none())
                || (state.measured > 0
                    && state
                        .pool_context
                        .as_ref()
                        .is_some_and(|s| !s.trim().is_empty() && s.len() <= 4096)),
            "invalid checkpoint pool context",
        )?;
        Ok(initial)
    }

    fn validate_checkpoint_proposals(
        &self,
        state: &State,
        initial: Option<&ValidatedSeeds>,
    ) -> Result<(), MeasuredSearchError> {
        let imported = state.imported();
        let mut identities = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        for proposal in state
            .parents
            .iter()
            .map(|p| &p.proposal)
            .chain(&state.pending)
        {
            if let Some(initial) = &initial
                && proposal.ordinal <= imported
            {
                ensure(
                    initial.parents.iter().any(|p| p.proposal == *proposal),
                    "altered imported proposal",
                )?;
            }
            let expression = parse_expression_with_catalog(&proposal.expression, &self.catalog)
                .map_err(|error| MeasuredSearchError::Invalid(error.to_string()))?;
            ensure(
                expression.kind() == ExprKind::Signal
                    && validate_transformed_with_catalog(
                        &expression,
                        &state.config.limits,
                        &self.catalog,
                    )
                    .is_ok()
                    && analyze_expression(&expression).rejection_reason.is_none()
                    && canonical(&expression) == proposal.expression
                    && semantic_fingerprint(&expression) == proposal.semantic_fingerprint
                    && state.seen.contains(&proposal.semantic_fingerprint)
                    && identities.insert(proposal.semantic_fingerprint.clone())
                    && proposal.ordinal > 0
                    && proposal.ordinal <= state.emitted
                    && !state.screened_out.contains(&proposal.ordinal)
                    && ordinals.insert(proposal.ordinal),
                "invalid checkpoint proposal",
            )?;
        }
        ensure(
            state
                .parents
                .iter()
                .all(|p| p.outcome.is_finite() && p.proposal.ordinal <= state.measured),
            "invalid checkpoint parent",
        )?;
        validate_parent_cohort(state)?;
        ensure(
            state
                .pending
                .iter()
                .enumerate()
                .all(|(i, p)| p.ordinal == state.measured + i + 1),
            "invalid pending order",
        )?;
        ensure(
            state
                .parents
                .windows(2)
                .all(|p| match state.config.parent_selection {
                    MeasuredParentSelection::Score
                    | MeasuredParentSelection::ScoreDiverseV1
                    | MeasuredParentSelection::LatestBatchScoreV1 => {
                        p[0].outcome.total_cmp(&p[1].outcome).is_gt()
                            || (p[0].outcome.total_cmp(&p[1].outcome).is_eq()
                                && p[0].proposal.ordinal < p[1].proposal.ordinal)
                    }
                    MeasuredParentSelection::Uniform
                    | MeasuredParentSelection::LatestBatchUniformV1 => {
                        p[0].proposal.ordinal < p[1].proposal.ordinal
                    }
                }),
            "invalid parent ordering",
        )?;
        Ok(())
    }

    /// Durably replace a checkpoint using the library's atomic writer.
    ///
    /// # Errors
    /// Returns a serialization or durable replacement error.
    pub fn write_checkpoint(&self, path: &Path) -> Result<(), MeasuredSearchError> {
        crate::artifact::atomic_write(path, self.checkpoint_json()?.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;

    fn parents() -> Vec<Parent> {
        let catalog = crate::builtin_catalog();
        [
            "rank(ts_mean(close, 20))",
            "rank(ts_mean(open, 20))",
            "rank(ts_mean(close, 21))",
            "rank(ts_mean(open, 21))",
            "volume",
            "rank(ts_std_dev(volume, 5))",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, source)| {
            let expr = parse_expression_with_catalog(source, catalog).unwrap();
            Parent {
                proposal: MeasuredProposal {
                    ordinal: i + 1,
                    expression: canonical(&expr),
                    semantic_fingerprint: semantic_fingerprint(&expr),
                    provenance: Provenance {
                        kind: crate::TransformKind::Initial,
                        operation: "fixture".into(),
                        parent_fingerprints: vec![format!("different-lineage-{i}")],
                        requested_operation: None,
                        affected_path: None,
                        old_subtree_fingerprint: None,
                        new_subtree_fingerprint: None,
                        retry_count: 0,
                    },
                },
                outcome: f64::from(6 - u32::try_from(i).unwrap()),
            }
        })
        .collect()
    }

    #[test]
    fn diverse_retention_golden_preserves_elites_and_ignores_lineage() {
        let mut selected = parents();
        let mut other = selected.clone();
        for p in &mut other {
            p.proposal.provenance.parent_fingerprints.clear();
        }
        diverse_parents(&mut selected, 4, crate::builtin_catalog()).unwrap();
        diverse_parents(&mut other, 4, crate::builtin_catalog()).unwrap();
        let ordinals = |p: &[Parent]| p.iter().map(|x| x.proposal.ordinal).collect::<Vec<_>>();
        assert_eq!(ordinals(&selected), vec![1, 2, 5, 6]);
        assert_eq!(ordinals(&selected), ordinals(&other));
    }

    #[test]
    fn diversity_fallback_keeps_capacity_and_does_not_promote_negative_scores() {
        let mut negative = parents();
        negative[4].outcome = -1.;
        negative[5].outcome = -2.;
        diverse_parents(&mut negative, 4, crate::builtin_catalog()).unwrap();
        assert_eq!(
            negative
                .iter()
                .map(|p| p.proposal.ordinal)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let mut similar = parents();
        similar.truncate(4);
        let expected = similar[..3].to_vec();
        diverse_parents(&mut similar, 3, crate::builtin_catalog()).unwrap();
        assert_eq!(similar, expected);
        let mut below_capacity = parents();
        let expected = below_capacity.clone();
        diverse_parents(&mut below_capacity, 8, crate::builtin_catalog()).unwrap();
        assert_eq!(below_capacity, expected);
    }
}
