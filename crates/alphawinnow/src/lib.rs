//! Standalone expression parsing, semantic deduplication, and local search.

pub mod analysis;
pub mod artifact;
pub mod ast;
pub mod canonical;
pub mod dedup;
pub mod dialect;
pub mod engine;
pub mod feedback;
pub mod novelty;
#[cfg(feature = "numeric-evidence")]
pub mod numeric;
pub mod operators;
#[cfg(feature = "parquet-input")]
pub mod parquet_input;
pub mod parser;
pub mod transform;
pub mod tree;

pub use analysis::{ExpressionAnalysis, RejectionReason, analyze_expression};
pub use artifact::{
    ArtifactError, CANDIDATE_SCHEMA, CHECKPOINT_SCHEMA, CandidateRecord, CheckpointDraft,
    MANIFEST_SCHEMA, PublicationPointer, RunCheckpoint, RunManifest, StructuralScoreComponents,
    publication_pointer_path, read_candidates, read_checkpoint, read_published_run,
    write_checkpoint_atomic, write_jsonl_atomic, write_manifest_atomic, write_run_transactional,
};
pub use ast::{Expr, ExprKind};
pub use canonical::{canonical, fingerprint, semantic_canonical, semantic_fingerprint};
pub use dedup::{
    DedupError, DedupResult, DuplicateRecord, RejectedRecord, deduplicate, deduplicate_with_catalog,
};
pub use dialect::{
    DIALECT_SCHEMA, DialectArtifact, DialectError, DialectSpec, UnaryScalarLowering,
    compile_dialect,
};
pub use engine::{
    SearchError, SearchResult, SearchRunOptions, run_search, run_search_with_catalog,
    run_search_with_options, run_search_with_options_and_catalog,
};
pub use feedback::{
    FEEDBACK_DATASET_SCHEMA, FeedbackAuditReport, FeedbackConfig, FeedbackDataset, FeedbackError,
    FeedbackRecord, GUIDED_CANDIDATE_SCHEMA, GuidanceEstimate, GuidedCandidate, audit_feedback,
    candidate_features, prioritize_candidates, write_guided_jsonl_atomic,
};
pub use novelty::{
    StructuralDescriptor, common_ancestor_depth, describe, describe_with_catalog,
    descriptor_distance,
};
#[cfg(feature = "numeric-evidence")]
pub use numeric::{
    BrainProxyMetrics, ColumnarDataset, DailyEvidence, EvaluationArtifact, EvaluationConfig,
    NumericColumn, NumericError, NumericEvaluator, SplitEvidence, evaluate_numeric,
    evaluate_numeric_with_catalog,
};
pub use operators::{
    ArgumentSpec, Arity, Catalog, FieldSpec, GroupSpec, KeywordConstraint, KeywordSpec,
    OperatorSpec, ScalarDomain, ValueDomain, builtin_catalog,
};
#[cfg(feature = "parquet-input")]
pub use parquet_input::{
    ParquetInputError, PreparedPanel, SharadarPanelConfig, prepare_sharadar_panel,
};
pub use parser::{ExpressionError, parse_expression, parse_expression_with_catalog};
pub use transform::{
    Limits, MutationClass, Provenance, TransformKind, TransformValidationError, crossover,
    crossover_with_catalog, crossover_with_paths, generate_with_catalog, mutate,
    mutate_with_catalog, mutate_with_class, validate_transformed,
    validate_transformed_with_catalog,
};
pub use tree::{
    ExprPath, KindSet, PathEntry, PathError, PathSegment, allowed_kinds, enumerate_paths,
    replace_subtree, subtree_at,
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Schema version for the public search configuration.
pub const SEARCH_SPEC_SCHEMA: u32 = 2;
/// Absolute memory-safety ceiling for a single run.
pub const MAX_CANDIDATES: u64 = 10_000_000;
/// Absolute worker-pool ceiling for a single process.
pub const MAX_THREADS: usize = 256;
/// Absolute retained-result ceiling for a single run.
pub const MAX_SHORTLIST: usize = 100_000;

/// Bounded configuration shared by the CLI and the future search engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpec {
    /// Artifact schema version.
    pub schema: u32,
    /// Seed controlling deterministic generation.
    pub seed: u64,
    /// Maximum number of worker threads.
    pub threads: usize,
    /// Hard candidate-generation ceiling.
    pub max_candidates: u64,
    /// Hard wall-clock ceiling in seconds.
    pub duration_seconds: u64,
    /// Hard expression limits.
    #[serde(default)]
    pub limits: Limits,
    /// Structural scoring weights.
    #[serde(default)]
    pub scoring: StructuralScoreConfig,
    /// Maximum number of records written to the shortlist.
    #[serde(default = "default_shortlist")]
    pub shortlist_size: usize,
    /// Operational resident-memory and finalization bounds.
    #[serde(default)]
    pub resources: ResourceLimits,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub population_capacity: usize,
    pub parent_pool_capacity: usize,
    pub elite_archive_capacity: usize,
    pub semantic_archive_capacity: usize,
    pub descriptor_archive_capacity: usize,
    pub finalization_reserve_milliseconds: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            population_capacity: 512,
            parent_pool_capacity: 128,
            elite_archive_capacity: 4_096,
            semantic_archive_capacity: 100_000,
            descriptor_archive_capacity: 4_096,
            finalization_reserve_milliseconds: 250,
        }
    }
}

impl ResourceLimits {
    fn validate(&self, shortlist_size: usize) -> Result<(), String> {
        if self.population_capacity == 0
            || self.parent_pool_capacity == 0
            || self.elite_archive_capacity < shortlist_size
            || self.semantic_archive_capacity == 0
            || self.descriptor_archive_capacity == 0
        {
            return Err(
                "capacities must be positive and elite archive must fit the shortlist".to_owned(),
            );
        }
        if self.parent_pool_capacity > self.population_capacity
            || self.population_capacity > 100_000
            || self.elite_archive_capacity > 1_000_000
            || self.semantic_archive_capacity > 10_000_000
            || self.descriptor_archive_capacity > 1_000_000
            || self.finalization_reserve_milliseconds > 60_000
        {
            return Err("resource capacities exceed supported bounds".to_owned());
        }
        Ok(())
    }
}

const fn default_shortlist() -> usize {
    100
}

impl SearchSpec {
    /// Validate hard resource bounds before search initialization.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError`] when a field is unsupported or zero.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema != SEARCH_SPEC_SCHEMA {
            return Err(SpecError::UnsupportedSchema(self.schema));
        }
        if self.threads == 0 {
            return Err(SpecError::ZeroThreads);
        }
        if self.threads > MAX_THREADS {
            return Err(SpecError::TooManyThreads(self.threads));
        }
        if self.max_candidates == 0 {
            return Err(SpecError::ZeroCandidates);
        }
        if self.max_candidates > MAX_CANDIDATES {
            return Err(SpecError::TooManyCandidates(self.max_candidates));
        }
        if self.duration_seconds == 0 {
            return Err(SpecError::ZeroDuration);
        }
        if self.shortlist_size == 0 {
            return Err(SpecError::ZeroShortlist);
        }
        if self.shortlist_size > MAX_SHORTLIST {
            return Err(SpecError::TooLargeShortlist(self.shortlist_size));
        }
        self.limits.validate().map_err(SpecError::InvalidLimits)?;
        self.scoring.validate().map_err(SpecError::InvalidScoring)?;
        self.resources
            .validate(self.shortlist_size)
            .map_err(SpecError::InvalidResources)
    }
}

/// Validation failures for [`SearchSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SpecError {
    /// The artifact schema cannot be read by this binary.
    #[error("unsupported search specification schema {0}")]
    UnsupportedSchema(u32),
    /// A worker pool must contain at least one thread.
    #[error("threads must be greater than zero")]
    ZeroThreads,
    /// Worker pools have an absolute safety ceiling.
    #[error("threads must not exceed {MAX_THREADS}, got {0}")]
    TooManyThreads(usize),
    /// A run must permit at least one candidate.
    #[error("max_candidates must be greater than zero")]
    ZeroCandidates,
    /// Candidate storage has an absolute safety ceiling.
    #[error("max_candidates must not exceed {MAX_CANDIDATES}, got {0}")]
    TooManyCandidates(u64),
    /// A run must have a positive duration.
    #[error("duration_seconds must be greater than zero")]
    ZeroDuration,
    /// At least one result must be retained.
    #[error("shortlist_size must be greater than zero")]
    ZeroShortlist,
    /// Retained results have an absolute safety ceiling.
    #[error("shortlist_size must not exceed {MAX_SHORTLIST}, got {0}")]
    TooLargeShortlist(usize),
    /// Invalid grammar limits.
    #[error("invalid expression limits: {0}")]
    InvalidLimits(String),
    /// Invalid scoring schema or weight.
    #[error("invalid structural scoring configuration: {0}")]
    InvalidScoring(String),
    /// Invalid resident-memory or finalization bounds.
    #[error("invalid resource limits: {0}")]
    InvalidResources(String),
}

/// Versioned weights for a deliberately non-financial local score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralScoreConfig {
    /// Scoring schema version.
    pub schema: u32,
    /// Preference for smaller expressions.
    pub simplicity_weight: f64,
    /// Preference for expressions absent from the optional archive.
    pub archive_novelty_weight: f64,
    /// Preference for underrepresented root operator families.
    pub family_diversity_weight: f64,
    /// Preference for distinct parentage.
    pub lineage_diversity_weight: f64,
    /// Preference for distance in layered AST descriptors.
    #[serde(default = "default_structural_novelty_weight")]
    pub structural_novelty_weight: f64,
    /// Preference for underrepresented transformation origins.
    #[serde(default = "default_transform_diversity_weight")]
    pub transform_diversity_weight: f64,
}

const fn default_structural_novelty_weight() -> f64 {
    0.35
}

const fn default_transform_diversity_weight() -> f64 {
    0.10
}

impl Default for StructuralScoreConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            simplicity_weight: 0.20,
            archive_novelty_weight: 0.15,
            family_diversity_weight: 0.10,
            lineage_diversity_weight: 0.10,
            structural_novelty_weight: default_structural_novelty_weight(),
            transform_diversity_weight: default_transform_diversity_weight(),
        }
    }
}

impl StructuralScoreConfig {
    fn validate(&self) -> Result<(), String> {
        if self.schema != 1 {
            return Err(format!("unsupported schema {}", self.schema));
        }
        let weights = [
            self.simplicity_weight,
            self.archive_novelty_weight,
            self.family_diversity_weight,
            self.lineage_diversity_weight,
            self.structural_novelty_weight,
            self.transform_diversity_weight,
        ];
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err("weights must be finite and non-negative".to_owned());
        }
        if weights.iter().sum::<f64>() == 0.0 {
            return Err("at least one weight must be positive".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn normalized(&self) -> StructuralScoreWeights {
        let total = self.simplicity_weight
            + self.archive_novelty_weight
            + self.family_diversity_weight
            + self.lineage_diversity_weight
            + self.structural_novelty_weight
            + self.transform_diversity_weight;
        StructuralScoreWeights {
            simplicity: self.simplicity_weight / total,
            archive_novelty: self.archive_novelty_weight / total,
            family_diversity: self.family_diversity_weight / total,
            lineage_diversity: self.lineage_diversity_weight / total,
            structural_novelty: self.structural_novelty_weight / total,
            transform_diversity: self.transform_diversity_weight / total,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralScoreWeights {
    pub simplicity: f64,
    pub archive_novelty: f64,
    pub family_diversity: f64,
    pub lineage_diversity: f64,
    pub structural_novelty: f64,
    pub transform_diversity: f64,
}

impl Default for StructuralScoreWeights {
    fn default() -> Self {
        StructuralScoreConfig::default().normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::{ResourceLimits, SEARCH_SPEC_SCHEMA, SearchSpec, SpecError, StructuralScoreConfig};
    use crate::Limits;

    #[test]
    fn valid_spec_round_trips_as_json() {
        let spec = SearchSpec {
            schema: SEARCH_SPEC_SCHEMA,
            seed: 42,
            threads: 4,
            max_candidates: 100_000,
            duration_seconds: 3_600,
            limits: Limits::default(),
            scoring: StructuralScoreConfig::default(),
            shortlist_size: 100,
            resources: ResourceLimits::default(),
        };

        spec.validate().expect("valid search spec");
        let encoded = serde_json::to_string(&spec).expect("serialize search spec");
        let decoded: SearchSpec = serde_json::from_str(&encoded).expect("parse search spec");
        assert_eq!(decoded, spec);
    }

    #[test]
    fn zero_threads_are_rejected() {
        let spec = SearchSpec {
            schema: SEARCH_SPEC_SCHEMA,
            seed: 42,
            threads: 0,
            max_candidates: 1,
            duration_seconds: 1,
            limits: Limits::default(),
            scoring: StructuralScoreConfig::default(),
            shortlist_size: 100,
            resources: ResourceLimits::default(),
        };

        assert_eq!(spec.validate(), Err(SpecError::ZeroThreads));
    }
}
