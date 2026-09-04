//! Offline measured-feedback guidance kept separate from structural scoring.

use std::{collections::BTreeMap, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CandidateRecord, artifact, canonical::digest};

pub const FEEDBACK_DATASET_SCHEMA: u32 = 1;
pub const GUIDED_CANDIDATE_SCHEMA: u32 = 1;

const FAMILY_BUCKETS: usize = 8;

const SUPPORTED_FEATURES: [&str; 26] = [
    "conditional",
    "depth",
    "field_count",
    "field_family_bucket_0_count",
    "field_family_bucket_1_count",
    "field_family_bucket_2_count",
    "field_family_bucket_3_count",
    "field_family_bucket_4_count",
    "field_family_bucket_5_count",
    "field_family_bucket_6_count",
    "field_family_bucket_7_count",
    "field_family_count",
    "group_count",
    "lineage_parent_count",
    "nodes",
    "nonlinear",
    "operator_arithmetic_count",
    "operator_conditional_count",
    "operator_count",
    "operator_cross_sectional_count",
    "operator_preprocess_count",
    "operator_time_series_count",
    "window_long_count",
    "window_medium_count",
    "window_short_count",
    "window_total_count",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackDataset {
    pub schema: u32,
    pub dataset_id: String,
    pub context_checksum: String,
    pub outcome_label: String,
    pub feature_scales: BTreeMap<String, f64>,
    pub records: Vec<FeedbackRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackRecord {
    pub record_id: String,
    pub features: BTreeMap<String, f64>,
    pub outcome: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackConfig {
    pub schema: u32,
    pub neighbors: usize,
    pub minimum_neighbors: usize,
    pub maximum_distance: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            neighbors: 8,
            minimum_neighbors: 3,
            maximum_distance: 0.75,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidanceEstimate {
    pub schema: u32,
    pub dataset_id: String,
    pub context_checksum: String,
    pub feedback_checksum: String,
    pub outcome_label: String,
    pub neighbor_count: usize,
    pub nearest_distance: f64,
    pub weighted_outcome: f64,
    pub weighted_dispersion: f64,
    pub conservative_outcome: f64,
    pub structural_score_used: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuidedCandidate {
    pub schema: u32,
    pub candidate: CandidateRecord,
    pub measured_guidance: Option<GuidanceEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeedbackAuditReport {
    pub schema: u32,
    pub dataset_id: String,
    pub feedback_checksum: String,
    pub records: usize,
    pub evaluated_records: usize,
    pub unguided_records: usize,
    pub coverage: f64,
    pub mean_absolute_error: Option<f64>,
    pub root_mean_square_error: Option<f64>,
    pub sign_accuracy: Option<f64>,
    pub balanced_sign_accuracy: Option<f64>,
    pub roc_auc: Option<f64>,
}

#[derive(Debug, Error)]
pub enum FeedbackError {
    #[error("unsupported feedback dataset schema {0}")]
    DatasetSchema(u32),
    #[error("unsupported feedback configuration schema {0}")]
    ConfigSchema(u32),
    #[error("feedback dataset metadata must not be empty")]
    EmptyMetadata,
    #[error("feedback dataset must contain at least one record and one feature")]
    EmptyDataset,
    #[error("unsupported feedback feature {0}")]
    UnsupportedFeature(String),
    #[error("feature scale for {0} must be finite and positive")]
    InvalidScale(String),
    #[error("feedback record IDs must be non-empty and unique")]
    InvalidRecordId,
    #[error("record {record_id} does not match the dataset feature set")]
    FeatureSet { record_id: String },
    #[error("record {record_id} contains a non-finite feature")]
    NonFiniteFeature { record_id: String },
    #[error("record {record_id} outcome must be finite and within [-1, 1]")]
    InvalidOutcome { record_id: String },
    #[error("record {record_id} confidence must be finite and within (0, 1]")]
    InvalidConfidence { record_id: String },
    #[error("neighbors must be positive and minimum_neighbors must not exceed neighbors")]
    InvalidNeighborCount,
    #[error("maximum_distance must be finite and within [0, 1]")]
    InvalidMaximumDistance,
    #[error(transparent)]
    Artifact(#[from] artifact::ArtifactError),
}

impl FeedbackDataset {
    /// Validate a complete immutable feedback dataset.
    ///
    /// # Errors
    /// Returns a machine-readable error for unsupported schemas, malformed
    /// metadata, feature mismatches, or invalid numeric values.
    pub fn validate(&self) -> Result<(), FeedbackError> {
        if self.schema != FEEDBACK_DATASET_SCHEMA {
            return Err(FeedbackError::DatasetSchema(self.schema));
        }
        if self.dataset_id.trim().is_empty()
            || self.context_checksum.trim().is_empty()
            || self.outcome_label.trim().is_empty()
        {
            return Err(FeedbackError::EmptyMetadata);
        }
        if self.feature_scales.is_empty() || self.records.is_empty() {
            return Err(FeedbackError::EmptyDataset);
        }
        for (name, scale) in &self.feature_scales {
            if SUPPORTED_FEATURES.binary_search(&name.as_str()).is_err() {
                return Err(FeedbackError::UnsupportedFeature(name.clone()));
            }
            if !scale.is_finite() || *scale <= 0.0 {
                return Err(FeedbackError::InvalidScale(name.clone()));
            }
        }
        let mut identifiers = std::collections::BTreeSet::new();
        for record in &self.records {
            if record.record_id.trim().is_empty() || !identifiers.insert(&record.record_id) {
                return Err(FeedbackError::InvalidRecordId);
            }
            if record.features.keys().ne(self.feature_scales.keys()) {
                return Err(FeedbackError::FeatureSet {
                    record_id: record.record_id.clone(),
                });
            }
            if record.features.values().any(|value| !value.is_finite()) {
                return Err(FeedbackError::NonFiniteFeature {
                    record_id: record.record_id.clone(),
                });
            }
            if !record.outcome.is_finite() || !(-1.0..=1.0).contains(&record.outcome) {
                return Err(FeedbackError::InvalidOutcome {
                    record_id: record.record_id.clone(),
                });
            }
            if !record.confidence.is_finite() || record.confidence <= 0.0 || record.confidence > 1.0
            {
                return Err(FeedbackError::InvalidConfidence {
                    record_id: record.record_id.clone(),
                });
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn checksum(&self) -> String {
        let mut canonical = self.clone();
        canonical
            .records
            .sort_by(|left, right| left.record_id.cmp(&right.record_id));
        digest(&serde_json::to_string(&canonical).unwrap_or_default())
    }
}

impl FeedbackConfig {
    /// Validate bounded nearest-neighbor guidance settings.
    ///
    /// # Errors
    /// Returns an error for an unsupported schema or invalid bounds.
    pub fn validate(&self) -> Result<(), FeedbackError> {
        if self.schema != 1 {
            return Err(FeedbackError::ConfigSchema(self.schema));
        }
        if self.neighbors == 0
            || self.minimum_neighbors == 0
            || self.minimum_neighbors > self.neighbors
            || self.neighbors > 10_000
        {
            return Err(FeedbackError::InvalidNeighborCount);
        }
        if !self.maximum_distance.is_finite() || !(0.0..=1.0).contains(&self.maximum_distance) {
            return Err(FeedbackError::InvalidMaximumDistance);
        }
        Ok(())
    }
}

/// Extract public, field-name-independent structural features from a candidate.
#[must_use]
pub fn candidate_features(candidate: &CandidateRecord) -> BTreeMap<String, f64> {
    let descriptor = &candidate.structural_descriptor;
    let operator_count = descriptor.operator_histogram.values().copied().sum::<u32>();
    let operator_family_count = |operators: &[&str]| {
        f64::from(
            operators
                .iter()
                .map(|name| {
                    descriptor
                        .operator_histogram
                        .get(*name)
                        .copied()
                        .unwrap_or(0)
                })
                .sum::<u32>(),
        )
    };
    let window_count =
        |bucket: &str| f64::from(descriptor.window_buckets.get(bucket).copied().unwrap_or(0));
    let mut features = BTreeMap::from([
        ("conditional".to_owned(), f64::from(descriptor.conditional)),
        ("depth".to_owned(), usize_as_f64(candidate.depth)),
        (
            "field_count".to_owned(),
            usize_as_f64(descriptor.fields.len()),
        ),
        (
            "field_family_count".to_owned(),
            usize_as_f64(descriptor.field_families.len()),
        ),
        (
            "group_count".to_owned(),
            usize_as_f64(descriptor.groups.len()),
        ),
        (
            "lineage_parent_count".to_owned(),
            usize_as_f64(descriptor.lineage_parents.len()),
        ),
        ("nodes".to_owned(), usize_as_f64(candidate.nodes)),
        ("nonlinear".to_owned(), f64::from(descriptor.nonlinear)),
        (
            "operator_arithmetic_count".to_owned(),
            operator_family_count(&["add", "divide", "multiply", "negate", "subtract"]),
        ),
        (
            "operator_conditional_count".to_owned(),
            operator_family_count(&["greater", "if_else"]),
        ),
        ("operator_count".to_owned(), f64::from(operator_count)),
        (
            "operator_cross_sectional_count".to_owned(),
            operator_family_count(&["group_rank", "rank", "zscore"]),
        ),
        (
            "operator_preprocess_count".to_owned(),
            operator_family_count(&["clip", "winsorize"]),
        ),
        (
            "operator_time_series_count".to_owned(),
            operator_family_count(&["ts_delta", "ts_mean", "ts_rank", "ts_std_dev", "ts_zscore"]),
        ),
        ("window_long_count".to_owned(), window_count("long")),
        ("window_medium_count".to_owned(), window_count("medium")),
        ("window_short_count".to_owned(), window_count("short")),
        (
            "window_total_count".to_owned(),
            descriptor
                .window_buckets
                .values()
                .copied()
                .map(f64::from)
                .sum(),
        ),
    ]);
    let mut family_buckets = [0_u32; FAMILY_BUCKETS];
    for family in &descriptor.field_families {
        family_buckets[field_family_bucket(family)] += 1;
    }
    for (bucket, count) in family_buckets.into_iter().enumerate() {
        features.insert(
            format!("field_family_bucket_{bucket}_count"),
            f64::from(count),
        );
    }
    features
}

fn field_family_bucket(family: &str) -> usize {
    usize::from_str_radix(&digest(family)[..2], 16).unwrap_or_default() % FAMILY_BUCKETS
}

/// Rank candidates with a separate measured-guidance artifact.
///
/// Structural scores remain byte-for-byte unchanged. Candidates without enough
/// comparable evidence sort after candidates with guidance, then retain the
/// original structural and semantic tie-breaks.
///
/// # Errors
/// Returns an error when the dataset or configuration is invalid.
pub fn prioritize_candidates(
    candidates: &[CandidateRecord],
    dataset: &FeedbackDataset,
    config: &FeedbackConfig,
) -> Result<Vec<GuidedCandidate>, FeedbackError> {
    dataset.validate()?;
    config.validate()?;
    let checksum = dataset.checksum();
    let mut guided = candidates
        .iter()
        .cloned()
        .map(|candidate| {
            let features = candidate_features(&candidate);
            let measured_guidance = estimate(&features, dataset, config, &checksum);
            GuidedCandidate {
                schema: GUIDED_CANDIDATE_SCHEMA,
                candidate,
                measured_guidance,
            }
        })
        .collect::<Vec<_>>();
    guided.sort_by(|left, right| {
        match (&left.measured_guidance, &right.measured_guidance) {
            (Some(left_guidance), Some(right_guidance)) => right_guidance
                .conservative_outcome
                .total_cmp(&left_guidance.conservative_outcome)
                .then_with(|| {
                    right_guidance
                        .weighted_outcome
                        .total_cmp(&left_guidance.weighted_outcome)
                }),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| {
            right
                .candidate
                .structural_score
                .total_cmp(&left.candidate.structural_score)
        })
        .then_with(|| {
            left.candidate
                .semantic_fingerprint
                .cmp(&right.candidate.semantic_fingerprint)
        })
    });
    Ok(guided)
}

/// Audit measured guidance with deterministic leave-one-record-out estimates.
///
/// Every target record is excluded from its own neighbor set. Error metrics
/// cover only records with enough comparable neighbors; coverage reports that
/// fraction explicitly. Sign metrics omit zero-valued outcomes and AUC is
/// emitted only when both outcome signs are present.
///
/// # Errors
/// Returns an error when the dataset or configuration is invalid.
pub fn audit_feedback(
    dataset: &FeedbackDataset,
    config: &FeedbackConfig,
) -> Result<FeedbackAuditReport, FeedbackError> {
    dataset.validate()?;
    config.validate()?;
    let checksum = dataset.checksum();
    let estimates = dataset
        .records
        .iter()
        .filter_map(|record| {
            estimate_excluding(
                &record.features,
                dataset,
                config,
                &checksum,
                Some(&record.record_id),
            )
            .map(|estimate| (record.outcome, estimate.weighted_outcome))
        })
        .collect::<Vec<_>>();
    let evaluated = estimates.len();
    let total = dataset.records.len();
    let mean_absolute_error = (evaluated > 0).then(|| {
        estimates
            .iter()
            .map(|(actual, predicted)| (actual - predicted).abs())
            .sum::<f64>()
            / usize_as_f64(evaluated)
    });
    let root_mean_square_error = (evaluated > 0).then(|| {
        (estimates
            .iter()
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
            / usize_as_f64(evaluated))
        .sqrt()
    });
    let signed = estimates
        .iter()
        .copied()
        .filter(|(actual, _)| *actual != 0.0)
        .collect::<Vec<_>>();
    let positives = signed
        .iter()
        .copied()
        .filter(|(actual, _)| *actual > 0.0)
        .collect::<Vec<_>>();
    let negatives = signed
        .iter()
        .copied()
        .filter(|(actual, _)| *actual < 0.0)
        .collect::<Vec<_>>();
    let sign_accuracy = (!signed.is_empty()).then(|| {
        usize_as_f64(
            signed
                .iter()
                .filter(|(actual, predicted)| (*actual > 0.0) == (*predicted > 0.0))
                .count(),
        ) / usize_as_f64(signed.len())
    });
    let balanced_sign_accuracy = (!positives.is_empty() && !negatives.is_empty()).then(|| {
        let true_positive_rate = usize_as_f64(
            positives
                .iter()
                .filter(|(_, predicted)| *predicted > 0.0)
                .count(),
        ) / usize_as_f64(positives.len());
        let true_negative_rate = usize_as_f64(
            negatives
                .iter()
                .filter(|(_, predicted)| *predicted <= 0.0)
                .count(),
        ) / usize_as_f64(negatives.len());
        f64::midpoint(true_positive_rate, true_negative_rate)
    });
    let roc_auc = rank_auc(&signed);
    Ok(FeedbackAuditReport {
        schema: 1,
        dataset_id: dataset.dataset_id.clone(),
        feedback_checksum: checksum,
        records: total,
        evaluated_records: evaluated,
        unguided_records: total - evaluated,
        coverage: usize_as_f64(evaluated) / usize_as_f64(total),
        mean_absolute_error,
        root_mean_square_error,
        sign_accuracy,
        balanced_sign_accuracy,
        roc_auc,
    })
}

/// Atomically write guided records as deterministic JSONL.
///
/// # Errors
/// Returns an error on serialization or durable replacement failure.
pub fn write_guided_jsonl_atomic(
    path: &Path,
    records: &[GuidedCandidate],
) -> Result<String, FeedbackError> {
    let mut content = String::new();
    for record in records {
        content.push_str(&serde_json::to_string(record).map_err(artifact::ArtifactError::from)?);
        content.push('\n');
    }
    artifact::atomic_write(path, content.as_bytes())?;
    Ok(digest(&content))
}

fn estimate(
    features: &BTreeMap<String, f64>,
    dataset: &FeedbackDataset,
    config: &FeedbackConfig,
    checksum: &str,
) -> Option<GuidanceEstimate> {
    estimate_excluding(features, dataset, config, checksum, None)
}

fn estimate_excluding(
    features: &BTreeMap<String, f64>,
    dataset: &FeedbackDataset,
    config: &FeedbackConfig,
    checksum: &str,
    excluded_record_id: Option<&str>,
) -> Option<GuidanceEstimate> {
    let mut neighbors = dataset
        .records
        .iter()
        .filter(|record| excluded_record_id != Some(record.record_id.as_str()))
        .map(|record| {
            let distance = dataset
                .feature_scales
                .iter()
                .map(|(name, scale)| {
                    ((features[name] - record.features[name]).abs() / scale).min(1.0)
                })
                .sum::<f64>()
                / usize_as_f64(dataset.feature_scales.len());
            (distance, record)
        })
        .filter(|(distance, _)| *distance <= config.maximum_distance)
        .collect::<Vec<_>>();
    neighbors.sort_by(|(left_distance, left), (right_distance, right)| {
        left_distance
            .total_cmp(right_distance)
            .then_with(|| left.record_id.cmp(&right.record_id))
    });
    neighbors.truncate(config.neighbors);
    if neighbors.len() < config.minimum_neighbors {
        return None;
    }
    let weights = neighbors
        .iter()
        .map(|(distance, record)| (1.0 - distance).max(1e-9).powi(2) * record.confidence)
        .collect::<Vec<_>>();
    let total_weight = weights.iter().sum::<f64>();
    let mean = neighbors
        .iter()
        .zip(&weights)
        .map(|((_, record), weight)| record.outcome * weight)
        .sum::<f64>()
        / total_weight;
    let variance = neighbors
        .iter()
        .zip(&weights)
        .map(|((_, record), weight)| (record.outcome - mean).powi(2) * weight)
        .sum::<f64>()
        / total_weight;
    let dispersion = variance.sqrt();
    let conservative = (mean - dispersion / usize_as_f64(neighbors.len()).sqrt()).clamp(-1.0, 1.0);
    Some(GuidanceEstimate {
        schema: 1,
        dataset_id: dataset.dataset_id.clone(),
        context_checksum: dataset.context_checksum.clone(),
        feedback_checksum: checksum.to_owned(),
        outcome_label: dataset.outcome_label.clone(),
        neighbor_count: neighbors.len(),
        nearest_distance: neighbors[0].0,
        weighted_outcome: mean,
        weighted_dispersion: dispersion,
        conservative_outcome: conservative,
        structural_score_used: false,
    })
}

fn rank_auc(signed: &[(f64, f64)]) -> Option<f64> {
    let positive_count = signed.iter().filter(|(actual, _)| *actual > 0.0).count();
    let negative_count = signed.iter().filter(|(actual, _)| *actual < 0.0).count();
    if positive_count == 0 || negative_count == 0 {
        return None;
    }
    let mut ranked = signed.to_vec();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut positive_rank_sum = 0.0;
    let mut start = 0;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len()
            && ranked[end].1.total_cmp(&ranked[start].1) == std::cmp::Ordering::Equal
        {
            end += 1;
        }
        let average_rank = f64::midpoint(usize_as_f64(start + 1), usize_as_f64(end));
        positive_rank_sum += average_rank
            * usize_as_f64(
                ranked[start..end]
                    .iter()
                    .filter(|(actual, _)| *actual > 0.0)
                    .count(),
            );
        start = end;
    }
    let positives = usize_as_f64(positive_count);
    let negatives = usize_as_f64(negative_count);
    Some((positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))
}

fn usize_as_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CANDIDATE_SCHEMA, Provenance, StructuralScoreComponents, StructuralScoreWeights,
        TransformKind, describe, parse_expression,
    };

    fn candidate(expression: &str, structural_score: f64) -> CandidateRecord {
        let expression = parse_expression(expression).unwrap();
        let provenance = Provenance {
            kind: TransformKind::Initial,
            operation: "fixture".to_owned(),
            parent_fingerprints: Vec::new(),
            requested_operation: None,
            affected_path: None,
            old_subtree_fingerprint: None,
            new_subtree_fingerprint: None,
            retry_count: 0,
        };
        CandidateRecord {
            schema: CANDIDATE_SCHEMA,
            run_id: "fixture".to_owned(),
            expression: crate::canonical(&expression),
            fingerprint: crate::fingerprint(&expression),
            semantic_fingerprint: crate::semantic_fingerprint(&expression),
            structural_score,
            score_schema: 2,
            score_components: StructuralScoreComponents::default(),
            normalized_weights: StructuralScoreWeights::default(),
            structural_descriptor: describe(&expression, &provenance),
            root_family: expression.operator().unwrap_or("field").to_owned(),
            nodes: expression.node_count(),
            depth: expression.depth(),
            provenance,
        }
    }

    fn dataset(records: Vec<FeedbackRecord>) -> FeedbackDataset {
        FeedbackDataset {
            schema: FEEDBACK_DATASET_SCHEMA,
            dataset_id: "public-synthetic-feedback-v1".to_owned(),
            context_checksum: "fixture-context".to_owned(),
            outcome_label: "bounded synthetic utility".to_owned(),
            feature_scales: BTreeMap::from([
                ("depth".to_owned(), 10.0),
                ("nodes".to_owned(), 40.0),
            ]),
            records,
        }
    }

    fn record(id: &str, nodes: f64, depth: f64, outcome: f64) -> FeedbackRecord {
        FeedbackRecord {
            record_id: id.to_owned(),
            features: BTreeMap::from([("depth".to_owned(), depth), ("nodes".to_owned(), nodes)]),
            outcome,
            confidence: 1.0,
        }
    }

    #[test]
    fn measured_guidance_is_separate_and_can_override_structural_order() {
        let simple = candidate("close", 0.9);
        let richer = candidate("rank(ts_mean(close, 20))", 0.2);
        let evidence = dataset(vec![
            record("simple-a", 1.0, 1.0, -0.8),
            record("simple-b", 2.0, 2.0, -0.6),
            record("simple-c", 3.0, 2.0, -0.4),
            record("rich-a", 4.0, 3.0, 0.6),
            record("rich-b", 5.0, 3.0, 0.8),
            record("rich-c", 6.0, 4.0, 1.0),
        ]);
        let prioritized = prioritize_candidates(
            &[simple.clone(), richer.clone()],
            &evidence,
            &FeedbackConfig {
                maximum_distance: 0.2,
                ..FeedbackConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            prioritized[0].candidate.semantic_fingerprint,
            richer.semantic_fingerprint
        );
        assert!((prioritized[0].candidate.structural_score - 0.2).abs() < f64::EPSILON);
        assert!(
            !prioritized[0]
                .measured_guidance
                .as_ref()
                .unwrap()
                .structural_score_used
        );
        assert!((simple.structural_score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn record_order_does_not_change_guidance() {
        let candidate = candidate("rank(ts_delta(close, 20))", 0.5);
        let records = vec![
            record("b", 5.0, 3.0, 0.8),
            record("a", 4.0, 3.0, 0.6),
            record("c", 6.0, 4.0, 1.0),
        ];
        let first = prioritize_candidates(
            std::slice::from_ref(&candidate),
            &dataset(records.clone()),
            &FeedbackConfig::default(),
        )
        .unwrap();
        let second = prioritize_candidates(
            &[candidate],
            &dataset(records.into_iter().rev().collect()),
            &FeedbackConfig::default(),
        )
        .unwrap();
        assert_eq!(first[0].measured_guidance, second[0].measured_guidance);
    }

    #[test]
    fn candidate_features_preserve_anonymous_family_and_operator_composition() {
        let value = candidate("rank(ts_mean(add(close, volume), 20))", 0.5);

        let features = candidate_features(&value);
        let bucket_total = (0..FAMILY_BUCKETS)
            .map(|bucket| features[&format!("field_family_bucket_{bucket}_count")])
            .sum::<f64>();

        assert!((bucket_total - 2.0).abs() < f64::EPSILON);
        assert!((features["operator_arithmetic_count"] - 1.0).abs() < f64::EPSILON);
        assert!((features["operator_cross_sectional_count"] - 1.0).abs() < f64::EPSILON);
        assert!((features["operator_time_series_count"] - 1.0).abs() < f64::EPSILON);
        assert!(features["operator_conditional_count"].abs() < f64::EPSILON);
    }

    #[test]
    fn insufficient_neighbors_produces_no_estimate() {
        let evidence = dataset(vec![record("only", 1.0, 1.0, 0.5)]);
        let prioritized = prioritize_candidates(
            &[candidate("close", 0.5)],
            &evidence,
            &FeedbackConfig::default(),
        )
        .unwrap();
        assert!(prioritized[0].measured_guidance.is_none());
    }

    #[test]
    fn feedback_audit_is_leave_one_out_and_reports_separation() {
        let feedback = dataset(vec![
            record("negative-a", 1.0, 1.0, -1.0),
            record("negative-b", 2.0, 1.0, -1.0),
            record("positive-a", 9.0, 4.0, 1.0),
            record("positive-b", 10.0, 4.0, 1.0),
        ]);
        let config = FeedbackConfig {
            schema: 1,
            neighbors: 1,
            minimum_neighbors: 1,
            maximum_distance: 1.0,
        };

        let report = audit_feedback(&feedback, &config).unwrap();
        assert_eq!(report.evaluated_records, 4);
        assert!((report.coverage - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.sign_accuracy, Some(1.0));
        assert_eq!(report.balanced_sign_accuracy, Some(1.0));
        assert_eq!(report.roc_auc, Some(1.0));
        assert_eq!(report.root_mean_square_error, Some(0.0));
    }

    #[test]
    fn malformed_feature_sets_are_rejected() {
        let mut evidence = dataset(vec![record("one", 1.0, 1.0, 0.5)]);
        evidence.records[0].features.remove("depth");
        assert!(matches!(
            evidence.validate(),
            Err(FeedbackError::FeatureSet { .. })
        ));
    }
}
