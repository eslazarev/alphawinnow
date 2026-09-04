use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CANDIDATE_SCHEMA, CandidateRecord, ExpressionError, RejectionReason, analyze_expression,
    canonical, fingerprint, parse_expression_with_catalog,
};

#[derive(Debug, Error)]
pub enum DedupError {
    #[error("invalid expression in input record {record}: {source}")]
    Expression {
        record: usize,
        source: ExpressionError,
    },
}

/// One record excluded because an earlier record has the same semantic family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateRecord {
    pub record: usize,
    pub expression: String,
    pub semantic_fingerprint: String,
    pub duplicate_of_record: usize,
}

/// One structurally proven non-informative record excluded from admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedRecord {
    pub record: usize,
    pub expression: String,
    pub reason: RejectionReason,
}

/// Deterministic partition of a valid JSONL stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DedupResult {
    pub accepted: Vec<CandidateRecord>,
    pub duplicates: Vec<DuplicateRecord>,
    pub rejected: Vec<RejectedRecord>,
}

impl DedupResult {
    /// Aggregate rejected records by stable machine-readable reason.
    #[must_use]
    pub fn rejection_counts(&self) -> std::collections::BTreeMap<RejectionReason, u64> {
        let mut counts = std::collections::BTreeMap::new();
        for rejected in &self.rejected {
            *counts.entry(rejected.reason).or_insert(0) += 1;
        }
        counts
    }
}

/// Revalidate, normalize, and partition a candidate stream without aborting on
/// provably trivial records.
///
/// # Errors
/// Returns an error when an expression is syntactically or grammatically invalid.
pub fn deduplicate(records: Vec<CandidateRecord>) -> Result<DedupResult, DedupError> {
    deduplicate_with_catalog(records, crate::operators::builtin_catalog())
}

/// Revalidate and partition candidates against an explicit public catalog.
///
/// # Errors
/// Returns an error when an expression is invalid for the supplied catalog.
pub fn deduplicate_with_catalog(
    records: Vec<CandidateRecord>,
    catalog: &crate::Catalog,
) -> Result<DedupResult, DedupError> {
    let mut seen = HashMap::<String, usize>::new();
    let mut accepted = Vec::new();
    let mut duplicates = Vec::new();
    let mut rejected = Vec::new();
    for (index, mut record) in records.into_iter().enumerate() {
        let record_number = index + 1;
        let expression =
            parse_expression_with_catalog(&record.expression, catalog).map_err(|source| {
                DedupError::Expression {
                    record: record_number,
                    source,
                }
            })?;
        let analysis = analyze_expression(&expression);
        if let Some(reason) = analysis.rejection_reason {
            rejected.push(RejectedRecord {
                record: record_number,
                expression: canonical(&expression),
                reason,
            });
            continue;
        }
        let identity = analysis.semantic_fingerprint;
        if let Some(duplicate_of_record) = seen.get(&identity) {
            duplicates.push(DuplicateRecord {
                record: record_number,
                expression: canonical(&expression),
                semantic_fingerprint: identity,
                duplicate_of_record: *duplicate_of_record,
            });
            continue;
        }
        seen.insert(identity.clone(), record_number);
        record.schema = CANDIDATE_SCHEMA;
        record.expression = canonical(&expression);
        record.fingerprint = fingerprint(&expression);
        record.semantic_fingerprint = identity;
        record.nodes = expression.node_count();
        record.depth = expression.depth();
        accepted.push(record);
    }
    Ok(DedupResult {
        accepted,
        duplicates,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use crate::{CANDIDATE_SCHEMA, Provenance, TransformKind};

    use super::*;

    fn record(expression: &str) -> CandidateRecord {
        CandidateRecord {
            schema: CANDIDATE_SCHEMA,
            run_id: String::new(),
            expression: expression.to_owned(),
            fingerprint: String::new(),
            semantic_fingerprint: String::new(),
            structural_score: 0.0,
            score_schema: 0,
            score_components: crate::artifact::StructuralScoreComponents::default(),
            normalized_weights: crate::StructuralScoreWeights::default(),
            structural_descriptor: crate::StructuralDescriptor::default(),
            root_family: "test".to_owned(),
            nodes: 0,
            depth: 0,
            provenance: Provenance {
                kind: TransformKind::Initial,
                operation: "fixture".to_owned(),
                parent_fingerprints: Vec::new(),
                requested_operation: None,
                affected_path: None,
                old_subtree_fingerprint: None,
                new_subtree_fingerprint: None,
                retry_count: 0,
            },
        }
    }

    #[test]
    fn partitions_accepted_duplicates_and_rejections_deterministically() {
        let records = vec![
            record("close"),
            record("subtract(close, close)"),
            record("multiply(close, 2)"),
            record("open"),
        ];
        let result = deduplicate(records.clone()).unwrap();
        assert_eq!(result, deduplicate(records).unwrap());
        assert_eq!(
            result
                .accepted
                .iter()
                .map(|record| record.expression.as_str())
                .collect::<Vec<_>>(),
            ["close", "open"]
        );
        assert_eq!(result.duplicates.len(), 1);
        assert_eq!(result.duplicates[0].record, 3);
        assert_eq!(result.duplicates[0].duplicate_of_record, 1);
        assert_eq!(result.rejected.len(), 1);
        assert_eq!(result.rejected[0].record, 2);
        assert_eq!(result.rejected[0].reason, RejectionReason::ProvablyZero);
    }

    #[test]
    fn negative_direction_remains_a_separate_accepted_family() {
        let result = deduplicate(vec![record("close"), record("multiply(close, -2)")]).unwrap();
        assert_eq!(result.accepted.len(), 2);
        assert!(result.duplicates.is_empty());
        assert!(result.rejected.is_empty());
    }
}
