//! Optional field-footprint coverage for an already ordered research shortlist.
//!
//! This is an allocation heuristic, not semantic deduplication or evidence of
//! return independence. Different mechanisms on identical fields remain valid
//! candidates; deferred indices are retained for other research batches.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

use crate::StructuralDescriptor;

/// Indices into the original, unmodified priority ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldDiversitySelection {
    pub schema: u32,
    pub selected: Vec<usize>,
    pub deferred_by_field_cap: Vec<usize>,
    pub remaining_after_budget: Vec<usize>,
}

/// Select up to `limit` entries, limiting each exact set of input fields.
///
/// Input order defines priority. Windows, wrappers, direction, and provenance
/// do not create additional field sets. No expression or identity is changed.
/// Empty field sets form one bucket. The cap is never relaxed to fill a budget.
/// Callers should first validate descriptors against their source expressions
/// and deduplicate candidates under their own expression semantics.
#[must_use]
pub fn select_field_diverse(
    ordered: &[StructuralDescriptor],
    limit: usize,
    max_per_field_set: NonZeroUsize,
) -> FieldDiversitySelection {
    let mut counts = BTreeMap::new();
    let mut result = FieldDiversitySelection {
        schema: 1,
        selected: Vec::new(),
        deferred_by_field_cap: Vec::new(),
        remaining_after_budget: Vec::new(),
    };
    for (index, descriptor) in ordered.iter().enumerate() {
        if result.selected.len() == limit {
            result.remaining_after_budget.push(index);
            continue;
        }
        let count = counts.entry(&descriptor.fields).or_insert(0_usize);
        if *count >= max_per_field_set.get() {
            result.deferred_by_field_cap.push(index);
        } else {
            *count += 1;
            result.selected.push(index);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn descriptor(fields: &[&str]) -> StructuralDescriptor {
        StructuralDescriptor {
            fields: fields.iter().map(|f| (*f).to_owned()).collect(),
            ..StructuralDescriptor::default()
        }
    }

    #[test]
    fn field_cap_ignores_wrapper_window_and_lineage_changes() {
        let original = descriptor(&["close", "volume"]);
        let mut variant = original.clone();
        variant.groups.insert("sector".into());
        variant.window_buckets.insert("long".into(), 1);
        variant.lineage_parents.insert("other-parent".into());
        variant.operator_histogram.insert("negate".into(), 1);
        let rows = vec![
            original,
            variant,
            descriptor(&["open"]),
            descriptor(&["high"]),
        ];
        let before = rows.clone();
        let result = select_field_diverse(&rows, 2, NonZeroUsize::new(1).unwrap());
        assert_eq!(result.selected, [0, 2]);
        assert_eq!(result.deferred_by_field_cap, [1]);
        assert_eq!(result.remaining_after_budget, [3]);
        assert_eq!(rows, before);
        assert_eq!(
            result,
            select_field_diverse(&rows, 2, NonZeroUsize::new(1).unwrap())
        );
    }

    #[test]
    fn cap_shortfall_preserves_every_index_without_backfill() {
        let rows = vec![descriptor(&["close"]); 7];
        let result = select_field_diverse(&rows, 5, NonZeroUsize::new(2).unwrap());
        assert_eq!(result.selected, [0, 1]);
        assert_eq!(result.deferred_by_field_cap, [2, 3, 4, 5, 6]);
        let all: BTreeSet<_> = result
            .selected
            .iter()
            .chain(&result.deferred_by_field_cap)
            .chain(&result.remaining_after_budget)
            .copied()
            .collect();
        assert_eq!(all, (0..7).collect());
    }

    #[test]
    fn zero_budget_empty_input_and_empty_fields_are_defined() {
        let rows = vec![descriptor(&[]); 2];
        let cap = NonZeroUsize::new(1).unwrap();
        assert_eq!(
            select_field_diverse(&rows, 0, cap).remaining_after_budget,
            [0, 1]
        );
        assert_eq!(select_field_diverse(&rows, 9, cap).selected, [0]);
        assert_eq!(select_field_diverse(&[], 9, cap).selected, [] as [usize; 0]);
    }
}
