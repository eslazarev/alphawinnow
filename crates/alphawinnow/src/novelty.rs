use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Catalog, Expr, Provenance, canonical, canonical::digest};

/// Auditable, non-financial summary used for approximate structural novelty.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralDescriptor {
    pub operator_shingles: BTreeSet<String>,
    pub operator_histogram: BTreeMap<String, u32>,
    pub ordered_paths: BTreeSet<String>,
    pub fields: BTreeSet<String>,
    pub field_families: BTreeSet<String>,
    pub window_buckets: BTreeMap<String, u32>,
    pub groups: BTreeSet<String>,
    pub nonlinear: bool,
    pub conditional: bool,
    pub lineage_parents: BTreeSet<String>,
}

impl StructuralDescriptor {
    #[must_use]
    pub fn signature(&self) -> String {
        let encoded = serde_json::to_string(self).unwrap_or_default();
        digest(&encoded)
    }
}

/// Build a structural descriptor without evaluating market behavior.
#[must_use]
pub fn describe(expression: &Expr, provenance: &Provenance) -> StructuralDescriptor {
    describe_with_catalog(expression, provenance, crate::operators::builtin_catalog())
}

/// Build a structural descriptor using the active catalog's field families.
///
/// This keeps external-catalog searches economically comparable without
/// exposing field names in downstream measured-feedback artifacts.
#[must_use]
pub fn describe_with_catalog(
    expression: &Expr,
    provenance: &Provenance,
    catalog: &Catalog,
) -> StructuralDescriptor {
    let mut descriptor = StructuralDescriptor {
        lineage_parents: provenance.parent_fingerprints.iter().cloned().collect(),
        ..StructuralDescriptor::default()
    };
    visit(expression, "$", None, catalog, &mut descriptor);
    descriptor
}

fn visit(
    expression: &Expr,
    path: &str,
    parent_operator: Option<&str>,
    catalog: &Catalog,
    descriptor: &mut StructuralDescriptor,
) {
    if let Some(operator) = expression.operator() {
        *descriptor
            .operator_histogram
            .entry(operator.to_owned())
            .or_default() += 1;
        descriptor
            .ordered_paths
            .insert(format!("{path}:{operator}"));
        if let Some(parent) = parent_operator {
            descriptor
                .operator_shingles
                .insert(format!("{parent}>{operator}"));
        }
        descriptor.nonlinear |= matches!(
            operator,
            "rank"
                | "zscore"
                | "ts_rank"
                | "ts_std_dev"
                | "ts_zscore"
                | "winsorize"
                | "clip"
                | "group_rank"
        );
        descriptor.conditional |= matches!(operator, "if_else" | "greater");
    }
    match expression {
        Expr::Field { name } => {
            descriptor.fields.insert(name.clone());
            descriptor
                .field_families
                .insert(field_family(name, catalog));
            descriptor.ordered_paths.insert(format!("{path}:field"));
        }
        Expr::Group { name } => {
            descriptor.groups.insert(name.clone());
            descriptor.ordered_paths.insert(format!("{path}:group"));
        }
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => {
            visit(arg, &format!("{path}.arg"), Some(op), catalog, descriptor);
            visit_kwargs(kwargs, path, op, catalog, descriptor);
        }
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => {
            visit(left, &format!("{path}.left"), Some(op), catalog, descriptor);
            visit(
                right,
                &format!("{path}.right"),
                Some(op),
                catalog,
                descriptor,
            );
            if matches!(
                op.as_str(),
                "ts_rank" | "ts_mean" | "ts_std_dev" | "ts_zscore" | "ts_delta"
            ) && let Expr::Scalar { value } = right.as_ref()
            {
                *descriptor
                    .window_buckets
                    .entry(window_bucket(*value).to_owned())
                    .or_default() += 1;
            }
            visit_kwargs(kwargs, path, op, catalog, descriptor);
        }
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => {
            for (index, value) in args.iter().enumerate() {
                visit(
                    value,
                    &format!("{path}.arg{index}"),
                    Some(op),
                    catalog,
                    descriptor,
                );
            }
            visit_kwargs(kwargs, path, op, catalog, descriptor);
        }
        Expr::Scalar { .. } | Expr::Bool { .. } => {
            descriptor
                .ordered_paths
                .insert(format!("{path}:{}", canonical(expression)));
        }
    }
}

fn visit_kwargs(
    kwargs: &BTreeMap<String, Expr>,
    path: &str,
    operator: &str,
    catalog: &Catalog,
    descriptor: &mut StructuralDescriptor,
) {
    for (name, value) in kwargs {
        visit(
            value,
            &format!("{path}.kw.{name}"),
            Some(operator),
            catalog,
            descriptor,
        );
    }
}

fn field_family(name: &str, catalog: &Catalog) -> String {
    catalog
        .field(name)
        .map_or_else(|| format!("public:{name}"), |field| field.family.clone())
}

fn window_bucket(value: f64) -> &'static str {
    if value <= 10.0 {
        "short"
    } else if value <= 60.0 {
        "medium"
    } else {
        "long"
    }
}

/// Normalized approximate structural distance in `0..=1`.
#[must_use]
pub fn descriptor_distance(left: &StructuralDescriptor, right: &StructuralDescriptor) -> f64 {
    let shingles = jaccard_distance(&left.operator_shingles, &right.operator_shingles);
    let paths = jaccard_distance(&left.ordered_paths, &right.ordered_paths);
    let fields = jaccard_distance(&left.fields, &right.fields);
    let families = jaccard_distance(&left.field_families, &right.field_families);
    let groups = jaccard_distance(&left.groups, &right.groups);
    let operators = histogram_distance(&left.operator_histogram, &right.operator_histogram);
    let windows = histogram_distance(&left.window_buckets, &right.window_buckets);
    let structure_flags = f64::from(
        u8::from(left.nonlinear != right.nonlinear)
            + u8::from(left.conditional != right.conditional),
    ) / 2.0;
    let lineage = jaccard_distance(&left.lineage_parents, &right.lineage_parents);
    (0.24 * shingles
        + 0.14 * operators
        + 0.16 * paths
        + 0.08 * fields
        + 0.10 * families
        + 0.08 * windows
        + 0.05 * groups
        + 0.10 * structure_flags
        + 0.05 * lineage)
        .clamp(0.0, 1.0)
}

#[must_use]
pub fn common_ancestor_depth(left: &StructuralDescriptor, right: &StructuralDescriptor) -> u8 {
    u8::from(
        !left.lineage_parents.is_disjoint(&right.lineage_parents)
            && !left.lineage_parents.is_empty(),
    )
}

fn jaccard_distance<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> f64 {
    let union = left.union(right).count();
    if union == 0 {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    1.0 - ratio(intersection, union)
}

fn histogram_distance(left: &BTreeMap<String, u32>, right: &BTreeMap<String, u32>) -> f64 {
    let keys: BTreeSet<_> = left.keys().chain(right.keys()).collect();
    let total = keys
        .iter()
        .map(|key| {
            left.get(*key)
                .copied()
                .unwrap_or(0)
                .max(right.get(*key).copied().unwrap_or(0))
        })
        .sum::<u32>();
    if total == 0 {
        return 0.0;
    }
    let difference = keys
        .iter()
        .map(|key| {
            left.get(*key)
                .copied()
                .unwrap_or(0)
                .abs_diff(right.get(*key).copied().unwrap_or(0))
        })
        .sum::<u32>();
    f64::from(difference) / f64::from(total)
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    let numerator = u32::try_from(numerator).unwrap_or(u32::MAX);
    let denominator = u32::try_from(denominator).unwrap_or(u32::MAX);
    f64::from(numerator) / f64::from(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExprKind, FieldSpec, TransformKind, parse_expression, parse_expression_with_catalog,
    };

    fn descriptor(expression: &str) -> StructuralDescriptor {
        describe(
            &parse_expression(expression).unwrap(),
            &Provenance {
                kind: TransformKind::Initial,
                operation: "fixture".to_owned(),
                parent_fingerprints: Vec::new(),
                requested_operation: None,
                affected_path: None,
                old_subtree_fingerprint: None,
                new_subtree_fingerprint: None,
                retry_count: 0,
            },
        )
    }

    #[test]
    fn near_trees_are_closer_than_independent_families() {
        let base = descriptor("rank(ts_mean(close, 20))");
        let near = descriptor("rank(ts_mean(open, 20))");
        let distant = descriptor("group_rank(volume, group(\"industry\"))");
        assert!(descriptor_distance(&base, &near) < descriptor_distance(&base, &distant));
    }

    #[test]
    fn descriptor_captures_horizon_group_and_conditionals() {
        let value = descriptor(
            "if_else(greater(volume, 2), ts_mean(close, 5), group_rank(open, group(\"sector\")))",
        );
        assert_eq!(value.window_buckets["short"], 1);
        assert!(value.groups.contains("sector"));
        assert!(value.conditional);
        assert!(value.nonlinear);
    }

    #[test]
    fn external_catalog_controls_field_families() {
        let mut catalog = crate::builtin_catalog().clone();
        catalog.fields.extend([
            FieldSpec {
                name: "local_quality_a".to_owned(),
                kind: ExprKind::Signal,
                family: "local:quality".to_owned(),
                allowed_roles: vec!["signal_input".to_owned()],
            },
            FieldSpec {
                name: "local_quality_b".to_owned(),
                kind: ExprKind::Signal,
                family: "local:quality".to_owned(),
                allowed_roles: vec!["signal_input".to_owned()],
            },
        ]);
        catalog.validate().unwrap();
        let expression =
            parse_expression_with_catalog("add(local_quality_a, local_quality_b)", &catalog)
                .unwrap();
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

        let descriptor = describe_with_catalog(&expression, &provenance, &catalog);

        assert_eq!(descriptor.fields.len(), 2);
        assert_eq!(
            descriptor.field_families,
            BTreeSet::from(["local:quality".to_owned()])
        );
    }
}
