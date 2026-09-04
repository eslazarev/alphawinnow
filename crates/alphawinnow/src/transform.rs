use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ast::{Expr, ExprKind},
    canonical::{canonical, fingerprint, semantic_fingerprint},
    operators::{self, Arity, OperatorSpec, ValueDomain},
    parser::parse_expression_with_catalog,
    tree::{ExprPath, PathEntry, PathSegment, enumerate_paths, replace_subtree, subtree_at},
};

const RETRY_BUDGET: u32 = 16;
const MAX_CONFIGURED_DEPTH: usize = 64;
const MAX_CONFIGURED_NODES: usize = 4_096;
const MAX_CONFIGURED_OPERATORS: usize = 2_048;

/// Hard grammar bounds applied to generated and transformed expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Limits {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_operators: usize,
    pub min_window: u16,
    pub max_window: u16,
    pub max_scalar_abs: u16,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_nodes: 40,
            max_operators: 16,
            min_window: 2,
            max_window: 252,
            max_scalar_abs: 1_000,
        }
    }
}

impl Limits {
    /// Validate internally consistent hard limits.
    ///
    /// # Errors
    /// Returns an explanation when a limit is zero, inverted, or unsupported.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_depth < 2 || self.max_nodes < 3 || self.max_operators == 0 {
            return Err("depth must be >= 2, nodes >= 3, and operators >= 1".to_owned());
        }
        if self.max_depth > MAX_CONFIGURED_DEPTH
            || self.max_nodes > MAX_CONFIGURED_NODES
            || self.max_operators > MAX_CONFIGURED_OPERATORS
            || self.max_operators > self.max_nodes
        {
            return Err(format!(
                "depth must be <= {MAX_CONFIGURED_DEPTH}, nodes <= {MAX_CONFIGURED_NODES}, and operators <= min({MAX_CONFIGURED_OPERATORS}, nodes)"
            ));
        }
        if self.min_window < 2 || self.max_window > 512 || self.min_window > self.max_window {
            return Err("window range must be ordered and contained in 2..=512".to_owned());
        }
        if self.max_scalar_abs == 0 {
            return Err("max_scalar_abs must be positive".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn accepts(&self, expression: &Expr) -> bool {
        self.accepts_with_catalog(expression, operators::builtin_catalog())
    }

    #[must_use]
    pub fn accepts_with_catalog(&self, expression: &Expr, catalog: &operators::Catalog) -> bool {
        expression.depth() <= self.max_depth
            && expression.node_count() <= self.max_nodes
            && expression.operator_count() <= self.max_operators
            && parameter_bounds(expression, self, catalog)
    }
}

fn parameter_bounds(expression: &Expr, limits: &Limits, catalog: &operators::Catalog) -> bool {
    match expression {
        Expr::Scalar { value } => value.abs() <= f64::from(limits.max_scalar_abs),
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => call_parameters_in_bounds(op, &[arg.as_ref()], kwargs, limits, catalog),
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => call_parameters_in_bounds(
            op,
            &[left.as_ref(), right.as_ref()],
            kwargs,
            limits,
            catalog,
        ),
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => call_parameters_in_bounds(
            op,
            &args.iter().collect::<Vec<_>>(),
            kwargs,
            limits,
            catalog,
        ),
        Expr::Field { .. } | Expr::Group { .. } | Expr::Bool { .. } => true,
    }
}

fn call_parameters_in_bounds(
    operator: &str,
    args: &[&Expr],
    kwargs: &BTreeMap<String, Expr>,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> bool {
    let Some(spec) = catalog.lookup(operator) else {
        return false;
    };
    args.iter().enumerate().all(|(index, argument)| {
        let argument = *argument;
        if !parameter_bounds(argument, limits, catalog) {
            return false;
        }
        match operator_input(spec, index).map(|input| &input.domain) {
            Some(ValueDomain::Window { .. } | ValueDomain::WindowSet { .. }) => {
                matches!(argument, Expr::Scalar { value }
                if value.fract() == 0.0
                && (f64::from(limits.min_window)..=f64::from(limits.max_window)).contains(value))
            }
            _ => true,
        }
    }) && kwargs
        .values()
        .all(|value| parameter_bounds(value, limits, catalog))
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransformValidationError {
    #[error("transformed expression does not round-trip through the public parser: {0}")]
    Parser(String),
    #[error("transformed expression exceeds configured limits")]
    Limits,
}

/// Validate the complete public grammar and configured limits after a transform.
///
/// # Errors
/// Returns a parser/type/domain or resource-limit error.
pub fn validate_transformed(
    expression: &Expr,
    limits: &Limits,
) -> Result<(), TransformValidationError> {
    validate_transformed_with_catalog(expression, limits, operators::builtin_catalog())
}

/// Validate a transformed tree against an explicitly supplied catalog.
///
/// # Errors
/// Returns a parser/type/domain or resource-limit error.
pub fn validate_transformed_with_catalog(
    expression: &Expr,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Result<(), TransformValidationError> {
    let formatted = canonical(expression);
    let parsed = parse_expression_with_catalog(&formatted, catalog)
        .map_err(|error| TransformValidationError::Parser(error.to_string()))?;
    if canonical(&parsed) != formatted {
        return Err(TransformValidationError::Parser(
            "canonical round-trip changed the expression".to_owned(),
        ));
    }
    if !limits.accepts_with_catalog(&parsed, catalog) {
        return Err(TransformValidationError::Limits);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformKind {
    Initial,
    Mutation,
    Crossover,
    FallbackGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationClass {
    Field,
    Window,
    Scalar,
    Group,
    BoundedKeyword,
    BooleanLiteral,
    Operator,
    Subtree,
}

impl MutationClass {
    const ALL: [Self; 8] = [
        Self::Field,
        Self::Window,
        Self::Scalar,
        Self::Group,
        Self::BoundedKeyword,
        Self::BooleanLiteral,
        Self::Operator,
        Self::Subtree,
    ];

    const fn operation(self) -> &'static str {
        match self {
            Self::Field => "mutate_field",
            Self::Window => "mutate_window",
            Self::Scalar => "mutate_scalar",
            Self::Group => "mutate_group",
            Self::BoundedKeyword => "mutate_bounded_keyword",
            Self::BooleanLiteral => "mutate_boolean_literal",
            Self::Operator => "replace_compatible_operator",
            Self::Subtree => "replace_typed_subtree",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub kind: TransformKind,
    pub operation: String,
    pub parent_fingerprints: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub affected_path: Option<ExprPath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub old_subtree_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_subtree_fingerprint: Option<String>,
    #[serde(default)]
    pub retry_count: u32,
}

/// Deterministically generate a valid signal expression for an index.
#[must_use]
pub fn generate(seed: u64, index: u64, limits: &Limits) -> (Expr, Provenance) {
    generate_with_catalog(seed, index, limits, operators::builtin_catalog())
}

/// Deterministically generate with an explicitly supplied public catalog.
#[must_use]
pub fn generate_with_catalog(
    seed: u64,
    index: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let mut rng = seeded(seed, index, 0x9e37_79b9_7f4a_7c15);
    let expression = fresh_signal(&mut rng, None, limits, catalog);
    let provenance = Provenance {
        kind: TransformKind::Initial,
        operation: "grammar_sample".to_owned(),
        parent_fingerprints: Vec::new(),
        requested_operation: None,
        affected_path: None,
        old_subtree_fingerprint: None,
        new_subtree_fingerprint: Some(fingerprint(&expression)),
        retry_count: 0,
    };
    (expression, provenance)
}

/// Deterministic indexed typed mutation with a bounded retry budget.
#[must_use]
pub fn mutate(parent: &Expr, seed: u64, ordinal: u64, limits: &Limits) -> (Expr, Provenance) {
    mutate_with_catalog(parent, seed, ordinal, limits, operators::builtin_catalog())
}

/// Deterministic indexed typed mutation with an explicit public catalog.
#[must_use]
pub fn mutate_with_catalog(
    parent: &Expr,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let mut rng = seeded(seed, ordinal, 0xd1b5_4a32_d192_ed03);
    let class = MutationClass::ALL[rng.random_range(0..MutationClass::ALL.len())];
    mutate_with_rng(parent, class, &mut rng, limits, catalog)
}

/// Deterministically request one specific mutation class.
#[must_use]
pub fn mutate_with_class(
    parent: &Expr,
    class: MutationClass,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
) -> (Expr, Provenance) {
    let mut rng = seeded(
        seed,
        ordinal,
        0xa076_1d64_78bd_642f ^ mutation_domain(class),
    );
    mutate_with_rng(
        parent,
        class,
        &mut rng,
        limits,
        operators::builtin_catalog(),
    )
}

fn mutate_with_rng(
    parent: &Expr,
    class: MutationClass,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let requested = class.operation().to_owned();
    for retry in 0..RETRY_BUDGET {
        let Some((path, replacement)) = mutation_replacement(parent, class, rng, limits, catalog)
        else {
            continue;
        };
        let Some(old) = subtree_at(parent, &path) else {
            continue;
        };
        if canonical(old) == canonical(&replacement) {
            continue;
        }
        let Ok(candidate) = replace_subtree(parent, &path, replacement.clone()) else {
            continue;
        };
        if candidate == *parent
            || validate_transformed_with_catalog(&candidate, limits, catalog).is_err()
        {
            continue;
        }
        return (
            candidate,
            successful_provenance(
                TransformKind::Mutation,
                &requested,
                path,
                old,
                &replacement,
                retry,
                vec![semantic_fingerprint(parent)],
            ),
        );
    }
    fallback_generation(parent, &requested, rng, limits, RETRY_BUDGET, catalog)
}

/// Deterministic typed subtree crossover using compatible indexed paths.
#[must_use]
pub fn crossover(
    left: &Expr,
    right: &Expr,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
) -> (Expr, Provenance) {
    crossover_with_catalog(
        left,
        right,
        seed,
        ordinal,
        limits,
        operators::builtin_catalog(),
    )
}

/// Deterministic typed subtree crossover with an explicit public catalog.
#[must_use]
pub fn crossover_with_catalog(
    left: &Expr,
    right: &Expr,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let mut rng = seeded(seed, ordinal, 0x94d0_49bb_1331_11eb);
    let receivers: Vec<_> = enumerate_paths(left)
        .into_iter()
        .filter(|entry| !entry.path.segments.is_empty())
        .collect();
    let donors = enumerate_paths(right);
    let pairs: Vec<_> = receivers
        .iter()
        .flat_map(|receiver| {
            donors
                .iter()
                .filter(move |donor| receiver.allowed_kinds.contains(donor.subtree_kind))
                .map(move |donor| (receiver.clone(), donor.clone()))
        })
        .collect();
    crossover_pairs(left, right, &pairs, &mut rng, limits, catalog)
}

/// Deterministic exact-path crossover, primarily for auditable replay and tests.
#[must_use]
pub fn crossover_with_paths(
    left: &Expr,
    right: &Expr,
    receiver_path: &ExprPath,
    donor_path: &ExprPath,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
) -> (Expr, Provenance) {
    let mut rng = seeded(seed, ordinal, 0xe703_7ed1_a0b4_28db);
    let receiver = enumerate_paths(left)
        .into_iter()
        .find(|entry| entry.path == *receiver_path);
    let donor = enumerate_paths(right)
        .into_iter()
        .find(|entry| entry.path == *donor_path);
    let pairs = match (receiver, donor) {
        (Some(receiver), Some(donor)) if receiver.allowed_kinds.contains(donor.subtree_kind) => {
            vec![(receiver, donor)]
        }
        _ => Vec::new(),
    };
    crossover_pairs(
        left,
        right,
        &pairs,
        &mut rng,
        limits,
        operators::builtin_catalog(),
    )
}

fn crossover_pairs(
    left: &Expr,
    right: &Expr,
    pairs: &[(PathEntry, PathEntry)],
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let requested = "typed_subtree_crossover";
    let start = if pairs.is_empty() {
        0
    } else {
        rng.random_range(0..pairs.len())
    };
    let mut attempts = 0;
    for retry in 0..RETRY_BUDGET {
        if pairs.is_empty() {
            break;
        }
        attempts += 1;
        let (receiver, donor) = &pairs[(start + retry as usize) % pairs.len()];
        let Some(old) = subtree_at(left, &receiver.path) else {
            continue;
        };
        let Some(replacement) = subtree_at(right, &donor.path) else {
            continue;
        };
        if canonical(old) == canonical(replacement) {
            continue;
        }
        let Ok(candidate) = replace_subtree(left, &receiver.path, replacement.clone()) else {
            continue;
        };
        if candidate == *left
            || validate_transformed_with_catalog(&candidate, limits, catalog).is_err()
        {
            continue;
        }
        return (
            candidate,
            successful_provenance(
                TransformKind::Crossover,
                requested,
                receiver.path.clone(),
                old,
                replacement,
                retry,
                vec![semantic_fingerprint(left), semantic_fingerprint(right)],
            ),
        );
    }
    fallback_generation(left, requested, rng, limits, attempts, catalog)
}

#[allow(clippy::too_many_arguments)]
fn successful_provenance(
    kind: TransformKind,
    operation: &str,
    path: ExprPath,
    old: &Expr,
    new: &Expr,
    retry_count: u32,
    parent_fingerprints: Vec<String>,
) -> Provenance {
    Provenance {
        kind,
        operation: operation.to_owned(),
        parent_fingerprints,
        requested_operation: Some(operation.to_owned()),
        affected_path: Some(path),
        old_subtree_fingerprint: Some(fingerprint(old)),
        new_subtree_fingerprint: Some(fingerprint(new)),
        retry_count,
    }
}

fn fallback_generation(
    parent: &Expr,
    requested: &str,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    retry_count: u32,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let expression = fresh_signal(rng, Some(parent), limits, catalog);
    let provenance = Provenance {
        kind: TransformKind::FallbackGeneration,
        operation: "fallback_generation".to_owned(),
        parent_fingerprints: Vec::new(),
        requested_operation: Some(requested.to_owned()),
        affected_path: None,
        old_subtree_fingerprint: None,
        new_subtree_fingerprint: Some(fingerprint(&expression)),
        retry_count,
    };
    (expression, provenance)
}

fn mutation_replacement(
    parent: &Expr,
    class: MutationClass,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Option<(ExprPath, Expr)> {
    let entries = enumerate_paths(parent);
    let eligible: Vec<_> = entries
        .into_iter()
        .filter(|entry| mutation_path_matches(parent, entry, class, catalog))
        .collect();
    let entry = eligible.get(rng.random_range(0..eligible.len().max(1)))?;
    let old = subtree_at(parent, &entry.path)?;
    let replacement = match class {
        MutationClass::Field => mutate_field(old, rng, catalog)?,
        MutationClass::Window => mutate_window(old, rng, limits, catalog)?,
        MutationClass::Scalar => mutate_scalar(old, rng, limits, catalog)?,
        MutationClass::Group => mutate_group(old, rng, catalog)?,
        MutationClass::BoundedKeyword => mutate_keyword(parent, &entry.path, rng, catalog)?,
        MutationClass::BooleanLiteral => match old {
            Expr::Bool { value } => Expr::Bool { value: !value },
            _ => return None,
        },
        MutationClass::Operator => replace_operator(old, rng, catalog)?,
        MutationClass::Subtree => {
            let depth = rng.random_range(1..=3);
            let kinds: Vec<_> = entry.allowed_kinds.iter().collect();
            let kind = kinds[rng.random_range(0..kinds.len())];
            generate_kind(rng, kind, depth, limits, catalog)
        }
    };
    Some((entry.path.clone(), replacement))
}

fn mutation_path_matches(
    parent: &Expr,
    entry: &PathEntry,
    class: MutationClass,
    catalog: &operators::Catalog,
) -> bool {
    let Some(subtree) = subtree_at(parent, &entry.path) else {
        return false;
    };
    match class {
        MutationClass::Field => matches!(subtree, Expr::Field { .. }),
        MutationClass::Window => is_window_path(parent, &entry.path, catalog),
        MutationClass::Scalar => {
            matches!(subtree, Expr::Scalar { .. })
                && !is_window_path(parent, &entry.path, catalog)
                && !matches!(
                    entry.path.segments.last(),
                    Some(PathSegment::Keyword { .. })
                )
        }
        MutationClass::Group => matches!(subtree, Expr::Group { .. }),
        MutationClass::BoundedKeyword => {
            matches!(
                entry.path.segments.last(),
                Some(PathSegment::Keyword { .. })
            )
        }
        MutationClass::BooleanLiteral => matches!(subtree, Expr::Bool { .. }),
        MutationClass::Operator => {
            subtree.operator().is_some() && has_operator_replacement(subtree, catalog)
        }
        MutationClass::Subtree => !entry.path.segments.is_empty(),
    }
}

fn is_window_path(parent: &Expr, path: &ExprPath, catalog: &operators::Catalog) -> bool {
    if !matches!(path.segments.last(), Some(PathSegment::BinaryRight)) {
        return false;
    }
    let parent_path = ExprPath {
        segments: path.segments[..path.segments.len() - 1].to_vec(),
    };
    let Some(owner) = subtree_at(parent, &parent_path) else {
        return false;
    };
    let Some(operator) = owner.operator().and_then(|name| catalog.lookup(name)) else {
        return false;
    };
    let Some(segment) = path.segments.last() else {
        return false;
    };
    let index = match segment {
        PathSegment::UnaryArg | PathSegment::BinaryLeft => 0,
        PathSegment::BinaryRight => 1,
        PathSegment::VariadicArg { index } => *index,
        PathSegment::Keyword { .. } => return false,
    };
    operator_input(operator, index).is_some_and(|input| {
        matches!(
            input.domain,
            ValueDomain::Window { .. } | ValueDomain::WindowSet { .. }
        )
    })
}

fn operator_input(operator: &OperatorSpec, index: usize) -> Option<&operators::ArgumentSpec> {
    match operator.arity {
        Arity::Variadic { .. } => operator.inputs.first(),
        _ => operator.inputs.get(index),
    }
}

fn mutate_field(old: &Expr, rng: &mut ChaCha8Rng, catalog: &operators::Catalog) -> Option<Expr> {
    let Expr::Field { name } = old else {
        return None;
    };
    let fields: Vec<_> = catalog
        .fields
        .iter()
        .filter(|field| field.kind == ExprKind::Signal)
        .map(|field| field.name.as_str())
        .collect();
    let replacement = choose_different(&fields, name, rng)?;
    Some(Expr::Field {
        name: replacement.to_owned(),
    })
}

fn mutate_window(
    old: &Expr,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Option<Expr> {
    let Expr::Scalar { value: old } = old else {
        return None;
    };
    let values: Vec<_> = window_values(limits, catalog)
        .into_iter()
        .filter(|value| f64::from(*value).to_bits() != old.to_bits())
        .collect();
    let value = *values.get(rng.random_range(0..values.len().max(1)))?;
    Some(Expr::Scalar {
        value: f64::from(value),
    })
}

fn mutate_scalar(
    old: &Expr,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Option<Expr> {
    let Expr::Scalar { value: old } = old else {
        return None;
    };
    let values: Vec<_> = scalar_values(limits, catalog)
        .into_iter()
        .filter(|value| *value != 0.0 && value.to_bits() != old.to_bits())
        .collect();
    let value = *values.get(rng.random_range(0..values.len().max(1)))?;
    Some(Expr::Scalar { value })
}

fn mutate_group(old: &Expr, rng: &mut ChaCha8Rng, catalog: &operators::Catalog) -> Option<Expr> {
    let Expr::Group { name } = old else {
        return None;
    };
    let groups: Vec<_> = catalog
        .groups
        .iter()
        .map(|group| group.name.as_str())
        .collect();
    let replacement = choose_different(&groups, name, rng)?;
    Some(Expr::Group {
        name: replacement.to_owned(),
    })
}

fn choose_different<'a>(values: &'a [&str], old: &str, rng: &mut ChaCha8Rng) -> Option<&'a str> {
    let choices: Vec<_> = values
        .iter()
        .copied()
        .filter(|value| *value != old)
        .collect();
    choices
        .get(rng.random_range(0..choices.len().max(1)))
        .copied()
}

fn mutate_keyword(
    parent: &Expr,
    path: &ExprPath,
    rng: &mut ChaCha8Rng,
    catalog: &operators::Catalog,
) -> Option<Expr> {
    let PathSegment::Keyword { name } = path.segments.last()? else {
        return None;
    };
    let owner_path = ExprPath {
        segments: path.segments[..path.segments.len() - 1].to_vec(),
    };
    let owner = subtree_at(parent, &owner_path)?;
    let old = match subtree_at(parent, path)? {
        Expr::Scalar { value } => *value,
        _ => return None,
    };
    let spec = catalog.lookup(owner.operator()?)?;
    let keyword = spec.keywords.iter().find(|keyword| keyword.name == *name)?;
    let values: Vec<_> = keyword_values(keyword)
        .into_iter()
        .filter(|value| value.to_bits() != old.to_bits())
        .filter(|value| keyword_replacement_is_valid(owner, name, *value, catalog))
        .collect();
    values
        .get(rng.random_range(0..values.len().max(1)))
        .copied()
        .map(|value| Expr::Scalar { value })
}

fn keyword_values(keyword: &operators::KeywordSpec) -> Vec<f64> {
    let mut values = Vec::new();
    let mut value = keyword.min;
    while value <= keyword.max + keyword.step / 10.0 && values.len() < 10_000 {
        values.push(value);
        value += keyword.step;
    }
    values
}

fn keyword_replacement_is_valid(
    owner: &Expr,
    name: &str,
    value: f64,
    catalog: &operators::Catalog,
) -> bool {
    let (operator, args, mut kwargs) = match owner {
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => (op, vec![arg.as_ref().clone()], kwargs.clone()),
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => (
            op,
            vec![left.as_ref().clone(), right.as_ref().clone()],
            kwargs.clone(),
        ),
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => (op, args.clone(), kwargs.clone()),
        _ => return false,
    };
    kwargs.insert(name.to_owned(), Expr::Scalar { value });
    operators::build_call_with_catalog(catalog, operator, args, kwargs).is_ok()
}

fn has_operator_replacement(expression: &Expr, catalog: &operators::Catalog) -> bool {
    operator_replacements(expression, &mut ChaCha8Rng::seed_from_u64(0), catalog).len() > 1
}

fn replace_operator(
    expression: &Expr,
    rng: &mut ChaCha8Rng,
    catalog: &operators::Catalog,
) -> Option<Expr> {
    let current = expression.operator()?;
    let replacements: Vec<_> = operator_replacements(expression, rng, catalog)
        .into_iter()
        .filter(|candidate| candidate.operator() != Some(current))
        .collect();
    replacements
        .get(rng.random_range(0..replacements.len().max(1)))
        .cloned()
}

fn operator_replacements(
    expression: &Expr,
    rng: &mut ChaCha8Rng,
    catalog: &operators::Catalog,
) -> Vec<Expr> {
    let (args, output) = match expression {
        Expr::UnaryCall { arg, kind, .. } => (vec![arg.as_ref().clone()], *kind),
        Expr::BinaryCall {
            left, right, kind, ..
        } => (vec![left.as_ref().clone(), right.as_ref().clone()], *kind),
        Expr::VariadicCall { args, kind, .. } => (args.clone(), *kind),
        _ => return Vec::new(),
    };
    catalog
        .operators
        .iter()
        .filter(|spec| spec.output == output)
        .filter_map(|spec| {
            let kwargs = generated_kwargs(spec, rng);
            operators::build_call_with_catalog(catalog, &spec.name, args.clone(), kwargs).ok()
        })
        .collect()
}

fn generated_kwargs(spec: &OperatorSpec, rng: &mut ChaCha8Rng) -> BTreeMap<String, Expr> {
    let mut kwargs = BTreeMap::new();
    for keyword in &spec.keywords {
        let values = keyword_values(keyword);
        let value = values[rng.random_range(0..values.len())];
        kwargs.insert(keyword.name.clone(), Expr::Scalar { value });
    }
    for constraint in &spec.keyword_constraints {
        match constraint {
            operators::KeywordConstraint::LessThan { left, right } => {
                let left_spec = spec
                    .keywords
                    .iter()
                    .find(|keyword| keyword.name == *left)
                    .expect("validated keyword constraint");
                let right_spec = spec
                    .keywords
                    .iter()
                    .find(|keyword| keyword.name == *right)
                    .expect("validated keyword constraint");
                let left_value = match &kwargs[left] {
                    Expr::Scalar { value } => *value,
                    _ => unreachable!(),
                };
                let right_value = match &kwargs[right] {
                    Expr::Scalar { value } => *value,
                    _ => unreachable!(),
                };
                if left_value >= right_value {
                    kwargs.insert(
                        left.clone(),
                        Expr::Scalar {
                            value: left_spec.min,
                        },
                    );
                    kwargs.insert(
                        right.clone(),
                        Expr::Scalar {
                            value: right_spec.max,
                        },
                    );
                }
            }
        }
    }
    kwargs
}

fn fresh_signal(
    rng: &mut ChaCha8Rng,
    parent: Option<&Expr>,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Expr {
    for _ in 0..RETRY_BUDGET {
        let depth = rng.random_range(2..=limits.max_depth.min(5));
        let sampled = generate_kind(rng, ExprKind::Signal, depth, limits, catalog);
        if parent.is_none_or(|parent| sampled != *parent)
            && validate_transformed_with_catalog(&sampled, limits, catalog).is_ok()
        {
            return sampled;
        }
    }
    for field in &catalog.fields {
        let candidate = Expr::Field {
            name: field.name.clone(),
        };
        if parent.is_none_or(|parent| candidate != *parent) {
            return candidate;
        }
    }
    fallback_field(rng, catalog)
}

fn generate_kind(
    rng: &mut ChaCha8Rng,
    kind: ExprKind,
    depth: usize,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Expr {
    match kind {
        ExprKind::Signal if depth <= 1 || rng.random_bool(0.28) => fallback_field(rng, catalog),
        ExprKind::Signal => generate_call(rng, kind, depth, limits, catalog)
            .unwrap_or_else(|| fallback_field(rng, catalog)),
        ExprKind::Scalar => nonzero_scalar(rng, limits, catalog),
        ExprKind::Group => Expr::Group {
            name: catalog.groups[rng.random_range(0..catalog.groups.len())]
                .name
                .clone(),
        },
        ExprKind::Boolean if depth <= 1 || rng.random_bool(0.4) => Expr::Bool {
            value: rng.random_bool(0.5),
        },
        ExprKind::Boolean => {
            generate_call(rng, kind, depth, limits, catalog).unwrap_or(Expr::Bool {
                value: rng.random_bool(0.5),
            })
        }
    }
}

fn generate_call(
    rng: &mut ChaCha8Rng,
    output: ExprKind,
    depth: usize,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Option<Expr> {
    let candidates: Vec<_> = catalog
        .operators
        .iter()
        .filter(|operator| operator.output == output && operator.generation_weight > 0)
        .collect();
    let total: u32 = candidates
        .iter()
        .map(|operator| u32::from(operator.generation_weight))
        .sum();
    if total == 0 {
        return None;
    }
    let mut draw = rng.random_range(0..total);
    let operator = candidates.into_iter().find(|operator| {
        let weight = u32::from(operator.generation_weight);
        if draw < weight {
            true
        } else {
            draw -= weight;
            false
        }
    })?;
    let count = match operator.arity {
        Arity::Unary => 1,
        Arity::Binary => 2,
        Arity::Exact { count } => count,
        Arity::Variadic { min, .. } => min,
    };
    let mut args = Vec::with_capacity(count);
    for index in 0..count {
        let input = operator_input(operator, index)?;
        let required = operator.required_input_kinds.get(index).copied();
        let kind = required.unwrap_or_else(|| input.kinds[rng.random_range(0..input.kinds.len())]);
        let value = match &input.domain {
            ValueDomain::Window { min, max } => {
                let min = (*min).max(limits.min_window);
                let max = (*max).min(limits.max_window);
                if min > max {
                    return None;
                }
                Expr::Scalar {
                    value: f64::from(rng.random_range(min..=max)),
                }
            }
            ValueDomain::WindowSet { values } => {
                let allowed = values
                    .iter()
                    .copied()
                    .filter(|value| *value >= limits.min_window && *value <= limits.max_window)
                    .collect::<Vec<_>>();
                Expr::Scalar {
                    value: f64::from(*allowed.get(rng.random_range(0..allowed.len().max(1)))?),
                }
            }
            ValueDomain::NonZeroScalar => nonzero_scalar(rng, limits, catalog),
            ValueDomain::Any => generate_kind(rng, kind, depth - 1, limits, catalog),
        };
        args.push(value);
    }
    let kwargs = generated_kwargs(operator, rng);
    operators::build_call_with_catalog(catalog, &operator.name, args, kwargs).ok()
}

fn nonzero_scalar(rng: &mut ChaCha8Rng, limits: &Limits, catalog: &operators::Catalog) -> Expr {
    let values: Vec<_> = scalar_values(limits, catalog)
        .into_iter()
        .filter(|value| *value != 0.0)
        .collect();
    Expr::Scalar {
        value: values
            .get(rng.random_range(0..values.len().max(1)))
            .copied()
            .unwrap_or(1.0),
    }
}

fn scalar_values(limits: &Limits, catalog: &operators::Catalog) -> Vec<f64> {
    let bound = f64::from(limits.max_scalar_abs);
    let minimum = catalog.scalar_domain.min.max(-bound);
    let maximum = catalog.scalar_domain.max.min(bound);
    let mut values = Vec::new();
    let mut value = minimum;
    while value <= maximum + catalog.scalar_domain.step / 10.0 && values.len() < 100_000 {
        values.push(value);
        value += catalog.scalar_domain.step;
    }
    values
}

fn window_values(limits: &Limits, catalog: &operators::Catalog) -> Vec<u16> {
    let mut values = std::collections::BTreeSet::new();
    for input in catalog
        .operators
        .iter()
        .flat_map(|operator| operator.inputs.iter())
    {
        match &input.domain {
            ValueDomain::Window { min, max } => {
                values.extend((*min).max(limits.min_window)..=(*max).min(limits.max_window));
            }
            ValueDomain::WindowSet { values: configured } => values.extend(
                configured
                    .iter()
                    .copied()
                    .filter(|value| *value >= limits.min_window && *value <= limits.max_window),
            ),
            ValueDomain::Any | ValueDomain::NonZeroScalar => {}
        }
    }
    values.into_iter().collect()
}

fn fallback_field(rng: &mut ChaCha8Rng, catalog: &operators::Catalog) -> Expr {
    Expr::Field {
        name: catalog.fields[rng.random_range(0..catalog.fields.len())]
            .name
            .clone(),
    }
}

fn mutation_domain(class: MutationClass) -> u64 {
    match class {
        MutationClass::Field => 1,
        MutationClass::Window => 2,
        MutationClass::Scalar => 3,
        MutationClass::Group => 4,
        MutationClass::BoundedKeyword => 5,
        MutationClass::BooleanLiteral => 6,
        MutationClass::Operator => 7,
        MutationClass::Subtree => 8,
    }
}

fn seeded(seed: u64, ordinal: u64, domain: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ ordinal.wrapping_mul(domain).rotate_left(17) ^ domain)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::parse_expression;
    use proptest::prelude::*;

    fn rich_parent() -> Expr {
        parse_expression(
            "add(group_rank(ts_mean(close, 20), group(\"sector\")), winsorize(if_else(true, multiply(open, 2), ts_delta(volume, 5)), std=2))",
        )
        .unwrap()
    }

    #[test]
    fn fixed_workload_covers_every_mutation_class() {
        let parent = rich_parent();
        let limits = Limits::default();
        let mut operations = BTreeSet::new();
        for (index, class) in MutationClass::ALL.into_iter().enumerate() {
            let (expression, provenance) =
                mutate_with_class(&parent, class, 41, u64::try_from(index).unwrap(), &limits);
            assert_eq!(provenance.kind, TransformKind::Mutation, "{class:?}");
            let path = provenance.affected_path.as_ref().unwrap();
            assert_eq!(
                provenance.old_subtree_fingerprint.as_deref(),
                Some(fingerprint(subtree_at(&parent, path).unwrap()).as_str())
            );
            assert_eq!(
                provenance.new_subtree_fingerprint.as_deref(),
                Some(fingerprint(subtree_at(&expression, path).unwrap()).as_str())
            );
            assert_eq!(provenance.parent_fingerprints.len(), 1);
            operations.insert(provenance.operation);
        }
        assert_eq!(operations.len(), MutationClass::ALL.len());
    }

    #[test]
    fn every_generation_enabled_catalog_operator_is_reachable() {
        let catalog = operators::builtin_catalog();
        let expected: BTreeSet<_> = catalog
            .operators
            .iter()
            .filter(|operator| operator.generation_weight > 0)
            .map(|operator| operator.name.clone())
            .collect();
        let mut observed = BTreeSet::new();
        for index in 0..10_000_u64 {
            let (expression, _) = generate(20_260_828, index, &Limits::default());
            collect_operators(&expression, &mut observed);
            if observed.is_superset(&expected) {
                break;
            }
        }
        assert_eq!(observed.intersection(&expected).count(), expected.len());
        for required in ["clip", "divide", "negate"] {
            assert!(observed.contains(required));
        }
    }

    #[test]
    fn synthetic_operator_generation_requires_only_catalog_data() {
        let mut catalog = operators::builtin_catalog().clone();
        catalog.operators.push(OperatorSpec {
            name: "synthetic_smooth".to_owned(),
            arity: Arity::Unary,
            inputs: vec![operators::ArgumentSpec {
                kinds: vec![ExprKind::Signal],
                domain: ValueDomain::Any,
            }],
            required_input_kinds: Vec::new(),
            output: ExprKind::Signal,
            commutative: false,
            associative: false,
            keywords: Vec::new(),
            keyword_constraints: Vec::new(),
            generation_weight: u16::MAX,
            parse_only_reason: None,
        });
        catalog.validate().unwrap();
        let (expression, _) = generate_with_catalog(7, 11, &Limits::default(), &catalog);
        let mut observed = BTreeSet::new();
        collect_operators(&expression, &mut observed);
        assert!(observed.contains("synthetic_smooth"));
        validate_transformed_with_catalog(&expression, &Limits::default(), &catalog).unwrap();
    }

    fn collect_operators(expression: &Expr, result: &mut BTreeSet<String>) {
        if let Some(operator) = expression.operator() {
            result.insert(operator.to_owned());
        }
        for child in expression.children() {
            collect_operators(child, result);
        }
    }

    #[test]
    fn ten_thousand_typed_transforms_round_trip_across_all_value_kinds() {
        let parent = rich_parent();
        let right = parse_expression(
            "if_else(greater(high, 3), clip(low, lower=-2, upper=2), group_rank(returns, group(\"country\")))",
        )
        .unwrap();
        let limits = Limits::default();
        let mut touched_kinds = BTreeSet::new();
        for index in 0..10_000_u64 {
            let (expression, provenance) = if index.is_multiple_of(3) {
                crossover(&parent, &right, 101, index, &limits)
            } else {
                let class =
                    MutationClass::ALL[usize::try_from(index).unwrap() % MutationClass::ALL.len()];
                mutate_with_class(&parent, class, 103, index, &limits)
            };
            validate_transformed(&expression, &limits).unwrap();
            let formatted = canonical(&expression);
            assert_eq!(canonical(&parse_expression(&formatted).unwrap()), formatted);
            if let Some(path) = provenance.affected_path
                && let Some(kind) = subtree_at(&parent, &path).map(Expr::kind)
            {
                touched_kinds.insert(kind);
            }
        }
        assert_eq!(
            touched_kinds,
            BTreeSet::from([
                ExprKind::Signal,
                ExprKind::Scalar,
                ExprKind::Boolean,
                ExprKind::Group,
            ])
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn arbitrary_seed_property_workload_is_typed_bounded_and_deterministic(seed: u64) {
            let parent = rich_parent();
            let right = parse_expression(
                "if_else(greater(high, 3), clip(low, lower=-2, upper=2), group_rank(returns, group(\"country\")))",
            )
            .unwrap();
            let limits = Limits::default();
            for slot in 0..100_u64 {
                let class = MutationClass::ALL
                    [usize::try_from(slot).unwrap() % MutationClass::ALL.len()];
                let left_result = mutate_with_class(&parent, class, seed, slot, &limits);
                let repeated = mutate_with_class(&parent, class, seed, slot, &limits);
                prop_assert_eq!(&left_result, &repeated);
                prop_assert!(validate_transformed(&left_result.0, &limits).is_ok());
                let crossed = crossover(&parent, &right, seed, slot, &limits);
                prop_assert!(validate_transformed(&crossed.0, &limits).is_ok());
            }
        }
    }

    #[test]
    fn transform_identity_is_deterministic_from_explicit_seed_material() {
        let parent = rich_parent();
        let right = parse_expression("rank(add(high, low))").unwrap();
        let limits = Limits::default();
        assert_eq!(
            mutate(&parent, 7, 9, &limits),
            mutate(&parent, 7, 9, &limits)
        );
        assert_eq!(
            crossover(&parent, &right, 11, 13, &limits),
            crossover(&parent, &right, 11, 13, &limits)
        );
    }

    #[test]
    fn forced_limit_failure_has_truthful_fallback_provenance() {
        let left = parse_expression("rank(close)").unwrap();
        let right = parse_expression("rank(open)").unwrap();
        let limits = Limits {
            max_depth: 2,
            max_nodes: 3,
            max_operators: 1,
            ..Limits::default()
        };
        let receiver = ExprPath {
            segments: vec![PathSegment::UnaryArg],
        };
        let donor = ExprPath::default();
        let (expression, provenance) =
            crossover_with_paths(&left, &right, &receiver, &donor, 5, 8, &limits);
        validate_transformed(&expression, &limits).unwrap();
        assert_eq!(provenance.kind, TransformKind::FallbackGeneration);
        assert_eq!(provenance.operation, "fallback_generation");
        assert_eq!(
            provenance.requested_operation.as_deref(),
            Some("typed_subtree_crossover")
        );
        assert!(provenance.parent_fingerprints.is_empty());
        assert!(provenance.affected_path.is_none());
        assert_eq!(provenance.retry_count, RETRY_BUDGET);
    }

    #[test]
    fn incompatible_crossover_paths_report_zero_attempts() {
        let left = parse_expression("rank(close)").unwrap();
        let right = parse_expression("if_else(true, open, high)").unwrap();
        let receiver = ExprPath {
            segments: vec![PathSegment::UnaryArg],
        };
        let donor = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 0 }],
        };
        let limits = Limits::default();
        let (expression, provenance) =
            crossover_with_paths(&left, &right, &receiver, &donor, 5, 8, &limits);

        validate_transformed(&expression, &limits).unwrap();
        assert_eq!(provenance.kind, TransformKind::FallbackGeneration);
        assert_eq!(provenance.retry_count, 0);
    }

    #[test]
    fn crossover_can_replace_one_multiply_signal_with_a_scalar() {
        let left = parse_expression("multiply(close, open)").unwrap();
        let right = parse_expression("divide(high, 2)").unwrap();
        let receiver = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 0 }],
        };
        let donor = ExprPath {
            segments: vec![PathSegment::BinaryRight],
        };
        let limits = Limits::default();
        let (expression, provenance) =
            crossover_with_paths(&left, &right, &receiver, &donor, 7, 9, &limits);

        assert_eq!(provenance.kind, TransformKind::Crossover);
        assert_eq!(canonical(&expression), "multiply(2, open)");
        validate_transformed(&expression, &limits).unwrap();
    }

    #[test]
    fn crossover_cannot_replace_the_last_multiply_signal_with_a_scalar() {
        let left = parse_expression("multiply(close, 3)").unwrap();
        let right = parse_expression("divide(high, 2)").unwrap();
        let receiver = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 0 }],
        };
        let donor = ExprPath {
            segments: vec![PathSegment::BinaryRight],
        };
        let limits = Limits::default();
        let (expression, provenance) =
            crossover_with_paths(&left, &right, &receiver, &donor, 7, 9, &limits);

        assert_eq!(provenance.kind, TransformKind::FallbackGeneration);
        assert_eq!(provenance.retry_count, RETRY_BUDGET);
        validate_transformed(&expression, &limits).unwrap();
    }
}
