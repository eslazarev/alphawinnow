use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    analysis::analyze_expression,
    ast::{Expr, ExprKind},
    canonical::{canonical, digest, fingerprint, semantic_fingerprint},
    operators::{self, Arity, OperatorSpec, ValueDomain},
    parser::parse_expression_with_catalog,
    tree::{ExprPath, PathEntry, PathSegment, enumerate_paths, replace_subtree, subtree_at},
};

const RETRY_BUDGET: u32 = 16;
const MAX_CATALOG_SCAFFOLDS: usize = 64;
const MAX_CATALOG_SCAFFOLD_UNIVERSE: usize = 4_096;
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
        Expr::Scalar { value } => {
            let domain = &catalog.scalar_domain;
            let offset = (value - domain.min) / domain.step;
            value.is_finite()
                && value.abs() <= f64::from(limits.max_scalar_abs)
                && (domain.min..=domain.max).contains(value)
                && (offset - offset.round()).abs() <= 1e-8
        }
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
        match operator_input(spec, index).map(|input| &input.domain) {
            Some(ValueDomain::Window { .. } | ValueDomain::WindowSet { .. }) => {
                matches!(argument, Expr::Scalar { value }
                if value.fract() == 0.0
                && (f64::from(limits.min_window)..=f64::from(limits.max_window)).contains(value))
            }
            _ => parameter_bounds(argument, limits, catalog),
        }
    }) && kwargs
        .values()
        // Keyword domains are checked by the parser, not scalar_domain.
        .all(|value| {
            matches!(value, Expr::Scalar { value }
            if value.is_finite() && value.abs() <= f64::from(limits.max_scalar_abs))
        })
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

/// Auditable provenance label for the deterministic, data-independent grammar cover.
pub const CATALOG_SCAFFOLD_OPERATION: &str = "catalog_scaffold";
/// Auditable provenance label for the seed-diverse, data-independent grammar cover.
pub const SEEDED_CATALOG_SCAFFOLD_OPERATION: &str = "seeded_catalog_scaffold";
/// Auditable provenance label for the seed-diverse quantitative-motif cover.
pub const MOTIF_CATALOG_SCAFFOLD_OPERATION: &str = "motif_catalog_scaffold";

/// Policy controlling the initial data-independent catalog cover.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogScaffoldPolicy {
    /// Preserve the historical fixed 64-expression cover.
    #[default]
    Fixed,
    /// Select a balanced, seed-keyed cover from a wider typed scaffold universe.
    SeededDiverse,
    /// Preserve broad root coverage, then reserve capacity for typed quantitative motifs.
    MotifDiverse,
}

/// Return whether a provenance operation belongs to either catalog-cover policy.
#[must_use]
pub fn is_catalog_scaffold_operation(operation: &str) -> bool {
    matches!(
        operation,
        CATALOG_SCAFFOLD_OPERATION
            | SEEDED_CATALOG_SCAFFOLD_OPERATION
            | MOTIF_CATALOG_SCAFFOLD_OPERATION
    )
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
    pub const ALL: [Self; 8] = [
        Self::Field,
        Self::Window,
        Self::Scalar,
        Self::Group,
        Self::BoundedKeyword,
        Self::BooleanLiteral,
        Self::Operator,
        Self::Subtree,
    ];

    #[must_use]
    pub const fn operation(self) -> &'static str {
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
    generate_with_catalog_scaffold_policy(
        seed,
        index,
        limits,
        catalog,
        CatalogScaffoldPolicy::Fixed,
    )
}

/// Deterministically generate with an explicit catalog-cover policy.
#[must_use]
pub fn generate_with_catalog_scaffold_policy(
    seed: u64,
    index: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
    scaffold_policy: CatalogScaffoldPolicy,
) -> (Expr, Provenance) {
    let scaffolds = match scaffold_policy {
        CatalogScaffoldPolicy::Fixed => catalog_scaffolds(limits, catalog),
        CatalogScaffoldPolicy::SeededDiverse => seeded_catalog_scaffolds(seed, limits, catalog),
        CatalogScaffoldPolicy::MotifDiverse => motif_catalog_scaffolds(seed, limits, catalog),
    };
    if let Some(expression) = scaffolds
        .get(usize::try_from(index).unwrap_or(usize::MAX))
        .cloned()
    {
        let provenance = Provenance {
            kind: TransformKind::Initial,
            operation: match scaffold_policy {
                CatalogScaffoldPolicy::Fixed => CATALOG_SCAFFOLD_OPERATION,
                CatalogScaffoldPolicy::SeededDiverse => SEEDED_CATALOG_SCAFFOLD_OPERATION,
                CatalogScaffoldPolicy::MotifDiverse => MOTIF_CATALOG_SCAFFOLD_OPERATION,
            }
            .to_owned(),
            parent_fingerprints: Vec::new(),
            requested_operation: None,
            affected_path: None,
            old_subtree_fingerprint: None,
            new_subtree_fingerprint: Some(fingerprint(&expression)),
            retry_count: 0,
        };
        return (expression, provenance);
    }
    sample_with_catalog(seed, index, limits, catalog)
}

/// Sample the typed grammar directly, without a fixed or seeded scaffold prefix.
/// Identical seed, index, limits and catalog produce identical results.
/// Sampling follows the catalog's generation weights; it is not uniform over trees.
#[must_use]
pub fn sample_with_catalog(
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

fn catalog_scaffolds(limits: &Limits, catalog: &operators::Catalog) -> Vec<Expr> {
    catalog_scaffold_universe(limits, catalog)
        .into_iter()
        .take(MAX_CATALOG_SCAFFOLDS)
        .collect()
}

#[allow(clippy::too_many_lines)]
fn catalog_scaffold_universe(limits: &Limits, catalog: &operators::Catalog) -> Vec<Expr> {
    let fields = catalog
        .fields
        .iter()
        .map(|field| Expr::Field {
            name: field.name.clone(),
        })
        .collect::<Vec<_>>();
    let mut result = Vec::with_capacity(MAX_CATALOG_SCAFFOLDS);
    let mut seen = BTreeSet::new();
    let mut push = |candidate: Expr| {
        if result.len() >= MAX_CATALOG_SCAFFOLD_UNIVERSE
            || validate_transformed_with_catalog(&candidate, limits, catalog).is_err()
        {
            return;
        }
        let identity = semantic_fingerprint(&candidate);
        if seen.insert(identity) {
            result.push(candidate);
        }
    };

    for field in &fields {
        push(field.clone());
    }
    for operator in ["multiply", "add"] {
        if catalog
            .lookup(operator)
            .is_none_or(|spec| spec.generation_weight == 0)
        {
            continue;
        }
        for left in 0..fields.len() {
            let first_right = if operator == "add" { left + 1 } else { left };
            for right in first_right..fields.len() {
                if let Ok(candidate) = operators::build_call_with_catalog(
                    catalog,
                    operator,
                    vec![fields[left].clone(), fields[right].clone()],
                    BTreeMap::new(),
                ) {
                    push(candidate);
                }
            }
        }
    }
    if catalog
        .lookup("subtract")
        .is_some_and(|spec| spec.generation_weight > 0)
    {
        for left in 0..fields.len() {
            for right in left + 1..fields.len() {
                for (first, second) in [(left, right), (right, left)] {
                    if let Ok(candidate) = operators::build_call_with_catalog(
                        catalog,
                        "subtract",
                        vec![fields[first].clone(), fields[second].clone()],
                        BTreeMap::new(),
                    ) {
                        push(candidate);
                    }
                }
            }
        }
    }
    let unary_inputs = seeded_scaffold_unary_inputs(&fields, limits, catalog);
    for spec in catalog
        .operators
        .iter()
        .filter(|spec| spec.generation_weight > 0 && spec.output == ExprKind::Signal)
    {
        match spec.arity {
            Arity::Unary
                if spec.inputs.len() == 1
                    && spec.inputs[0].kinds.contains(&ExprKind::Signal)
                    && matches!(spec.inputs[0].domain, ValueDomain::Any) =>
            {
                for input in &unary_inputs {
                    if let Ok(candidate) = operators::build_call_with_catalog(
                        catalog,
                        &spec.name,
                        vec![input.clone()],
                        BTreeMap::new(),
                    ) {
                        push(candidate);
                    }
                }
            }
            Arity::Binary
                if spec.inputs.len() == 2
                    && spec.inputs[0].kinds.contains(&ExprKind::Signal)
                    && spec.inputs[1].kinds.contains(&ExprKind::Signal)
                    && matches!(spec.inputs[0].domain, ValueDomain::Any)
                    && matches!(spec.inputs[1].domain, ValueDomain::Any) =>
            {
                for left in 0..fields.len() {
                    let first_right = if spec.commutative { left } else { 0 };
                    for right in first_right..fields.len() {
                        if let Ok(candidate) = operators::build_call_with_catalog(
                            catalog,
                            &spec.name,
                            vec![fields[left].clone(), fields[right].clone()],
                            BTreeMap::new(),
                        ) {
                            push(candidate);
                        }
                    }
                }
            }
            Arity::Binary
                if spec.inputs.len() == 2
                    && spec.inputs[0].kinds.contains(&ExprKind::Signal)
                    && matches!(spec.inputs[0].domain, ValueDomain::Any)
                    && matches!(
                        spec.inputs[1].domain,
                        ValueDomain::Window { .. } | ValueDomain::WindowSet { .. }
                    ) =>
            {
                for field in &fields {
                    for window in scaffold_windows(&spec.inputs[1].domain, limits) {
                        if let Ok(candidate) = operators::build_call_with_catalog(
                            catalog,
                            &spec.name,
                            vec![
                                field.clone(),
                                Expr::Scalar {
                                    value: f64::from(window),
                                },
                            ],
                            BTreeMap::new(),
                        ) {
                            push(candidate);
                        }
                    }
                }
            }
            Arity::Exact { count: 3 }
                if spec.inputs.len() == 3
                    && spec.inputs[0].kinds.contains(&ExprKind::Signal)
                    && spec.inputs[1].kinds.contains(&ExprKind::Signal)
                    && matches!(spec.inputs[0].domain, ValueDomain::Any)
                    && matches!(spec.inputs[1].domain, ValueDomain::Any)
                    && matches!(
                        spec.inputs[2].domain,
                        ValueDomain::Window { .. } | ValueDomain::WindowSet { .. }
                    ) =>
            {
                for left in 0..fields.len() {
                    let first_right = if spec.commutative { left } else { 0 };
                    for right in first_right..fields.len() {
                        for window in scaffold_windows(&spec.inputs[2].domain, limits) {
                            if let Ok(candidate) = operators::build_call_with_catalog(
                                catalog,
                                &spec.name,
                                vec![
                                    fields[left].clone(),
                                    fields[right].clone(),
                                    Expr::Scalar {
                                        value: f64::from(window),
                                    },
                                ],
                                BTreeMap::new(),
                            ) {
                                push(candidate);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    result
}

fn seeded_scaffold_unary_inputs(
    fields: &[Expr],
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Vec<Expr> {
    let mut result = fields.to_vec();
    let mut seen = result
        .iter()
        .map(semantic_fingerprint)
        .collect::<BTreeSet<_>>();
    for operator in ["multiply", "add", "subtract", "divide"] {
        if catalog
            .lookup(operator)
            .is_none_or(|spec| spec.generation_weight == 0)
        {
            continue;
        }
        for left in 0..fields.len() {
            for right in 0..fields.len() {
                if matches!(operator, "multiply" | "add") && right < left
                    || matches!(operator, "add" | "subtract") && right == left
                {
                    continue;
                }
                let Ok(candidate) = operators::build_call_with_catalog(
                    catalog,
                    operator,
                    vec![fields[left].clone(), fields[right].clone()],
                    BTreeMap::new(),
                ) else {
                    continue;
                };
                if !limits.accepts_with_catalog(&candidate, catalog)
                    || analyze_expression(&candidate).rejection_reason.is_some()
                {
                    continue;
                }
                if seen.insert(semantic_fingerprint(&candidate)) {
                    result.push(candidate);
                }
            }
        }
    }
    result
}

fn scaffold_windows(domain: &ValueDomain, limits: &Limits) -> Vec<u16> {
    let mut values = match domain {
        ValueDomain::WindowSet { values } => values.clone(),
        ValueDomain::Window { min, max } => {
            let lower = (*min).max(limits.min_window);
            let upper = (*max).min(limits.max_window);
            if lower > upper {
                Vec::new()
            } else {
                vec![lower, lower + (upper - lower) / 2, upper]
            }
        }
        _ => Vec::new(),
    };
    values.retain(|value| (limits.min_window..=limits.max_window).contains(value));
    values.sort_unstable();
    values.dedup();
    values.truncate(8);
    values
}

fn seeded_catalog_scaffolds(seed: u64, limits: &Limits, catalog: &operators::Catalog) -> Vec<Expr> {
    let universe = catalog_scaffold_universe(limits, catalog)
        .into_iter()
        .filter(|candidate| analyze_expression(candidate).rejection_reason.is_none())
        .collect::<Vec<_>>();
    let (fields, candidates): (Vec<_>, Vec<_>) = universe
        .into_iter()
        .partition(|candidate| matches!(candidate, Expr::Field { .. }));
    let mut groups = BTreeMap::<String, Vec<(String, String, Expr)>>::new();
    for candidate in candidates {
        let identity = semantic_fingerprint(&candidate);
        let operator = candidate.operator().unwrap_or("field").to_owned();
        let key = digest(&format!("seeded-catalog-scaffold:{seed}:{identity}"));
        groups
            .entry(operator)
            .or_default()
            .push((key, identity, candidate));
    }
    for candidates in groups.values_mut() {
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    }
    let mut queues = groups
        .into_iter()
        .map(|(operator, candidates)| {
            (
                digest(&format!("seeded-catalog-scaffold-family:{seed}:{operator}")),
                operator,
                VecDeque::from(candidates),
            )
        })
        .collect::<Vec<_>>();
    queues.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let mut result = fields;
    while result.len() < MAX_CATALOG_SCAFFOLDS {
        let mut progressed = false;
        for (_, _, queue) in &mut queues {
            if let Some((_, _, candidate)) = queue.pop_front()
                && result.len() < MAX_CATALOG_SCAFFOLDS
            {
                result.push(candidate);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    result
}

fn scaffold_call(
    catalog: &operators::Catalog,
    operator: &str,
    arguments: Vec<Expr>,
) -> Option<Expr> {
    catalog
        .lookup(operator)
        .filter(|spec| spec.generation_weight > 0)
        .and_then(|_| {
            operators::build_call_with_catalog(catalog, operator, arguments, BTreeMap::new()).ok()
        })
}

fn motif_windows(limits: &Limits) -> Vec<u16> {
    [2, 5, 10, 20, 40]
        .into_iter()
        .filter(|window| (limits.min_window..=limits.max_window).contains(window))
        .collect()
}

fn add_motif(
    groups: &mut BTreeMap<String, Vec<Expr>>,
    family: &str,
    candidate: Option<Expr>,
    limits: &Limits,
    catalog: &operators::Catalog,
) {
    let Some(candidate) = candidate else {
        return;
    };
    if validate_transformed_with_catalog(&candidate, limits, catalog).is_ok()
        && analyze_expression(&candidate).rejection_reason.is_none()
    {
        groups.entry(family.to_owned()).or_default().push(candidate);
    }
}

#[allow(clippy::too_many_lines)]
fn quantitative_motif_universe(
    limits: &Limits,
    catalog: &operators::Catalog,
) -> BTreeMap<String, Vec<Expr>> {
    let fields = catalog
        .fields
        .iter()
        .map(|field| {
            (
                field.family.as_str(),
                Expr::Field {
                    name: field.name.clone(),
                },
            )
        })
        .collect::<Vec<_>>();
    let prices = fields
        .iter()
        .filter(|(family, _)| *family == "price")
        .map(|(_, expression)| expression.clone())
        .collect::<Vec<_>>();
    let liquidity = fields
        .iter()
        .filter(|(family, _)| *family == "liquidity")
        .map(|(_, expression)| expression.clone())
        .collect::<Vec<_>>();
    let windows = motif_windows(limits);
    let short_windows = windows.iter().copied().filter(|window| *window <= 10);
    let mut groups = BTreeMap::<String, Vec<Expr>>::new();

    for liquid in &liquidity {
        for price in &prices {
            for window in &windows {
                let scalar = Expr::Scalar {
                    value: f64::from(*window),
                };
                let covariance = scaffold_call(
                    catalog,
                    "ts_cov",
                    vec![liquid.clone(), price.clone(), scalar],
                );
                add_motif(
                    &mut groups,
                    "liquidity_price_covariance",
                    covariance.clone(),
                    limits,
                    catalog,
                );
                for smooth in short_windows.clone() {
                    let smooth_scalar = Expr::Scalar {
                        value: f64::from(smooth),
                    };
                    add_motif(
                        &mut groups,
                        "smoothed_liquidity_price_covariance",
                        covariance.clone().and_then(|input| {
                            scaffold_call(catalog, "ts_ema", vec![input, smooth_scalar.clone()])
                        }),
                        limits,
                        catalog,
                    );
                    add_motif(
                        &mut groups,
                        "averaged_liquidity_price_covariance",
                        covariance.clone().and_then(|input| {
                            scaffold_call(catalog, "ts_mean", vec![input, smooth_scalar.clone()])
                        }),
                        limits,
                        catalog,
                    );
                }
            }
        }
    }

    for left in 0..prices.len() {
        for right in left + 1..prices.len() {
            for window in &windows {
                let correlation = scaffold_call(
                    catalog,
                    "ts_corr",
                    vec![
                        prices[left].clone(),
                        prices[right].clone(),
                        Expr::Scalar {
                            value: f64::from(*window),
                        },
                    ],
                );
                add_motif(
                    &mut groups,
                    "price_correlation",
                    correlation.clone(),
                    limits,
                    catalog,
                );
                for outer in short_windows.clone() {
                    let scalar = Expr::Scalar {
                        value: f64::from(outer),
                    };
                    add_motif(
                        &mut groups,
                        "smoothed_price_correlation",
                        correlation.clone().and_then(|input| {
                            scaffold_call(catalog, "ts_mean", vec![input, scalar.clone()])
                        }),
                        limits,
                        catalog,
                    );
                    add_motif(
                        &mut groups,
                        "bounded_price_correlation",
                        correlation.clone().and_then(|input| {
                            scaffold_call(catalog, "ts_min", vec![input, scalar.clone()])
                        }),
                        limits,
                        catalog,
                    );
                }
            }
        }
    }

    for price in &prices {
        for window in &windows {
            let dispersion = scaffold_call(
                catalog,
                "ts_std_dev",
                vec![
                    price.clone(),
                    Expr::Scalar {
                        value: f64::from(*window),
                    },
                ],
            );
            add_motif(
                &mut groups,
                "absolute_price_dispersion",
                dispersion
                    .clone()
                    .and_then(|input| scaffold_call(catalog, "abs", vec![input])),
                limits,
                catalog,
            );
            for smooth in short_windows.clone() {
                add_motif(
                    &mut groups,
                    "smoothed_price_dispersion",
                    dispersion.clone().and_then(|input| {
                        scaffold_call(
                            catalog,
                            "ts_ema",
                            vec![
                                input,
                                Expr::Scalar {
                                    value: f64::from(smooth),
                                },
                            ],
                        )
                    }),
                    limits,
                    catalog,
                );
            }
        }
    }
    groups
}

fn motif_catalog_scaffolds(seed: u64, limits: &Limits, catalog: &operators::Catalog) -> Vec<Expr> {
    let universe = catalog_scaffold_universe(limits, catalog)
        .into_iter()
        .filter(|candidate| analyze_expression(candidate).rejection_reason.is_none())
        .collect::<Vec<_>>();
    let (fields, candidates): (Vec<_>, Vec<_>) = universe
        .into_iter()
        .partition(|candidate| matches!(candidate, Expr::Field { .. }));
    let mut identities = fields
        .iter()
        .map(semantic_fingerprint)
        .collect::<BTreeSet<_>>();
    let mut result = fields;

    let mut root_groups = BTreeMap::<String, Vec<(String, String, Expr)>>::new();
    for candidate in candidates {
        let identity = semantic_fingerprint(&candidate);
        let operator = candidate.operator().unwrap_or("field").to_owned();
        let key = digest(&format!("motif-root-cover:{seed}:{identity}"));
        root_groups
            .entry(operator)
            .or_default()
            .push((key, identity, candidate));
    }
    for candidates in root_groups.values_mut() {
        candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    }
    let mut root_queues = root_groups.into_iter().collect::<Vec<_>>();
    root_queues.sort_by(|left, right| {
        digest(&format!("motif-root-family:{seed}:{}", left.0))
            .cmp(&digest(&format!("motif-root-family:{seed}:{}", right.0)))
            .then_with(|| left.0.cmp(&right.0))
    });
    for (_, candidates) in &root_queues {
        if let Some((_, identity, candidate)) = candidates.first()
            && result.len() < MAX_CATALOG_SCAFFOLDS
            && identities.insert(identity.clone())
        {
            result.push(candidate.clone());
        }
    }

    let mut motif_queues = quantitative_motif_universe(limits, catalog)
        .into_iter()
        .map(|(family, candidates)| {
            let mut candidates = candidates
                .into_iter()
                .map(|candidate| {
                    let identity = semantic_fingerprint(&candidate);
                    let key = digest(&format!("motif-candidate:{seed}:{family}:{identity}"));
                    (key, identity, candidate)
                })
                .collect::<Vec<_>>();
            candidates
                .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
            (
                digest(&format!("motif-family:{seed}:{family}")),
                family,
                VecDeque::from(candidates),
            )
        })
        .collect::<Vec<_>>();
    motif_queues.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    while result.len() < MAX_CATALOG_SCAFFOLDS {
        let mut progressed = false;
        for (_, _, queue) in &mut motif_queues {
            while let Some((_, identity, candidate)) = queue.pop_front() {
                if identities.insert(identity) {
                    result.push(candidate);
                    progressed = true;
                    break;
                }
            }
            if result.len() >= MAX_CATALOG_SCAFFOLDS {
                break;
            }
        }
        if !progressed {
            break;
        }
    }

    if result.len() < MAX_CATALOG_SCAFFOLDS {
        let mut remainder = root_queues
            .into_iter()
            .flat_map(|(_, candidates)| candidates)
            .collect::<Vec<_>>();
        remainder.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        for (_, identity, candidate) in remainder {
            if result.len() >= MAX_CATALOG_SCAFFOLDS {
                break;
            }
            if identities.insert(identity) {
                result.push(candidate);
            }
        }
    }
    result
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

/// Choose uniformly among mutation classes with a successful bounded local trial.
/// Unlike the legacy policy, classes without matching paths are not sampled.
/// Each class gets at most the usual retry budget; at most eight candidates are
/// retained. If none succeeds, emit an explicitly recorded fresh fallback.
/// This opt-in policy does not alter the legacy mutation RNG stream.
#[must_use]
pub fn mutate_applicable_with_catalog(
    parent: &Expr,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let paths = enumerate_paths(parent);
    let mut candidates = Vec::new();
    for class in MutationClass::ALL {
        if !paths
            .iter()
            .any(|entry| mutation_path_matches(parent, entry, class, catalog))
        {
            continue;
        }
        let mut rng = seeded(
            seed,
            ordinal,
            0xa076_1d64_78bd_642f ^ mutation_domain(class),
        );
        if let Some(candidate) = try_mutate_with_rng(parent, class, &mut rng, limits, catalog) {
            candidates.push(candidate);
        }
    }
    let mut rng = seeded(seed, ordinal, 0x3eed_819f_a214_2a79);
    if candidates.is_empty() {
        fallback_generation(
            parent,
            "applicable_mutation",
            &mut rng,
            limits,
            RETRY_BUDGET,
            catalog,
        )
    } else {
        let index = rng.random_range(0..candidates.len());
        candidates.swap_remove(index)
    }
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
    mutate_with_class_and_catalog(
        parent,
        class,
        seed,
        ordinal,
        limits,
        operators::builtin_catalog(),
    )
}

/// Deterministically request one mutation class under an explicit catalog.
#[must_use]
pub fn mutate_with_class_and_catalog(
    parent: &Expr,
    class: MutationClass,
    seed: u64,
    ordinal: u64,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    let mut rng = seeded(
        seed,
        ordinal,
        0xa076_1d64_78bd_642f ^ mutation_domain(class),
    );
    mutate_with_rng(parent, class, &mut rng, limits, catalog)
}

fn mutate_with_rng(
    parent: &Expr,
    class: MutationClass,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> (Expr, Provenance) {
    if let Some(candidate) = try_mutate_with_rng(parent, class, rng, limits, catalog) {
        return candidate;
    }
    fallback_generation(
        parent,
        class.operation(),
        rng,
        limits,
        RETRY_BUDGET,
        catalog,
    )
}

fn try_mutate_with_rng(
    parent: &Expr,
    class: MutationClass,
    rng: &mut ChaCha8Rng,
    limits: &Limits,
    catalog: &operators::Catalog,
) -> Option<(Expr, Provenance)> {
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
        return Some((
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
        ));
    }
    None
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
    fn generation_domain_is_enforced_after_transforms_without_restricting_windows() {
        let mut catalog = operators::builtin_catalog().clone();
        catalog.scalar_domain = operators::ScalarDomain {
            min: -2.0,
            max: 2.0,
            step: 0.5,
        };
        let limits = Limits::default();
        for source in [
            "ts_var(divide(volume,40),40)",
            "multiply(20,low)",
            "multiply(close,0.25)",
            "multiply(close,-2.5)",
        ] {
            // Parsing user input remains broader than research generation.
            let expr = parse_expression_with_catalog(source, &catalog).unwrap();
            assert!(
                validate_transformed_with_catalog(&expr, &limits, &catalog).is_err(),
                "{source}"
            );
        }
        for source in [
            "ts_var(divide(volume,2),40)",
            "multiply(close,-1.5)",
            "winsorize(ts_mean(close,40),std=3)",
        ] {
            let expr = parse_expression_with_catalog(source, &catalog).unwrap();
            validate_transformed_with_catalog(&expr, &limits, &catalog).unwrap();
        }
    }

    #[test]
    fn narrow_domain_generation_mutation_and_crossover_remain_valid() {
        let mut catalog = operators::builtin_catalog().clone();
        catalog.scalar_domain = operators::ScalarDomain {
            min: -2.0,
            max: 2.0,
            step: 0.5,
        };
        let limits = Limits::default();
        for index in 0..512 {
            let (left, _) = generate_with_catalog(919, index, &limits, &catalog);
            let (right, _) = generate_with_catalog(929, index, &limits, &catalog);
            validate_transformed_with_catalog(&left, &limits, &catalog).unwrap();
            validate_transformed_with_catalog(&right, &limits, &catalog).unwrap();
            let (child, _) = crossover_with_catalog(&left, &right, 919, index, &limits, &catalog);
            validate_transformed_with_catalog(&child, &limits, &catalog).unwrap();
            let (mutant, _) = mutate_with_catalog(&child, 929, index, &limits, &catalog);
            validate_transformed_with_catalog(&mutant, &limits, &catalog).unwrap();
        }
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
    fn initial_generation_starts_with_a_deterministic_catalog_scaffold() {
        let catalog = operators::builtin_catalog();
        let limits = Limits::default();
        let generated = (0..MAX_CATALOG_SCAFFOLDS as u64)
            .map(|index| generate_with_catalog(7, index, &limits, catalog))
            .collect::<Vec<_>>();
        assert!(generated.iter().all(|(_, provenance)| {
            provenance.operation == CATALOG_SCAFFOLD_OPERATION
                && provenance.kind == TransformKind::Initial
        }));
        let expressions = generated
            .iter()
            .map(|(expression, _)| canonical(expression))
            .collect::<BTreeSet<_>>();
        assert_eq!(expressions.len(), MAX_CATALOG_SCAFFOLDS);
        for field in &catalog.fields {
            assert!(expressions.contains(&field.name));
        }
        for expression in [
            "multiply(close, close)",
            "multiply(open, open)",
            "multiply(high, low)",
        ] {
            let expected = canonical(&parse_expression(expression).unwrap());
            assert!(expressions.contains(&expected), "missing {expected}");
        }
        let (_, after_scaffold) =
            generate_with_catalog(7, MAX_CATALOG_SCAFFOLDS as u64, &limits, catalog);
        assert_eq!(after_scaffold.operation, "grammar_sample");
    }

    #[test]
    fn seeded_catalog_scaffold_is_balanced_deterministic_and_seed_diverse() {
        let catalog = operators::builtin_catalog();
        let limits = Limits::default();
        let generate_cover = |seed| {
            (0..MAX_CATALOG_SCAFFOLDS as u64)
                .map(|index| {
                    generate_with_catalog_scaffold_policy(
                        seed,
                        index,
                        &limits,
                        catalog,
                        CatalogScaffoldPolicy::SeededDiverse,
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = generate_cover(7);
        let repeated = generate_cover(7);
        let other_seed = generate_cover(11);
        assert_eq!(first, repeated);
        assert!(first.iter().all(|(_, provenance)| {
            provenance.operation == SEEDED_CATALOG_SCAFFOLD_OPERATION
                && provenance.kind == TransformKind::Initial
        }));
        let expressions = first
            .iter()
            .map(|(expression, _)| canonical(expression))
            .collect::<BTreeSet<_>>();
        let other_expressions = other_seed
            .iter()
            .map(|(expression, _)| canonical(expression))
            .collect::<BTreeSet<_>>();
        assert_eq!(expressions.len(), MAX_CATALOG_SCAFFOLDS);
        assert_ne!(expressions, other_expressions);
        for field in &catalog.fields {
            assert!(expressions.contains(&field.name));
        }
        let root_operators = first
            .iter()
            .filter_map(|(expression, _)| expression.operator())
            .collect::<BTreeSet<_>>();
        assert!(root_operators.len() >= 12);

        for seed in 0..256 {
            let scaffold = seeded_catalog_scaffolds(seed, &limits, catalog);
            assert_eq!(
                scaffold.len(),
                MAX_CATALOG_SCAFFOLDS,
                "seed {seed} did not fill the scaffold budget"
            );
            assert!(
                scaffold
                    .iter()
                    .all(|candidate| analyze_expression(candidate).rejection_reason.is_none()),
                "seed {seed} selected a trivial scaffold candidate"
            );
        }
    }

    #[test]
    fn motif_catalog_scaffold_preserves_breadth_and_adds_nested_quantitative_families() {
        let catalog = operators::builtin_catalog();
        let limits = Limits::default();
        let generate_cover = |seed| {
            (0..MAX_CATALOG_SCAFFOLDS as u64)
                .map(|index| {
                    generate_with_catalog_scaffold_policy(
                        seed,
                        index,
                        &limits,
                        catalog,
                        CatalogScaffoldPolicy::MotifDiverse,
                    )
                })
                .collect::<Vec<_>>()
        };
        let first = generate_cover(7);
        let repeated = generate_cover(7);
        let other_seed = generate_cover(11);
        assert_eq!(first, repeated);
        assert!(first.iter().all(|(_, provenance)| {
            provenance.operation == MOTIF_CATALOG_SCAFFOLD_OPERATION
                && provenance.kind == TransformKind::Initial
        }));
        let expressions = first
            .iter()
            .map(|(expression, _)| canonical(expression))
            .collect::<BTreeSet<_>>();
        let other_expressions = other_seed
            .iter()
            .map(|(expression, _)| canonical(expression))
            .collect::<BTreeSet<_>>();
        assert_eq!(expressions.len(), MAX_CATALOG_SCAFFOLDS);
        assert_ne!(expressions, other_expressions);
        for field in &catalog.fields {
            assert!(expressions.contains(&field.name));
        }
        let root_operators = first
            .iter()
            .filter_map(|(expression, _)| expression.operator())
            .collect::<BTreeSet<_>>();
        assert!(root_operators.len() >= 20);
        assert!(
            expressions
                .iter()
                .any(|expression| expression.starts_with("ts_cov("))
        );
        assert!(
            expressions
                .iter()
                .any(|expression| expression.starts_with("ts_corr("))
        );
        assert!(expressions.iter().any(|expression| {
            (expression.starts_with("ts_ema(") || expression.starts_with("ts_mean("))
                && expression.contains("ts_cov(")
        }));
        assert!(expressions.iter().any(|expression| {
            (expression.starts_with("ts_mean(") || expression.starts_with("ts_min("))
                && expression.contains("ts_corr(")
        }));
        assert!(
            expressions
                .iter()
                .any(|expression| expression.contains("ts_std_dev("))
        );

        for seed in 0..64 {
            let scaffold = motif_catalog_scaffolds(seed, &limits, catalog);
            assert_eq!(scaffold.len(), MAX_CATALOG_SCAFFOLDS);
            assert!(scaffold.iter().all(|candidate| {
                analyze_expression(candidate).rejection_reason.is_none()
                    && limits.accepts_with_catalog(candidate, catalog)
            }));
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
        let (expression, _) = generate_with_catalog(7, 111, &Limits::default(), &catalog);
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
