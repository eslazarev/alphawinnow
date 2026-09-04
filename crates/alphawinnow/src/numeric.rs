//! Optional, offline measured evidence over immutable columnar input.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use crate::parse_expression;
use crate::{
    Catalog, Expr, builtin_catalog, canonical::digest, fingerprint, parse_expression_with_catalog,
};

pub const NUMERIC_EVIDENCE_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericColumn {
    /// Row-major values: `row * asset_count + asset`.
    pub values: Vec<Option<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnarDataset {
    pub schema: u32,
    pub dataset_id: String,
    pub timestamps: Vec<i64>,
    pub assets: Vec<String>,
    pub fields: BTreeMap<String, NumericColumn>,
    /// Return realized during each row. A horizon-one target for signal row
    /// `t` is therefore read from row `t + 1`, never from `t`.
    pub realized_returns: NumericColumn,
    /// Static asset membership for group operators.
    #[serde(default)]
    pub groups: BTreeMap<String, Vec<String>>,
    pub source_label: String,
    pub preprocessing: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub schema: u32,
    pub horizon_rows: usize,
    /// First row whose signal may enter reported evidence. Earlier rows stay
    /// available to rolling operators as causally prior warm-up history.
    #[serde(default)]
    pub evaluation_start: usize,
    pub train_end: usize,
    pub validation_end: usize,
    pub transaction_cost_bps: f64,
    pub annualization_rows: Option<u32>,
    /// Optional platform-style root signal linear decay; zero disables it.
    #[serde(default)]
    pub signal_decay: usize,
    /// Maximum absolute post-normalization position weight. Zero disables the
    /// cap; enabled values follow the platform convention `(0, 1]`.
    #[serde(default)]
    pub truncation: f64,
    /// Emit one continuous, auditable daily `PnL` row for the overall period.
    #[serde(default)]
    pub include_daily: bool,
    /// Derive explicitly labelled local estimates using documented BRAIN-like
    /// metric conventions. These are never remote platform results.
    #[serde(default)]
    pub brain_proxy: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainProxyMetrics {
    pub methodology: String,
    pub estimated_sharpe: Option<f64>,
    pub estimated_returns: Option<f64>,
    pub estimated_turnover: Option<f64>,
    pub estimated_drawdown: Option<f64>,
    pub estimated_margin: Option<f64>,
    pub estimated_fitness: Option<f64>,
    pub remote_platform_result: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailyEvidence {
    pub signal_row: usize,
    pub target_end_row: usize,
    pub signal_timestamp: i64,
    pub target_timestamp: i64,
    pub measured_gross_return: f64,
    pub measured_net_return: f64,
    pub measured_turnover: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitEvidence {
    pub split: String,
    pub start_row: usize,
    pub end_row_exclusive: usize,
    pub observations: usize,
    pub measured_pearson_correlation: Option<f64>,
    pub measured_mean_row_gross_return: Option<f64>,
    pub measured_mean_row_net_return: Option<f64>,
    pub measured_mean_row_turnover: Option<f64>,
    pub measured_annual_gross_return: Option<f64>,
    pub measured_annual_net_return: Option<f64>,
    pub measured_gross_sharpe: Option<f64>,
    pub measured_net_sharpe: Option<f64>,
    pub measured_additive_gross_drawdown: Option<f64>,
    pub measured_additive_net_drawdown: Option<f64>,
    pub brain_proxy: Option<BrainProxyMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationArtifact {
    pub schema: u32,
    pub dataset_id: String,
    pub dataset_checksum: String,
    pub source_label: String,
    pub expression: String,
    pub expression_fingerprint: String,
    pub horizon_rows: usize,
    pub evaluation_start: usize,
    pub signal_decay: usize,
    pub truncation: f64,
    pub transaction_cost_bps: f64,
    pub annualization_rows: Option<u32>,
    pub alignment: String,
    pub preprocessing: Vec<String>,
    pub evaluator_assumptions: Vec<String>,
    /// Continuous evidence over `evaluation_start..dataset_end`. Unlike the
    /// diagnostic splits, this does not reset turnover at split boundaries.
    pub overall: SplitEvidence,
    pub splits: Vec<SplitEvidence>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub daily: Vec<DailyEvidence>,
    pub structural_score_used: bool,
}

#[derive(Debug, Error)]
pub enum NumericError {
    #[error("invalid dataset: {0}")]
    Dataset(String),
    #[error("invalid evaluation configuration: {0}")]
    Config(String),
    #[error("cannot evaluate expression: {0}")]
    Evaluation(String),
    #[error("cannot serialize numeric input: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error(transparent)]
    Expression(#[from] crate::ExpressionError),
}

/// Validated, reusable evaluator for many expressions over one immutable panel.
///
/// Dataset validation and checksum construction are intentionally performed
/// once. This matters for large columnar panels where repeating those two
/// read-only passes can cost more than evaluating a simple expression.
pub struct NumericEvaluator<'a> {
    dataset: &'a ColumnarDataset,
    catalog: &'a Catalog,
    config: EvaluationConfig,
    dataset_checksum: String,
}

impl ColumnarDataset {
    /// Validate shape, ordering, finiteness, and group membership.
    ///
    /// # Errors
    /// Returns an invariant diagnostic for malformed immutable input.
    pub fn validate(&self) -> Result<(), NumericError> {
        if self.schema != 1 {
            return Err(NumericError::Dataset(format!(
                "unsupported dataset schema {}",
                self.schema
            )));
        }
        if self.dataset_id.is_empty()
            || self.source_label.is_empty()
            || self.timestamps.len() < 3
            || self.assets.len() < 2
        {
            return Err(NumericError::Dataset(
                "dataset id/source, at least three rows, and two assets are required".to_owned(),
            ));
        }
        if self.timestamps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(NumericError::Dataset(
                "timestamps must be strictly increasing".to_owned(),
            ));
        }
        let expected = self.timestamps.len().saturating_mul(self.assets.len());
        validate_column("realized_returns", &self.realized_returns, expected)?;
        if self.fields.is_empty() {
            return Err(NumericError::Dataset(
                "at least one numeric field is required".to_owned(),
            ));
        }
        for (name, column) in &self.fields {
            validate_column(name, column, expected)?;
        }
        for (name, labels) in &self.groups {
            if labels.len() != self.assets.len() || labels.iter().any(String::is_empty) {
                return Err(NumericError::Dataset(format!(
                    "group `{name}` must contain one non-empty label per asset"
                )));
            }
        }
        Ok(())
    }

    fn checksum(&self) -> Result<String, NumericError> {
        Ok(digest(&serde_json::to_string(self)?))
    }
}

impl EvaluationConfig {
    fn validate(&self, rows: usize) -> Result<(), NumericError> {
        if self.schema != 1 {
            return Err(NumericError::Config(format!(
                "unsupported config schema {}",
                self.schema
            )));
        }
        if self.horizon_rows == 0
            || self.evaluation_start >= self.train_end
            || self.train_end >= self.validation_end
            || self.validation_end >= rows
            || self.signal_decay > 512
            || self.truncation.is_sign_negative()
            || self.truncation > 1.0
            || !self.truncation.is_finite()
            || self.transaction_cost_bps.is_sign_negative()
            || !self.transaction_cost_bps.is_finite()
        {
            return Err(NumericError::Config(
                "require positive horizon, ordered non-empty splits, decay <= 512, truncation in [0, 1], and finite non-negative costs"
                    .to_owned(),
            ));
        }
        for (start, end) in [
            (self.evaluation_start, self.train_end),
            (self.train_end, self.validation_end),
            (self.validation_end, rows),
        ] {
            if end.saturating_sub(start) <= self.horizon_rows {
                return Err(NumericError::Config(
                    "every split must contain more rows than the target horizon".to_owned(),
                ));
            }
        }
        if self.brain_proxy && self.annualization_rows.is_none() {
            return Err(NumericError::Config(
                "brain proxy metrics require --annualization-rows".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Evaluate one expression without changing structural search scores.
///
/// Signal row `t` is paired only with returns realized in rows
/// `t + 1..=t + horizon_rows`. Targets never cross split boundaries.
///
/// # Errors
/// Returns dataset, configuration, expression, or numeric-evaluation errors.
pub fn evaluate_numeric(
    expression: &str,
    dataset: &ColumnarDataset,
    config: &EvaluationConfig,
) -> Result<EvaluationArtifact, NumericError> {
    NumericEvaluator::new(dataset, config.clone())?.evaluate(expression)
}

/// Evaluate one expression against an explicit portable field/operator catalog.
///
/// # Errors
/// Returns dataset, catalog-backed expression, configuration, or numeric errors.
pub fn evaluate_numeric_with_catalog(
    expression: &str,
    dataset: &ColumnarDataset,
    config: &EvaluationConfig,
    catalog: &Catalog,
) -> Result<EvaluationArtifact, NumericError> {
    NumericEvaluator::new_with_catalog(dataset, config.clone(), catalog)?.evaluate(expression)
}

impl<'a> NumericEvaluator<'a> {
    /// Validate one dataset/configuration pair for repeated local evaluation.
    ///
    /// # Errors
    /// Returns the same dataset, configuration, or serialization diagnostics
    /// as [`evaluate_numeric`].
    pub fn new(
        dataset: &'a ColumnarDataset,
        config: EvaluationConfig,
    ) -> Result<Self, NumericError> {
        Self::new_with_catalog(dataset, config, builtin_catalog())
    }

    /// Validate one dataset/configuration pair for repeated evaluation against
    /// an explicit portable catalog.
    ///
    /// # Errors
    /// Returns dataset, configuration, or serialization diagnostics.
    pub fn new_with_catalog(
        dataset: &'a ColumnarDataset,
        config: EvaluationConfig,
        catalog: &'a Catalog,
    ) -> Result<Self, NumericError> {
        dataset.validate()?;
        config.validate(dataset.timestamps.len())?;
        let dataset_checksum = dataset.checksum()?;
        Ok(Self {
            dataset,
            catalog,
            config,
            dataset_checksum,
        })
    }

    /// Evaluate one expression without revalidating or rehashing the panel.
    ///
    /// # Errors
    /// Returns expression parsing or numeric-evaluation diagnostics.
    pub fn evaluate(&self, expression: &str) -> Result<EvaluationArtifact, NumericError> {
        evaluate_numeric_prevalidated(
            expression,
            self.dataset,
            &self.config,
            &self.dataset_checksum,
            self.catalog,
        )
    }
}

fn evaluate_numeric_prevalidated(
    expression: &str,
    dataset: &ColumnarDataset,
    config: &EvaluationConfig,
    dataset_checksum: &str,
    catalog: &Catalog,
) -> Result<EvaluationArtifact, NumericError> {
    let expression = parse_expression_with_catalog(expression, catalog)?;
    let raw_signal = evaluate_signal(&expression, dataset)?;
    let signal = if config.signal_decay == 0 {
        raw_signal
    } else {
        linear_decay(&raw_signal, dataset, config.signal_decay)
    };
    let rows = dataset.timestamps.len();
    let (overall, daily) = evaluate_period(
        "overall",
        config.evaluation_start,
        rows,
        &signal,
        dataset,
        config,
        config.include_daily,
    );
    let splits = [
        ("train", config.evaluation_start, config.train_end),
        ("validation", config.train_end, config.validation_end),
        ("holdout", config.validation_end, rows),
    ]
    .into_iter()
    .map(|(name, start, end)| evaluate_period(name, start, end, &signal, dataset, config, false).0)
    .collect();
    Ok(EvaluationArtifact {
        schema: NUMERIC_EVIDENCE_SCHEMA,
        dataset_id: dataset.dataset_id.clone(),
        dataset_checksum: dataset_checksum.to_owned(),
        source_label: dataset.source_label.clone(),
        expression: crate::canonical(&expression),
        expression_fingerprint: fingerprint(&expression),
        horizon_rows: config.horizon_rows,
        evaluation_start: config.evaluation_start,
        signal_decay: config.signal_decay,
        truncation: config.truncation,
        transaction_cost_bps: config.transaction_cost_bps,
        annualization_rows: config.annualization_rows,
        alignment: "signal[t] -> sum(realized_returns[t+1..=t+horizon]); targets stay inside split"
            .to_owned(),
        preprocessing: dataset.preprocessing.clone(),
        evaluator_assumptions: vec![
            "time-series windows use current and prior rows only".to_owned(),
            "optional root signal decay is causal linear weighting of current and prior rows"
                .to_owned(),
            "horizon returns are additive sums of future realized-return rows".to_owned(),
            "missing signal/target pairs are excluded without imputation".to_owned(),
            "cross-sectional ranks break ties deterministically by asset order".to_owned(),
            "positions are cross-sectionally demeaned, normalized to unit gross, and optionally truncated with deterministic redistribution".to_owned(),
            "cost equals absolute position turnover times transaction_cost_bps".to_owned(),
            "overall evidence is continuous and does not reset positions at diagnostic split boundaries"
                .to_owned(),
            "brain proxy metrics are local formula-derived estimates, never remote platform results"
                .to_owned(),
        ],
        overall,
        splits,
        daily,
        structural_score_used: false,
    })
}

fn validate_column(
    name: &str,
    column: &NumericColumn,
    expected: usize,
) -> Result<(), NumericError> {
    if column.values.len() != expected {
        return Err(NumericError::Dataset(format!(
            "column `{name}` has {} values, expected {expected}",
            column.values.len()
        )));
    }
    if column
        .values
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(NumericError::Dataset(format!(
            "column `{name}` contains a non-finite value"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum Evaluated {
    Signal(NumericColumn),
    Scalar(f64),
    Boolean(Vec<Option<bool>>),
    Group(String),
}

fn evaluate_signal(
    expression: &Expr,
    dataset: &ColumnarDataset,
) -> Result<NumericColumn, NumericError> {
    match evaluate(expression, dataset)? {
        Evaluated::Signal(column) => Ok(column),
        _ => Err(NumericError::Evaluation(
            "root did not produce a numeric signal".to_owned(),
        )),
    }
}

#[allow(clippy::too_many_lines)]
fn evaluate(expression: &Expr, dataset: &ColumnarDataset) -> Result<Evaluated, NumericError> {
    match expression {
        Expr::Field { name } => dataset
            .fields
            .get(name)
            .cloned()
            .map(Evaluated::Signal)
            .ok_or_else(|| NumericError::Evaluation(format!("dataset lacks field `{name}`"))),
        Expr::Scalar { value } => Ok(Evaluated::Scalar(*value)),
        Expr::Bool { value } => Ok(Evaluated::Boolean(vec![
            Some(*value);
            dataset.timestamps.len()
                * dataset.assets.len()
        ])),
        Expr::Group { name } => Ok(Evaluated::Group(name.clone())),
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => {
            let input = signal(evaluate(arg, dataset)?, op)?;
            let result = match op.as_str() {
                "negate" => map_column(&input, |value| -value),
                "rank" => cross_sectional(&input, dataset, rank_values),
                "zscore" => cross_sectional(&input, dataset, zscore_values),
                "winsorize" => {
                    let width = scalar_kwarg(kwargs, "std")?;
                    cross_sectional(&input, dataset, |values| winsorize_values(values, width))
                }
                "clip" => {
                    let lower = scalar_kwarg(kwargs, "lower")?;
                    let upper = scalar_kwarg(kwargs, "upper")?;
                    map_column(&input, |value| value.clamp(lower, upper))
                }
                _ => return Err(unsupported(op)),
            };
            Ok(Evaluated::Signal(result))
        }
        Expr::BinaryCall {
            op, left, right, ..
        } => match op.as_str() {
            "ts_rank" | "ts_mean" | "ts_std_dev" | "ts_zscore" | "ts_delta" => {
                let input = signal(evaluate(left, dataset)?, op)?;
                let evaluated = evaluate(right, dataset)?;
                let window = scalar(&evaluated, op)?;
                let window = scalar_to_window(window)?;
                Ok(Evaluated::Signal(time_series(op, &input, dataset, window)))
            }
            "subtract" => binary_signal(left, right, dataset, |a, b| a - b),
            "divide" => {
                let input = signal(evaluate(left, dataset)?, op)?;
                let evaluated = evaluate(right, dataset)?;
                let denominator = scalar(&evaluated, op)?;
                Ok(Evaluated::Signal(map_column(&input, |value| {
                    value / denominator
                })))
            }
            "greater" => {
                let input = signal(evaluate(left, dataset)?, op)?;
                let evaluated = evaluate(right, dataset)?;
                let threshold = scalar(&evaluated, op)?;
                Ok(Evaluated::Boolean(
                    input
                        .values
                        .iter()
                        .map(|value| value.map(|value| value > threshold))
                        .collect(),
                ))
            }
            "group_rank" => {
                let input = signal(evaluate(left, dataset)?, op)?;
                let group = group(evaluate(right, dataset)?, op)?;
                Ok(Evaluated::Signal(group_rank(&input, dataset, &group)?))
            }
            _ => Err(unsupported(op)),
        },
        Expr::VariadicCall { op, args, .. } => match op.as_str() {
            "add" | "multiply" => {
                let mut columns = Vec::new();
                let mut scalars = Vec::new();
                for argument in args {
                    match evaluate(argument, dataset)? {
                        Evaluated::Signal(column) => columns.push(column),
                        Evaluated::Scalar(value) => scalars.push(value),
                        _ => return Err(unsupported(op)),
                    }
                }
                let length = dataset.timestamps.len() * dataset.assets.len();
                let mut values = Vec::with_capacity(length);
                for index in 0..length {
                    let mut value = if op == "add" { 0.0 } else { 1.0 };
                    let mut valid = true;
                    for column in &columns {
                        let Some(item) = column.values[index] else {
                            valid = false;
                            break;
                        };
                        if op == "add" {
                            value += item;
                        } else {
                            value *= item;
                        }
                    }
                    for scalar in &scalars {
                        if op == "add" {
                            value += scalar;
                        } else {
                            value *= scalar;
                        }
                    }
                    values.push(valid.then_some(value));
                }
                Ok(Evaluated::Signal(NumericColumn { values }))
            }
            "if_else" => {
                let condition = boolean(evaluate(&args[0], dataset)?, op)?;
                let when_true = signal(evaluate(&args[1], dataset)?, op)?;
                let when_false = signal(evaluate(&args[2], dataset)?, op)?;
                let values = condition
                    .iter()
                    .zip(when_true.values.iter().zip(&when_false.values))
                    .map(|(condition, (when_true, when_false))| match condition {
                        Some(true) => *when_true,
                        Some(false) => *when_false,
                        None => None,
                    })
                    .collect();
                Ok(Evaluated::Signal(NumericColumn { values }))
            }
            _ => Err(unsupported(op)),
        },
    }
}

fn binary_signal(
    left: &Expr,
    right: &Expr,
    dataset: &ColumnarDataset,
    operation: impl Fn(f64, f64) -> f64,
) -> Result<Evaluated, NumericError> {
    let left = signal(evaluate(left, dataset)?, "binary")?;
    let right = signal(evaluate(right, dataset)?, "binary")?;
    let values = left
        .values
        .iter()
        .zip(&right.values)
        .map(|(left, right)| match (left, right) {
            (Some(left), Some(right)) => Some(operation(*left, *right)),
            _ => None,
        })
        .collect();
    Ok(Evaluated::Signal(NumericColumn { values }))
}

fn signal(value: Evaluated, operator: &str) -> Result<NumericColumn, NumericError> {
    match value {
        Evaluated::Signal(value) => Ok(value),
        _ => Err(NumericError::Evaluation(format!(
            "operator `{operator}` expected a signal"
        ))),
    }
}

fn scalar(value: &Evaluated, operator: &str) -> Result<f64, NumericError> {
    match value {
        Evaluated::Scalar(value) => Ok(*value),
        _ => Err(NumericError::Evaluation(format!(
            "operator `{operator}` expected a scalar"
        ))),
    }
}

fn boolean(value: Evaluated, operator: &str) -> Result<Vec<Option<bool>>, NumericError> {
    match value {
        Evaluated::Boolean(value) => Ok(value),
        _ => Err(NumericError::Evaluation(format!(
            "operator `{operator}` expected a boolean"
        ))),
    }
}

fn group(value: Evaluated, operator: &str) -> Result<String, NumericError> {
    match value {
        Evaluated::Group(value) => Ok(value),
        _ => Err(NumericError::Evaluation(format!(
            "operator `{operator}` expected a group"
        ))),
    }
}

fn unsupported(operator: &str) -> NumericError {
    NumericError::Evaluation(format!("operator `{operator}` has no numeric backend"))
}

fn scalar_kwarg(kwargs: &BTreeMap<String, Expr>, name: &str) -> Result<f64, NumericError> {
    match kwargs.get(name) {
        Some(Expr::Scalar { value }) => Ok(*value),
        _ => Err(NumericError::Evaluation(format!(
            "keyword `{name}` must be a scalar"
        ))),
    }
}

fn scalar_to_window(value: f64) -> Result<usize, NumericError> {
    if value.fract() != 0.0 || !(1.0..=f64::from(u32::MAX)).contains(&value) {
        return Err(NumericError::Evaluation(
            "invalid numeric window".to_owned(),
        ));
    }
    value
        .to_string()
        .parse::<usize>()
        .map_err(|_| NumericError::Evaluation("numeric window is too large".to_owned()))
}

fn map_column(column: &NumericColumn, operation: impl Fn(f64) -> f64) -> NumericColumn {
    NumericColumn {
        values: column
            .values
            .iter()
            .map(|value| value.map(&operation))
            .collect(),
    }
}

fn cross_sectional(
    column: &NumericColumn,
    dataset: &ColumnarDataset,
    operation: impl Fn(&[Option<f64>]) -> Vec<Option<f64>>,
) -> NumericColumn {
    let assets = dataset.assets.len();
    let mut values = Vec::with_capacity(column.values.len());
    for row in 0..dataset.timestamps.len() {
        values.extend(operation(&column.values[row * assets..(row + 1) * assets]));
    }
    NumericColumn { values }
}

fn rank_values(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut indexed: Vec<_> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.map(|value| (index, value)))
        .collect();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let denominator = f64::from(u32::try_from(indexed.len().saturating_sub(1)).unwrap_or(u32::MAX));
    let mut result = vec![None; values.len()];
    for (rank, (index, _)) in indexed.into_iter().enumerate() {
        result[index] = Some(if denominator == 0.0 {
            0.5
        } else {
            f64::from(u32::try_from(rank).unwrap_or(u32::MAX)) / denominator
        });
    }
    result
}

fn zscore_values(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let observed: Vec<_> = values.iter().flatten().copied().collect();
    let Some((mean, deviation)) = mean_and_deviation(&observed) else {
        return vec![None; values.len()];
    };
    values
        .iter()
        .map(|value| {
            value.map(|value| {
                if deviation == 0.0 {
                    0.0
                } else {
                    (value - mean) / deviation
                }
            })
        })
        .collect()
}

fn winsorize_values(values: &[Option<f64>], width: f64) -> Vec<Option<f64>> {
    let observed: Vec<_> = values.iter().flatten().copied().collect();
    let Some((mean, deviation)) = mean_and_deviation(&observed) else {
        return vec![None; values.len()];
    };
    let lower = mean - width * deviation;
    let upper = mean + width * deviation;
    values
        .iter()
        .map(|value| value.map(|value| value.clamp(lower, upper)))
        .collect()
}

fn mean_and_deviation(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let count = f64::from(u32::try_from(values.len()).ok()?);
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    Some((mean, variance.sqrt()))
}

fn time_series(
    operator: &str,
    input: &NumericColumn,
    dataset: &ColumnarDataset,
    window: usize,
) -> NumericColumn {
    let rows = dataset.timestamps.len();
    let assets = dataset.assets.len();
    let mut result = vec![None; rows * assets];
    for row in 0..rows {
        for asset in 0..assets {
            let index = row * assets + asset;
            match operator {
                "ts_delta" if row >= window => {
                    result[index] = match (
                        input.values[index],
                        input.values[(row - window) * assets + asset],
                    ) {
                        (Some(current), Some(previous)) => Some(current - previous),
                        _ => None,
                    };
                }
                "ts_mean" | "ts_std_dev" | "ts_zscore" | "ts_rank" if row + 1 >= window => {
                    let start = row + 1 - window;
                    let values: Vec<_> = (start..=row)
                        .map(|item| input.values[item * assets + asset])
                        .collect();
                    if values.iter().all(Option::is_some) {
                        match operator {
                            "ts_mean" => {
                                let sum = values.iter().flatten().sum::<f64>();
                                result[index] = Some(
                                    sum / f64::from(u32::try_from(window).unwrap_or(u32::MAX)),
                                );
                            }
                            "ts_std_dev" | "ts_zscore" => {
                                let observed: Vec<_> = values.iter().flatten().copied().collect();
                                if let Some((mean, deviation)) = mean_and_deviation(&observed) {
                                    result[index] = Some(if operator == "ts_std_dev" {
                                        deviation
                                    } else if deviation == 0.0 {
                                        0.0
                                    } else {
                                        (observed[window - 1] - mean) / deviation
                                    });
                                }
                            }
                            "ts_rank" => {
                                let ranked = rank_values(&values);
                                result[index] = ranked[window - 1];
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                _ => {}
            }
        }
    }
    NumericColumn { values: result }
}

fn linear_decay(input: &NumericColumn, dataset: &ColumnarDataset, window: usize) -> NumericColumn {
    let rows = dataset.timestamps.len();
    let assets = dataset.assets.len();
    let mut result = vec![None; rows * assets];
    let denominator =
        f64::from(u32::try_from(window.saturating_mul(window + 1) / 2).unwrap_or(u32::MAX));
    for row in window.saturating_sub(1)..rows {
        for asset in 0..assets {
            let mut total = 0.0;
            let mut complete = true;
            for offset in 0..window {
                let weight = f64::from(u32::try_from(window - offset).unwrap_or(u32::MAX));
                if let Some(value) = input.values[(row - offset) * assets + asset] {
                    total += value * weight;
                } else {
                    complete = false;
                    break;
                }
            }
            if complete {
                result[row * assets + asset] = Some(total / denominator);
            }
        }
    }
    NumericColumn { values: result }
}

fn group_rank(
    input: &NumericColumn,
    dataset: &ColumnarDataset,
    group: &str,
) -> Result<NumericColumn, NumericError> {
    let labels = dataset
        .groups
        .get(group)
        .ok_or_else(|| NumericError::Evaluation(format!("dataset lacks group `{group}`")))?;
    let assets = dataset.assets.len();
    let mut result = vec![None; input.values.len()];
    for row in 0..dataset.timestamps.len() {
        let mut members = BTreeMap::<&str, Vec<usize>>::new();
        for (asset, label) in labels.iter().enumerate() {
            members.entry(label).or_default().push(asset);
        }
        for indices in members.values() {
            let values: Vec<_> = indices
                .iter()
                .map(|asset| input.values[row * assets + asset])
                .collect();
            for (position, ranked) in rank_values(&values).into_iter().enumerate() {
                result[row * assets + indices[position]] = ranked;
            }
        }
    }
    Ok(NumericColumn { values: result })
}

fn evaluate_period(
    name: &str,
    start: usize,
    end: usize,
    signal: &NumericColumn,
    dataset: &ColumnarDataset,
    config: &EvaluationConfig,
    include_daily: bool,
) -> (SplitEvidence, Vec<DailyEvidence>) {
    let assets = dataset.assets.len();
    let mut signal_samples = Vec::new();
    let mut return_samples = Vec::new();
    let mut daily_gross = Vec::new();
    let mut daily_net = Vec::new();
    let mut daily_turnover = Vec::new();
    let mut daily = Vec::new();
    let mut previous = vec![0.0; assets];
    for row in start..end - config.horizon_rows {
        let row_signal = &signal.values[row * assets..(row + 1) * assets];
        let positions = normalized_positions(row_signal, config.truncation);
        let mut gross = 0.0;
        let mut has_position = false;
        for asset in 0..assets {
            let outcome = future_return(dataset, row, asset, config.horizon_rows);
            if let (Some(signal), Some(outcome)) = (row_signal[asset], outcome) {
                signal_samples.push(signal);
                return_samples.push(outcome);
            }
            if let (Some(position), Some(outcome)) = (positions[asset], outcome) {
                gross += position * outcome;
                has_position = true;
            }
        }
        if has_position {
            let turnover = positions
                .iter()
                .enumerate()
                .map(|(asset, value)| (value.unwrap_or(0.0) - previous[asset]).abs())
                .sum::<f64>();
            previous = positions.iter().map(|value| value.unwrap_or(0.0)).collect();
            daily_gross.push(gross);
            daily_turnover.push(turnover);
            let net = gross - turnover * config.transaction_cost_bps / 10_000.0;
            daily_net.push(net);
            if include_daily {
                let target_end_row = row + config.horizon_rows;
                daily.push(DailyEvidence {
                    signal_row: row,
                    target_end_row,
                    signal_timestamp: dataset.timestamps[row],
                    target_timestamp: dataset.timestamps[target_end_row],
                    measured_gross_return: gross,
                    measured_net_return: net,
                    measured_turnover: turnover,
                });
            }
        }
    }
    let annualization = config.annualization_rows.map(f64::from);
    let gross_sharpe = annualization.and_then(|rows| annual_sharpe(&daily_gross, rows));
    let net_sharpe = annualization.and_then(|rows| annual_sharpe(&daily_net, rows));
    let mean_gross = mean(&daily_gross);
    let mean_net = mean(&daily_net);
    let mean_turnover = mean(&daily_turnover);
    let brain_proxy = config.brain_proxy.then(|| {
        brain_proxy_metrics(
            gross_sharpe,
            mean_gross,
            mean_turnover,
            &daily_gross,
            annualization.expect("brain proxy validation requires annualization rows"),
        )
    });
    let evidence = SplitEvidence {
        split: name.to_owned(),
        start_row: start,
        end_row_exclusive: end,
        observations: signal_samples.len(),
        measured_pearson_correlation: pearson(&signal_samples, &return_samples),
        measured_mean_row_gross_return: mean_gross,
        measured_mean_row_net_return: mean_net,
        measured_mean_row_turnover: mean_turnover,
        measured_annual_gross_return: annualization.and_then(|rows| mean_gross.map(|x| x * rows)),
        measured_annual_net_return: annualization.and_then(|rows| mean_net.map(|x| x * rows)),
        measured_gross_sharpe: gross_sharpe,
        measured_net_sharpe: net_sharpe,
        measured_additive_gross_drawdown: additive_drawdown(&daily_gross),
        measured_additive_net_drawdown: additive_drawdown(&daily_net),
        brain_proxy,
    };
    (evidence, daily)
}

fn annual_sharpe(values: &[f64], annualization_rows: f64) -> Option<f64> {
    if values.len() < 2 || !annualization_rows.is_finite() || annualization_rows <= 0.0 {
        return None;
    }
    let average = mean(values)?;
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
    (variance > 0.0).then(|| average / variance.sqrt() * annualization_rows.sqrt())
}

fn additive_drawdown(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut cumulative = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut drawdown = 0.0_f64;
    for value in values {
        cumulative += value;
        peak = peak.max(cumulative);
        drawdown = drawdown.max(peak - cumulative);
    }
    Some(drawdown)
}

fn brain_proxy_metrics(
    sharpe: Option<f64>,
    mean_gross: Option<f64>,
    turnover: Option<f64>,
    daily_gross: &[f64],
    annualization_rows: f64,
) -> BrainProxyMetrics {
    let returns = mean_gross.map(|value| value * annualization_rows * 2.0);
    let drawdown = additive_drawdown(daily_gross).map(|value| value * 2.0);
    let margin = mean_gross
        .zip(turnover)
        .and_then(|(value, turnover)| (turnover > 0.0).then_some(value / turnover));
    let fitness = sharpe
        .zip(returns)
        .zip(turnover)
        .map(|((sharpe, returns), turnover)| sharpe * (returns.abs() / turnover.max(0.125)).sqrt());
    BrainProxyMetrics {
        methodology: "brain-proxy-v1".to_owned(),
        estimated_sharpe: sharpe,
        estimated_returns: returns,
        estimated_turnover: turnover,
        estimated_drawdown: drawdown,
        estimated_margin: margin,
        estimated_fitness: fitness,
        remote_platform_result: false,
    }
}

fn future_return(
    dataset: &ColumnarDataset,
    row: usize,
    asset: usize,
    horizon: usize,
) -> Option<f64> {
    let assets = dataset.assets.len();
    let mut result = 0.0;
    for future in row + 1..=row + horizon {
        result += dataset.realized_returns.values[future * assets + asset]?;
    }
    Some(result)
}

fn normalized_positions(signal: &[Option<f64>], truncation: f64) -> Vec<Option<f64>> {
    let observed: Vec<_> = signal.iter().flatten().copied().collect();
    let Some(mean) = mean(&observed) else {
        return vec![None; signal.len()];
    };
    let centered: Vec<_> = signal
        .iter()
        .map(|value| value.map(|value| value - mean))
        .collect();
    let gross = centered
        .iter()
        .flatten()
        .map(|value| value.abs())
        .sum::<f64>();
    if gross == 0.0 {
        return vec![None; signal.len()];
    }
    let mut positions: Vec<_> = centered
        .into_iter()
        .map(|value| value.map(|value| value / gross))
        .collect();
    if truncation == 0.0 {
        return positions;
    }
    let observed = positions.iter().flatten().count();
    if f64::from(u32::try_from(observed).unwrap_or(u32::MAX)) * truncation < 1.0 {
        return vec![None; signal.len()];
    }
    let mut fixed = vec![false; positions.len()];
    loop {
        let fixed_gross = positions
            .iter()
            .enumerate()
            .filter(|(index, _)| fixed[*index])
            .filter_map(|(_, value)| *value)
            .map(f64::abs)
            .sum::<f64>();
        let free_gross = positions
            .iter()
            .enumerate()
            .filter(|(index, _)| !fixed[*index])
            .filter_map(|(_, value)| *value)
            .map(f64::abs)
            .sum::<f64>();
        if free_gross == 0.0 {
            break;
        }
        let scale = (1.0 - fixed_gross).max(0.0) / free_gross;
        let mut newly_fixed = false;
        for (index, value) in positions.iter_mut().enumerate() {
            if fixed[index] {
                continue;
            }
            if let Some(weight) = value {
                *weight *= scale;
                if weight.abs() > truncation {
                    *weight = weight.signum() * truncation;
                    fixed[index] = true;
                    newly_fixed = true;
                }
            }
        }
        if !newly_fixed {
            break;
        }
    }
    positions
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| {
        values.iter().sum::<f64>() / f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX))
    })
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = mean(left)?;
    let right_mean = mean(right)?;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = left - left_mean;
        let right = right - right_mean;
        covariance += left * right;
        left_variance += left * left;
        right_variance += right * right;
    }
    let denominator = (left_variance * right_variance).sqrt();
    (denominator > 0.0).then_some(covariance / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_catalog_enables_dataset_specific_fields() {
        let mut dataset = fixture();
        let values = dataset.fields.remove("close").unwrap();
        dataset.fields.insert("synthetic_flow".to_owned(), values);
        let mut catalog = builtin_catalog().clone();
        catalog.fields = vec![crate::FieldSpec {
            name: "synthetic_flow".to_owned(),
            kind: crate::ExprKind::Signal,
            family: "synthetic".to_owned(),
            allowed_roles: vec!["signal_input".to_owned(), "condition_input".to_owned()],
        }];
        catalog.validate().unwrap();
        let evidence =
            evaluate_numeric_with_catalog("rank(synthetic_flow)", &dataset, &config(), &catalog)
                .unwrap();
        assert_eq!(evidence.expression, "rank(synthetic_flow)");
    }

    fn fixture() -> ColumnarDataset {
        let rows = 12;
        let assets = vec!["a".to_owned(), "b".to_owned()];
        let base = [
            1.0, -3.0, 2.0, 7.0, -5.0, 4.0, 9.0, -2.0, 6.0, -8.0, 3.0, 5.0,
        ];
        let mut close = Vec::new();
        let mut realized = vec![Some(0.0), Some(0.0)];
        for value in base {
            close.extend([Some(value), Some(-value)]);
            realized.extend([Some(value), Some(-value)]);
        }
        realized.truncate(rows * assets.len());
        ColumnarDataset {
            schema: 1,
            dataset_id: "synthetic-alignment-v1".to_owned(),
            timestamps: (0..rows).map(|row| i64::try_from(row).unwrap()).collect(),
            assets,
            fields: BTreeMap::from([("close".to_owned(), NumericColumn { values: close })]),
            realized_returns: NumericColumn { values: realized },
            groups: BTreeMap::from([(
                "sector".to_owned(),
                vec!["one".to_owned(), "one".to_owned()],
            )]),
            source_label: "license-safe synthetic fixture".to_owned(),
            preprocessing: vec!["none".to_owned()],
        }
    }

    fn config() -> EvaluationConfig {
        EvaluationConfig {
            schema: 1,
            horizon_rows: 1,
            evaluation_start: 0,
            train_end: 6,
            validation_end: 9,
            transaction_cost_bps: 5.0,
            annualization_rows: None,
            signal_decay: 0,
            truncation: 0.0,
            include_daily: false,
            brain_proxy: false,
        }
    }

    #[test]
    fn target_alignment_uses_only_returns_after_the_signal_row() {
        let evidence = evaluate_numeric("close", &fixture(), &config()).unwrap();
        assert!((evidence.splits[0].measured_pearson_correlation.unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(evidence.splits[0].observations, 10);
        assert!(evidence.alignment.contains("t+1"));
        assert!(!evidence.structural_score_used);
    }

    #[test]
    fn rolling_operators_do_not_read_future_rows() {
        let dataset = fixture();
        let expression = parse_expression("ts_mean(close, 2)").unwrap();
        let before = evaluate_signal(&expression, &dataset).unwrap();
        let mut changed = dataset.clone();
        changed.fields.get_mut("close").unwrap().values[20] = Some(1_000_000.0);
        let after = evaluate_signal(&expression, &changed).unwrap();
        assert_eq!(&before.values[..20], &after.values[..20]);
    }

    #[test]
    fn rolling_standard_deviation_and_zscore_use_population_window() {
        let dataset = fixture();
        let deviation =
            evaluate_signal(&parse_expression("ts_std_dev(close, 2)").unwrap(), &dataset).unwrap();
        let standardized =
            evaluate_signal(&parse_expression("ts_zscore(close, 2)").unwrap(), &dataset).unwrap();
        assert_eq!(deviation.values[0], None);
        assert!((deviation.values[2].unwrap() - 2.0).abs() < 1e-12);
        assert!((standardized.values[2].unwrap() + 1.0).abs() < 1e-12);
        assert!((standardized.values[4].unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn root_linear_decay_weights_current_row_most_and_stays_causal() {
        let dataset = fixture();
        let input = dataset.fields["close"].clone();
        let decayed = linear_decay(&input, &dataset, 2);
        assert_eq!(decayed.values[0], None);
        assert!((decayed.values[2].unwrap() - (-5.0 / 3.0)).abs() < 1e-12);
        let mut changed = input.clone();
        changed.values[4] = Some(1_000_000.0);
        let after = linear_decay(&changed, &dataset, 2);
        assert_eq!(&decayed.values[..4], &after.values[..4]);
    }

    #[test]
    fn truncation_caps_and_redistributes_to_unit_gross() {
        let positions =
            normalized_positions(&[Some(100.0), Some(1.0), Some(-1.0), Some(-2.0)], 0.40);
        let weights: Vec<_> = positions.into_iter().flatten().collect();
        assert!(weights.iter().all(|weight| weight.abs() <= 0.40 + 1e-12));
        assert!((weights.iter().map(|weight| weight.abs()).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn validation_and_holdout_targets_never_cross_split_boundaries() {
        let evidence = evaluate_numeric("rank(close)", &fixture(), &config()).unwrap();
        assert_eq!(evidence.splits[1].observations, 4);
        assert_eq!(evidence.splits[2].observations, 4);
    }

    #[test]
    fn evaluation_start_keeps_prior_rows_as_operator_warm_up() {
        let mut config = config();
        config.evaluation_start = 2;
        config.annualization_rows = Some(252);
        config.include_daily = true;
        config.brain_proxy = true;
        let evidence = evaluate_numeric("ts_mean(close, 2)", &fixture(), &config).unwrap();
        assert_eq!(evidence.overall.start_row, 2);
        assert_eq!(evidence.daily[0].signal_row, 2);
        assert_eq!(evidence.daily[0].target_end_row, 3);
        assert_eq!(evidence.daily[0].signal_timestamp, 2);
        assert_eq!(evidence.daily[0].target_timestamp, 3);
        assert_eq!(
            evidence.overall.brain_proxy.as_ref().unwrap().methodology,
            "brain-proxy-v1"
        );
        assert!(
            !evidence
                .overall
                .brain_proxy
                .as_ref()
                .unwrap()
                .remote_platform_result
        );
    }

    #[test]
    fn brain_proxy_formulas_are_explicit_and_deterministic() {
        let gross = [0.01, -0.02, 0.03];
        let turnover = Some(0.4 / 3.0);
        let average = mean(&gross);
        let sharpe = annual_sharpe(&gross, 252.0);
        let proxy = brain_proxy_metrics(sharpe, average, turnover, &gross, 252.0);
        let expected_returns = (0.02 / 3.0) * 252.0 * 2.0;
        assert!((proxy.estimated_returns.unwrap() - expected_returns).abs() < 1e-12);
        assert!((proxy.estimated_turnover.unwrap() - 0.4 / 3.0).abs() < 1e-12);
        assert!((proxy.estimated_drawdown.unwrap() - 0.04).abs() < 1e-12);
        assert!((proxy.estimated_margin.unwrap() - 0.05).abs() < 1e-12);
        let expected_fitness =
            sharpe.unwrap() * (expected_returns.abs() / (0.4_f64 / 3.0).max(0.125)).sqrt();
        assert!((proxy.estimated_fitness.unwrap() - expected_fitness).abs() < 1e-12);
    }

    #[test]
    fn brain_proxy_requires_annualization_rows() {
        let mut invalid = config();
        invalid.brain_proxy = true;
        let error = evaluate_numeric("close", &fixture(), &invalid).unwrap_err();
        assert!(error.to_string().contains("annualization"));
    }
}
