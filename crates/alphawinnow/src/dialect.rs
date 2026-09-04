//! Generic, local-only target-dialect lowering and compatibility checks.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Expr, ExprKind, canonical, parse_expression};

pub const DIALECT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnaryScalarLowering {
    pub replacement: String,
    pub scalar: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialectSpec {
    pub schema: u32,
    pub name: String,
    pub allowed_operators: BTreeSet<String>,
    pub allowed_fields: BTreeSet<String>,
    #[serde(default)]
    pub unary_scalar_lowerings: BTreeMap<String, UnaryScalarLowering>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialectArtifact {
    pub schema: u32,
    pub dialect: String,
    pub source_expression: String,
    pub compiled_expression: String,
    pub lowerings_applied: Vec<String>,
    pub operators: BTreeSet<String>,
    pub fields: BTreeSet<String>,
}

#[derive(Debug, Error)]
pub enum DialectError {
    #[error("unsupported dialect schema {0}")]
    Schema(u32),
    #[error("dialect name and allowlists must be non-empty")]
    Empty,
    #[error("invalid lowering for `{0}`")]
    Lowering(String),
    #[error("target dialect does not allow operator `{0}`")]
    Operator(String),
    #[error("target dialect does not allow field `{0}`")]
    Field(String),
    #[error(transparent)]
    Expression(#[from] crate::ExpressionError),
}

impl DialectSpec {
    /// Validate the immutable target contract before compiling an expression.
    ///
    /// # Errors
    /// Returns a schema, allowlist, or lowering diagnostic for an invalid
    /// dialect artifact.
    pub fn validate(&self) -> Result<(), DialectError> {
        if self.schema != DIALECT_SCHEMA {
            return Err(DialectError::Schema(self.schema));
        }
        if self.name.trim().is_empty()
            || self.allowed_operators.is_empty()
            || self.allowed_fields.is_empty()
        {
            return Err(DialectError::Empty);
        }
        for (source, lowering) in &self.unary_scalar_lowerings {
            if source.trim().is_empty()
                || lowering.replacement.trim().is_empty()
                || !lowering.scalar.is_finite()
                || !self.allowed_operators.contains(&lowering.replacement)
            {
                return Err(DialectError::Lowering(source.clone()));
            }
        }
        Ok(())
    }
}

/// Lower one expression and fail closed on unavailable target symbols.
///
/// # Errors
/// Returns an expression, dialect, unavailable-operator, or unavailable-field
/// diagnostic without emitting a partially compiled expression.
pub fn compile_dialect(
    source: &str,
    dialect: &DialectSpec,
) -> Result<DialectArtifact, DialectError> {
    dialect.validate()?;
    let expression = parse_expression(source)?;
    let mut lowerings = Vec::new();
    let compiled = lower(&expression, dialect, &mut lowerings)?;
    let mut operators = BTreeSet::new();
    let mut fields = BTreeSet::new();
    collect(&compiled, &mut operators, &mut fields);
    for operator in &operators {
        if !dialect.allowed_operators.contains(operator) {
            return Err(DialectError::Operator(operator.clone()));
        }
    }
    for field in &fields {
        if !dialect.allowed_fields.contains(field) {
            return Err(DialectError::Field(field.clone()));
        }
    }
    Ok(DialectArtifact {
        schema: DIALECT_SCHEMA,
        dialect: dialect.name.clone(),
        source_expression: canonical(&expression),
        compiled_expression: canonical(&compiled),
        lowerings_applied: lowerings,
        operators,
        fields,
    })
}

fn lower(
    expression: &Expr,
    dialect: &DialectSpec,
    applied: &mut Vec<String>,
) -> Result<Expr, DialectError> {
    Ok(match expression {
        Expr::Field { .. } | Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => {
            expression.clone()
        }
        Expr::UnaryCall {
            op,
            arg,
            kwargs,
            kind,
        } => {
            let arg = lower(arg, dialect, applied)?;
            let kwargs = lower_kwargs(kwargs, dialect, applied)?;
            if let Some(rule) = dialect.unary_scalar_lowerings.get(op) {
                if !kwargs.is_empty() || *kind != ExprKind::Signal {
                    return Err(DialectError::Lowering(op.clone()));
                }
                applied.push(format!("{op}->{}", rule.replacement));
                Expr::VariadicCall {
                    op: rule.replacement.clone(),
                    args: vec![Expr::Scalar { value: rule.scalar }, arg],
                    kwargs: BTreeMap::new(),
                    kind: *kind,
                }
            } else {
                Expr::UnaryCall {
                    op: op.clone(),
                    arg: Box::new(arg),
                    kwargs,
                    kind: *kind,
                }
            }
        }
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            kind,
        } => Expr::BinaryCall {
            op: op.clone(),
            left: Box::new(lower(left, dialect, applied)?),
            right: Box::new(lower(right, dialect, applied)?),
            kwargs: lower_kwargs(kwargs, dialect, applied)?,
            kind: *kind,
        },
        Expr::VariadicCall {
            op,
            args,
            kwargs,
            kind,
        } => Expr::VariadicCall {
            op: op.clone(),
            args: args
                .iter()
                .map(|arg| lower(arg, dialect, applied))
                .collect::<Result<_, _>>()?,
            kwargs: lower_kwargs(kwargs, dialect, applied)?,
            kind: *kind,
        },
    })
}

fn lower_kwargs(
    kwargs: &BTreeMap<String, Expr>,
    dialect: &DialectSpec,
    applied: &mut Vec<String>,
) -> Result<BTreeMap<String, Expr>, DialectError> {
    kwargs
        .iter()
        .map(|(name, value)| Ok((name.clone(), lower(value, dialect, applied)?)))
        .collect()
}

fn collect(expression: &Expr, operators: &mut BTreeSet<String>, fields: &mut BTreeSet<String>) {
    match expression {
        Expr::Field { name } => {
            fields.insert(name.clone());
        }
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => {
            operators.insert(op.clone());
            collect(arg, operators, fields);
            for value in kwargs.values() {
                collect(value, operators, fields);
            }
        }
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => {
            operators.insert(op.clone());
            collect(left, operators, fields);
            collect(right, operators, fields);
            for value in kwargs.values() {
                collect(value, operators, fields);
            }
        }
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => {
            operators.insert(op.clone());
            for value in args.iter().chain(kwargs.values()) {
                collect(value, operators, fields);
            }
        }
        Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialect() -> DialectSpec {
        DialectSpec {
            schema: DIALECT_SCHEMA,
            name: "fixture-v1".to_owned(),
            allowed_operators: ["multiply", "rank", "ts_delta"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            allowed_fields: ["returns"].into_iter().map(str::to_owned).collect(),
            unary_scalar_lowerings: BTreeMap::from([(
                "negate".to_owned(),
                UnaryScalarLowering {
                    replacement: "multiply".to_owned(),
                    scalar: -1.0,
                },
            )]),
        }
    }

    #[test]
    fn nested_unary_lowering_is_structural_and_auditable() {
        let artifact = compile_dialect("rank(negate(ts_delta(returns, 5)))", &dialect()).unwrap();
        assert_eq!(
            artifact.compiled_expression,
            "rank(multiply(-1, ts_delta(returns, 5)))"
        );
        assert_eq!(artifact.lowerings_applied, ["negate->multiply"]);
    }

    #[test]
    fn unavailable_symbols_fail_closed() {
        let error = compile_dialect("zscore(returns)", &dialect()).unwrap_err();
        assert!(matches!(error, DialectError::Operator(value) if value == "zscore"));
    }
}
