use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    ast::{Expr, ExprKind},
    canonical::{canonical, digest},
};

/// Conservative, machine-readable reason that a signal cannot enter a pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    /// Exact structural analysis proves that the signal is identically zero.
    ProvablyZero,
    /// Exact structural analysis proves that the signal is constant.
    ProvablyConstant,
}

impl std::fmt::Display for RejectionReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ProvablyZero => "provably_zero",
            Self::ProvablyConstant => "provably_constant",
        })
    }
}

/// Semantic identity and admission analysis computed in a single pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpressionAnalysis {
    pub semantic_canonical: String,
    pub semantic_fingerprint: String,
    pub rejection_reason: Option<RejectionReason>,
}

/// Analyze final-root scale identity and conservative triviality.
#[must_use]
pub fn analyze_expression(expression: &Expr) -> ExpressionAnalysis {
    let semantic_canonical = canonical(&semantic_normal_form(expression));
    let rejection_reason = match signal_fact(expression) {
        SignalFact::Zero => Some(RejectionReason::ProvablyZero),
        SignalFact::Constant => Some(RejectionReason::ProvablyConstant),
        SignalFact::Unknown => None,
    };
    ExpressionAnalysis {
        semantic_fingerprint: digest(&semantic_canonical),
        semantic_canonical,
        rejection_reason,
    }
}

/// Build a conservative normal form for final-root scaling only.
#[must_use]
pub(crate) fn semantic_normal_form(expression: &Expr) -> Expr {
    let simplified = simplify_deterministic_conditionals(expression);
    semantic_root_normal_form(&simplified)
}

fn semantic_root_normal_form(expression: &Expr) -> Expr {
    if let Some(repeated) = repeated_root_addend(expression) {
        return semantic_normal_form(repeated);
    }

    if matches!(expression.operator(), Some("multiply" | "divide")) {
        let mut scalar = 1.0;
        let mut factors = Vec::new();
        let recognized = collect_root_product(expression, &mut scalar, &mut factors);
        if recognized && scalar.is_finite() && scalar != 0.0 {
            if scalar > 0.0 {
                return rebuild_positive_product(factors);
            }
            return rebuild_negative_product(factors);
        }
    }
    expression.clone()
}

fn simplify_deterministic_conditionals(expression: &Expr) -> Expr {
    match expression {
        Expr::Field { .. } | Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => {
            expression.clone()
        }
        Expr::UnaryCall {
            op,
            arg,
            kwargs,
            kind,
        } => Expr::UnaryCall {
            op: op.clone(),
            arg: Box::new(simplify_deterministic_conditionals(arg)),
            kwargs: kwargs
                .iter()
                .map(|(name, value)| (name.clone(), simplify_deterministic_conditionals(value)))
                .collect(),
            kind: *kind,
        },
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            kind,
        } => Expr::BinaryCall {
            op: op.clone(),
            left: Box::new(simplify_deterministic_conditionals(left)),
            right: Box::new(simplify_deterministic_conditionals(right)),
            kwargs: kwargs
                .iter()
                .map(|(name, value)| (name.clone(), simplify_deterministic_conditionals(value)))
                .collect(),
            kind: *kind,
        },
        Expr::VariadicCall {
            op,
            args,
            kwargs,
            kind,
        } => {
            let args: Vec<_> = args
                .iter()
                .map(simplify_deterministic_conditionals)
                .collect();
            if op == "if_else"
                && let [condition, when_true, when_false] = args.as_slice()
            {
                if let Expr::Bool { value } = condition {
                    return if *value {
                        when_true.clone()
                    } else {
                        when_false.clone()
                    };
                }
                if canonical(when_true) == canonical(when_false) {
                    return when_true.clone();
                }
            }
            Expr::VariadicCall {
                op: op.clone(),
                args,
                kwargs: kwargs
                    .iter()
                    .map(|(name, value)| (name.clone(), simplify_deterministic_conditionals(value)))
                    .collect(),
                kind: *kind,
            }
        }
    }
}

fn repeated_root_addend(expression: &Expr) -> Option<&Expr> {
    let Expr::VariadicCall { op, args, .. } = expression else {
        return None;
    };
    if op != "add" {
        return None;
    }
    if let Some(first) = args.first() {
        let direct_identity = canonical(first);
        if args.len() >= 2
            && args
                .iter()
                .all(|addend| canonical(addend) == direct_identity)
        {
            return Some(first);
        }
    }
    let mut addends = Vec::new();
    collect_root_addends(expression, &mut addends);
    let first = addends.first()?;
    let identity = canonical(first);
    (addends.len() >= 2 && addends.iter().all(|addend| canonical(addend) == identity))
        .then_some(*first)
}

fn collect_root_addends<'a>(expression: &'a Expr, addends: &mut Vec<&'a Expr>) {
    if let Expr::VariadicCall { op, args, .. } = expression
        && op == "add"
    {
        for arg in args {
            collect_root_addends(arg, addends);
        }
    } else {
        addends.push(expression);
    }
}

fn collect_root_product(expression: &Expr, scalar: &mut f64, factors: &mut Vec<Expr>) -> bool {
    match expression {
        Expr::Scalar { value } => {
            *scalar *= value;
            true
        }
        Expr::VariadicCall { op, args, .. } if op == "multiply" => args
            .iter()
            .all(|arg| collect_root_product(arg, scalar, factors)),
        Expr::BinaryCall {
            op, left, right, ..
        } if op == "divide" => {
            if let Expr::Scalar { value } = right.as_ref() {
                if !collect_root_product(left, scalar, factors) {
                    return false;
                }
                *scalar /= value;
                true
            } else {
                false
            }
        }
        _ => {
            factors.push(expression.clone());
            true
        }
    }
}

fn rebuild_positive_product(mut factors: Vec<Expr>) -> Expr {
    if factors.len() == 1 {
        return semantic_normal_form(&factors.remove(0));
    }
    multiply(factors)
}

fn rebuild_negative_product(mut factors: Vec<Expr>) -> Expr {
    if factors.len() == 1 {
        let base = semantic_normal_form(&factors.remove(0));
        return multiply(vec![Expr::Scalar { value: -1.0 }, base]);
    }
    factors.push(Expr::Scalar { value: -1.0 });
    multiply(factors)
}

fn multiply(args: Vec<Expr>) -> Expr {
    Expr::VariadicCall {
        op: "multiply".to_owned(),
        args,
        kwargs: BTreeMap::new(),
        kind: ExprKind::Signal,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignalFact {
    Unknown,
    Constant,
    Zero,
}

fn signal_fact(expression: &Expr) -> SignalFact {
    match expression {
        Expr::Field { .. } => SignalFact::Unknown,
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => unary_fact(op, signal_fact(arg), kwargs),
        Expr::BinaryCall {
            op, left, right, ..
        } => binary_fact(op, left, right),
        Expr::VariadicCall { op, args, .. } => variadic_fact(op, args),
        Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => SignalFact::Constant,
    }
}

fn unary_fact(op: &str, arg: SignalFact, kwargs: &BTreeMap<String, Expr>) -> SignalFact {
    match (op, arg) {
        ("negate" | "winsorize", SignalFact::Zero) => SignalFact::Zero,
        ("clip", SignalFact::Zero) => {
            let lower = scalar_keyword(kwargs, "lower");
            let upper = scalar_keyword(kwargs, "upper");
            if lower.is_some_and(|value| value <= 0.0) && upper.is_some_and(|value| value >= 0.0) {
                SignalFact::Zero
            } else {
                SignalFact::Constant
            }
        }
        (_, SignalFact::Constant | SignalFact::Zero) => SignalFact::Constant,
        (_, SignalFact::Unknown) => SignalFact::Unknown,
    }
}

fn binary_fact(op: &str, left: &Expr, right: &Expr) -> SignalFact {
    let left_fact = signal_fact(left);
    match op {
        "subtract" if canonical(left) == canonical(right) => SignalFact::Zero,
        "subtract" => {
            let right_fact = signal_fact(right);
            if left_fact == SignalFact::Zero && right_fact == SignalFact::Zero {
                SignalFact::Zero
            } else {
                combine_constant(left_fact, right_fact)
            }
        }
        "divide" | "ts_mean" => preserve_zero(left_fact),
        "ts_delta" | "ts_std_dev" | "ts_zscore" => match left_fact {
            SignalFact::Zero | SignalFact::Constant => SignalFact::Zero,
            SignalFact::Unknown => SignalFact::Unknown,
        },
        "ts_rank" | "group_rank" => match left_fact {
            SignalFact::Zero | SignalFact::Constant => SignalFact::Constant,
            SignalFact::Unknown => SignalFact::Unknown,
        },
        _ => SignalFact::Unknown,
    }
}

fn variadic_fact(op: &str, args: &[Expr]) -> SignalFact {
    match op {
        "multiply" => {
            if args.iter().any(is_literal_zero)
                || args.iter().any(|arg| signal_fact(arg) == SignalFact::Zero)
            {
                SignalFact::Zero
            } else if args
                .iter()
                .filter(|arg| arg.kind() == ExprKind::Signal)
                .all(|arg| signal_fact(arg) == SignalFact::Constant)
            {
                SignalFact::Constant
            } else {
                SignalFact::Unknown
            }
        }
        "add" => {
            let facts: Vec<_> = args.iter().map(signal_fact).collect();
            if facts.iter().all(|fact| *fact == SignalFact::Zero) {
                SignalFact::Zero
            } else if facts
                .iter()
                .all(|fact| matches!(fact, SignalFact::Zero | SignalFact::Constant))
            {
                SignalFact::Constant
            } else {
                SignalFact::Unknown
            }
        }
        "if_else" => conditional_fact(args),
        _ => SignalFact::Unknown,
    }
}

fn conditional_fact(args: &[Expr]) -> SignalFact {
    let [condition, when_true, when_false] = args else {
        return SignalFact::Unknown;
    };
    if let Expr::Bool { value } = condition {
        return signal_fact(if *value { when_true } else { when_false });
    }
    if canonical(when_true) == canonical(when_false) {
        return signal_fact(when_true);
    }
    let when_true = signal_fact(when_true);
    let when_false = signal_fact(when_false);
    if when_true == SignalFact::Zero && when_false == SignalFact::Zero {
        SignalFact::Zero
    } else {
        SignalFact::Unknown
    }
}

const fn combine_constant(left: SignalFact, right: SignalFact) -> SignalFact {
    if matches!(left, SignalFact::Zero | SignalFact::Constant)
        && matches!(right, SignalFact::Zero | SignalFact::Constant)
    {
        SignalFact::Constant
    } else {
        SignalFact::Unknown
    }
}

const fn preserve_zero(fact: SignalFact) -> SignalFact {
    fact
}

fn scalar_keyword(kwargs: &BTreeMap<String, Expr>, name: &str) -> Option<f64> {
    match kwargs.get(name) {
        Some(Expr::Scalar { value }) => Some(*value),
        _ => None,
    }
}

fn is_literal_zero(expression: &Expr) -> bool {
    matches!(expression, Expr::Scalar { value } if *value == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_expression;

    #[test]
    fn deterministic_conditionals_share_the_selected_branch_family() {
        let selected = parse_expression("rank(close)").unwrap();
        for expression in [
            "if_else(true, rank(close), open)",
            "if_else(false, open, rank(close))",
            "multiply(if_else(true, rank(close), open), 2)",
        ] {
            let expression = parse_expression(expression).unwrap();
            assert_eq!(
                analyze_expression(&expression).semantic_fingerprint,
                analyze_expression(&selected).semantic_fingerprint
            );
        }
    }

    #[test]
    fn reports_machine_readable_zero_and_constant_reasons() {
        let zero = parse_expression("subtract(close, close)").unwrap();
        let constant = parse_expression("rank(multiply(close, 0))").unwrap();
        assert_eq!(
            analyze_expression(&zero).rejection_reason,
            Some(RejectionReason::ProvablyZero)
        );
        assert_eq!(
            analyze_expression(&constant).rejection_reason,
            Some(RejectionReason::ProvablyConstant)
        );
        assert_eq!(
            serde_json::to_string(&RejectionReason::ProvablyZero).unwrap(),
            "\"provably_zero\""
        );
    }

    #[test]
    fn stays_conservative_for_a_varying_condition_with_constant_branches() {
        let expression = parse_expression(
            "if_else(greater(close, 0), rank(multiply(close, 0)), clip(multiply(close, 0), lower=1, upper=2))",
        )
        .unwrap();
        assert_eq!(analyze_expression(&expression).rejection_reason, None);
    }
}
