//! Opt-in exact unit-scaling reduction; existing candidate identities are unchanged.

use crate::{Expr, ExprKind};

/// Remove multiplication/division by exactly one, including below nonlinear calls.
///
/// Assumes ordinary element-wise arithmetic for keyword-free `multiply` and
/// `divide`. Calls with keywords (for example missing-value filtering), custom
/// aliases and time-window operators are not reduced. No non-unit scaling,
/// addition, reassociation or sign cancellation is performed.
///
/// This returns a new expression, not a replacement for the existing semantic
/// fingerprint contract. Consumers opting in must version their dedup policy
/// and apply the same reduction to historical and proposed expressions.
#[must_use]
pub fn reduce_unit_scaling(expression: &Expr) -> Expr {
    let mut reduced = expression.clone();
    match &mut reduced {
        Expr::UnaryCall { arg, kwargs, .. } => {
            **arg = reduce_unit_scaling(arg);
            for value in kwargs.values_mut() {
                *value = reduce_unit_scaling(value);
            }
        }
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            kind,
        } => {
            **left = reduce_unit_scaling(left);
            **right = reduce_unit_scaling(right);
            for value in kwargs.values_mut() {
                *value = reduce_unit_scaling(value);
            }
            if kwargs.is_empty() && *kind == ExprKind::Signal {
                if (op == "divide" || op == "multiply") && is_one(right) && left.kind() == *kind {
                    return *left.clone();
                }
                if op == "multiply" && is_one(left) && right.kind() == *kind {
                    return *right.clone();
                }
            }
        }
        Expr::VariadicCall {
            op,
            args,
            kwargs,
            kind,
        } => {
            for value in args.iter_mut() {
                *value = reduce_unit_scaling(value);
            }
            for value in kwargs.values_mut() {
                *value = reduce_unit_scaling(value);
            }
            if op == "multiply" && kwargs.is_empty() && *kind == ExprKind::Signal && args.len() >= 2
            {
                let other: Vec<_> = args.iter().filter(|arg| !is_one(arg)).collect();
                if let [only] = other.as_slice()
                    && only.kind() == *kind
                {
                    return (*only).clone();
                }
            }
        }
        _ => {}
    }
    reduced
}

fn is_one(expression: &Expr) -> bool {
    matches!(expression, Expr::Scalar { value } if value.to_bits() == 1.0_f64.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{canonical, parse_expression};

    #[test]
    fn nested_unit_scaling_golden_cases() {
        for (input, expected) in [
            ("rank(multiply(close, 1))", "rank(close)"),
            (
                "winsorize(divide(close, 1), std=2)",
                "winsorize(close, std=2)",
            ),
            ("rank(multiply(1, divide(close, 1), 1))", "rank(close)"),
        ] {
            let original = parse_expression(input).unwrap();
            let result = reduce_unit_scaling(&original);
            assert_eq!(
                canonical(&result),
                canonical(&parse_expression(expected).unwrap())
            );
            assert_eq!(result, reduce_unit_scaling(&result));
            assert_eq!(original, parse_expression(input).unwrap());
            assert_eq!(result.kind(), original.kind());
        }
    }

    #[test]
    fn nonunit_scalars_and_window_one_are_preserved() {
        for input in [
            "rank(multiply(close, 2))",
            "rank(multiply(close, -1))",
            "rank(multiply(close, 0))",
            "rank(divide(close, 2))",
            "multiply(close, open, 1)",
            "ts_mean(close, 2)",
        ] {
            let expr = parse_expression(input).unwrap();
            assert_eq!(reduce_unit_scaling(&expr), expr);
        }
        let expr = Expr::BinaryCall {
            op: "ts_decay_linear".into(),
            left: Box::new(Expr::Field {
                name: "close".into(),
            }),
            right: Box::new(Expr::Scalar { value: 1.0 }),
            kwargs: std::collections::BTreeMap::default(),
            kind: ExprKind::Signal,
        };
        assert_eq!(reduce_unit_scaling(&expr), expr);
    }

    #[test]
    fn keywords_and_custom_aliases_are_not_assumed_to_be_identity() {
        let mut expr = parse_expression("divide(close, 1)").unwrap();
        if let Expr::BinaryCall { kwargs, .. } = &mut expr {
            kwargs.insert("filter".into(), Expr::Bool { value: true });
        }
        assert_eq!(reduce_unit_scaling(&expr), expr);
        if let Expr::BinaryCall { op, kwargs, .. } = &mut expr {
            *op = "custom_division".into();
            kwargs.clear();
        }
        assert_eq!(reduce_unit_scaling(&expr), expr);
    }
}
