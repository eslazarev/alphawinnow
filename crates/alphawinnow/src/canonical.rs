use sha2::{Digest, Sha256};

use crate::{analysis::analyze_expression, ast::Expr, operators};

/// Exact normalized formatting, including deterministic commutative ordering.
#[must_use]
pub fn canonical(expression: &Expr) -> String {
    match expression {
        Expr::Field { name } => name.clone(),
        Expr::Scalar { value } => scalar(*value),
        Expr::Group { name } => format!("group(\"{name}\")"),
        Expr::Bool { value } => value.to_string(),
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => call(op, vec![canonical(arg)], kwargs),
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => {
            let mut args = vec![canonical(left), canonical(right)];
            if operators::lookup(op).is_some_and(|spec| spec.commutative) {
                args.sort();
            }
            call(op, args, kwargs)
        }
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => {
            let mut args: Vec<_> = args.iter().map(canonical).collect();
            if operators::lookup(op).is_some_and(|spec| spec.commutative) {
                args.sort();
            }
            call(op, args, kwargs)
        }
    }
}

fn call(
    op: &str,
    mut args: Vec<String>,
    kwargs: &std::collections::BTreeMap<String, Expr>,
) -> String {
    args.extend(
        kwargs
            .iter()
            .map(|(name, value)| format!("{name}={}", canonical(value))),
    );
    format!("{op}({})", args.join(", "))
}

fn scalar(value: f64) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    value.to_string()
}

/// Canonical identity after removing positive scaling only at the final root.
#[must_use]
pub fn semantic_canonical(expression: &Expr) -> String {
    analyze_expression(expression).semantic_canonical
}

#[must_use]
pub fn fingerprint(expression: &Expr) -> String {
    digest(&canonical(expression))
}

#[must_use]
pub fn semantic_fingerprint(expression: &Expr) -> String {
    analyze_expression(expression).semantic_fingerprint
}

#[must_use]
pub fn digest(input: &str) -> String {
    format!("{:x}", Sha256::digest(input.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_expression;

    #[test]
    fn semantic_identity_golden_cases() {
        let equivalent = [
            ("close", "multiply(close, 2)"),
            ("close", "divide(close, 4)"),
            ("add(close, open)", "add(open, close)"),
            ("multiply(close, open)", "multiply(open, 2, close)"),
            ("winsorize(close, std=2.0)", "winsorize(close, std=2)"),
        ];
        for (left, right) in equivalent {
            let left = parse_expression(left).unwrap();
            let right = parse_expression(right).unwrap();
            assert_eq!(semantic_fingerprint(&left), semantic_fingerprint(&right));
        }

        let distinct = [
            ("close", "multiply(close, -2)"),
            ("rank(close)", "rank(multiply(close, 2))"),
            (
                "clip(close, lower=-2, upper=2)",
                "clip(multiply(close, 2), upper=2, lower=-2)",
            ),
            (
                "winsorize(close, std=2)",
                "winsorize(multiply(close, 2), std=2)",
            ),
            (
                "if_else(greater(close, 0), close, open)",
                "if_else(greater(multiply(close, 2), 0), close, open)",
            ),
            (
                "add(rank(close), rank(open))",
                "add(multiply(rank(close), 2), rank(open))",
            ),
            (
                "group_rank(close, group(\"sector\"))",
                "group_rank(close, group(\"country\"))",
            ),
        ];
        for (left, right) in distinct {
            let left = parse_expression(left).unwrap();
            let right = parse_expression(right).unwrap();
            assert_ne!(semantic_fingerprint(&left), semantic_fingerprint(&right));
        }
    }

    #[test]
    fn fingerprint_is_stable_sha256() {
        let expression = parse_expression("rank(close)").unwrap();
        assert_eq!(fingerprint(&expression).len(), 64);
        assert_eq!(fingerprint(&expression), fingerprint(&expression));
    }
}
