use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Expr, ExprKind,
    operators::{self, Arity},
};

/// One immutable step from an expression root to an argument or keyword slot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "slot", rename_all = "snake_case")]
pub enum PathSegment {
    UnaryArg,
    BinaryLeft,
    BinaryRight,
    VariadicArg { index: usize },
    Keyword { name: String },
}

/// Stable path to an AST subtree. The empty path identifies the root.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExprPath {
    pub segments: Vec<PathSegment>,
}

impl std::fmt::Display for ExprPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("$")?;
        for segment in &self.segments {
            match segment {
                PathSegment::UnaryArg => formatter.write_str(".arg")?,
                PathSegment::BinaryLeft => formatter.write_str(".left")?,
                PathSegment::BinaryRight => formatter.write_str(".right")?,
                PathSegment::VariadicArg { index } => write!(formatter, ".args[{index}]")?,
                PathSegment::Keyword { name } => write!(formatter, ".kwargs.{name}")?,
            }
        }
        Ok(())
    }
}

/// Compact set of expression kinds accepted by a typed tree slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindSet(u8);

impl KindSet {
    pub const SIGNAL_OR_SCALAR: Self = Self(0b0011);

    #[must_use]
    pub const fn singleton(kind: ExprKind) -> Self {
        Self(match kind {
            ExprKind::Signal => 0b0001,
            ExprKind::Scalar => 0b0010,
            ExprKind::Boolean => 0b0100,
            ExprKind::Group => 0b1000,
        })
    }

    #[must_use]
    pub const fn contains(self, kind: ExprKind) -> bool {
        self.0 & Self::singleton(kind).0 != 0
    }

    #[must_use]
    pub fn from_kinds(kinds: impl IntoIterator<Item = ExprKind>) -> Self {
        kinds
            .into_iter()
            .fold(Self(0), |set, kind| Self(set.0 | Self::singleton(kind).0))
    }

    pub fn iter(self) -> impl Iterator<Item = ExprKind> {
        [
            ExprKind::Signal,
            ExprKind::Scalar,
            ExprKind::Boolean,
            ExprKind::Group,
        ]
        .into_iter()
        .filter(move |kind| self.contains(*kind))
    }
}

/// A reachable subtree and the kinds accepted by its containing slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathEntry {
    pub path: ExprPath,
    pub subtree_kind: ExprKind,
    pub allowed_kinds: KindSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PathError {
    #[error("expression path does not exist: {0}")]
    Missing(ExprPath),
    #[error("path {path} accepts {allowed:?}, got {actual:?}")]
    WrongKind {
        path: ExprPath,
        allowed: KindSet,
        actual: ExprKind,
    },
    #[error("replacement at {path} violates its containing operator: {message}")]
    InvalidContext { path: ExprPath, message: String },
}

/// Enumerate the root, every positional argument, and every keyword slot.
#[must_use]
pub fn enumerate_paths(expression: &Expr) -> Vec<PathEntry> {
    enumerate_paths_with_catalog(expression, operators::builtin_catalog())
}

/// Enumerate typed slots using the same catalog that was used to parse the tree.
#[must_use]
pub fn enumerate_paths_with_catalog(
    expression: &Expr,
    catalog: &operators::Catalog,
) -> Vec<PathEntry> {
    let mut entries = Vec::new();
    let root = ExprPath::default();
    enumerate(
        expression,
        &root,
        KindSet::singleton(expression.kind()),
        &mut entries,
        catalog,
    );
    entries
}

fn enumerate(
    expression: &Expr,
    path: &ExprPath,
    allowed_kinds: KindSet,
    entries: &mut Vec<PathEntry>,
    catalog: &operators::Catalog,
) {
    entries.push(PathEntry {
        path: path.clone(),
        subtree_kind: expression.kind(),
        allowed_kinds,
    });
    match expression {
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => {
            let allowed = positional_kinds(op, 0, arg, catalog);
            enumerate(
                arg,
                &extended(path, PathSegment::UnaryArg),
                allowed,
                entries,
                catalog,
            );
            enumerate_keywords(kwargs, path, entries, catalog);
        }
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => {
            enumerate(
                left,
                &extended(path, PathSegment::BinaryLeft),
                positional_kinds(op, 0, left, catalog),
                entries,
                catalog,
            );
            enumerate(
                right,
                &extended(path, PathSegment::BinaryRight),
                positional_kinds(op, 1, right, catalog),
                entries,
                catalog,
            );
            enumerate_keywords(kwargs, path, entries, catalog);
        }
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => {
            for (index, arg) in args.iter().enumerate() {
                enumerate(
                    arg,
                    &extended(path, PathSegment::VariadicArg { index }),
                    positional_kinds(op, index, arg, catalog),
                    entries,
                    catalog,
                );
            }
            enumerate_keywords(kwargs, path, entries, catalog);
        }
        Expr::Field { .. } | Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => {}
    }
}

fn enumerate_keywords(
    kwargs: &std::collections::BTreeMap<String, Expr>,
    path: &ExprPath,
    entries: &mut Vec<PathEntry>,
    catalog: &operators::Catalog,
) {
    for (name, value) in kwargs {
        enumerate(
            value,
            &extended(path, PathSegment::Keyword { name: name.clone() }),
            KindSet::singleton(ExprKind::Scalar),
            entries,
            catalog,
        );
    }
}

fn positional_kinds(
    op: &str,
    index: usize,
    actual: &Expr,
    catalog: &operators::Catalog,
) -> KindSet {
    let Some(spec) = catalog.lookup(op) else {
        return KindSet::singleton(actual.kind());
    };
    let input = match spec.arity {
        Arity::Variadic { .. } => spec.inputs.first(),
        Arity::Exact { .. } | Arity::Unary | Arity::Binary => spec.inputs.get(index),
    };
    input.map_or_else(
        || KindSet::singleton(actual.kind()),
        |input| KindSet::from_kinds(input.kinds.iter().copied()),
    )
}

fn extended(path: &ExprPath, segment: PathSegment) -> ExprPath {
    let mut segments = path.segments.clone();
    segments.push(segment);
    ExprPath { segments }
}

/// Look up a subtree without modifying its owner.
#[must_use]
pub fn subtree_at<'a>(expression: &'a Expr, path: &ExprPath) -> Option<&'a Expr> {
    let mut current = expression;
    for segment in &path.segments {
        current = match (current, segment) {
            (Expr::UnaryCall { arg, .. }, PathSegment::UnaryArg) => arg,
            (Expr::BinaryCall { left, .. }, PathSegment::BinaryLeft) => left,
            (Expr::BinaryCall { right, .. }, PathSegment::BinaryRight) => right,
            (Expr::VariadicCall { args, .. }, PathSegment::VariadicArg { index }) => {
                args.get(*index)?
            }
            (
                Expr::UnaryCall { kwargs, .. }
                | Expr::BinaryCall { kwargs, .. }
                | Expr::VariadicCall { kwargs, .. },
                PathSegment::Keyword { name },
            ) => kwargs.get(name)?,
            _ => return None,
        };
    }
    Some(current)
}

/// Return all kinds accepted by a path in the current typed tree.
#[must_use]
pub fn allowed_kinds(expression: &Expr, path: &ExprPath) -> Option<KindSet> {
    allowed_kinds_with_catalog(expression, path, operators::builtin_catalog())
}

/// Return accepted kinds using explicit catalog signatures, including custom operators.
#[must_use]
pub fn allowed_kinds_with_catalog(
    expression: &Expr,
    path: &ExprPath,
    catalog: &operators::Catalog,
) -> Option<KindSet> {
    enumerate_paths_with_catalog(expression, catalog)
        .into_iter()
        .find(|entry| entry.path == *path)
        .map(|entry| entry.allowed_kinds)
}

/// Clone a tree with exactly one typed subtree replaced.
///
/// # Errors
/// Returns an error for a missing path or a replacement of the wrong kind.
pub fn replace_subtree(
    expression: &Expr,
    path: &ExprPath,
    replacement: Expr,
) -> Result<Expr, PathError> {
    replace_subtree_with_catalog(expression, path, replacement, operators::builtin_catalog())
}

/// Replace one subtree and validate every containing operator against an explicit catalog.
///
/// # Errors
/// Returns an error for a missing path, incompatible kind, or invalid operator context.
pub fn replace_subtree_with_catalog(
    expression: &Expr,
    path: &ExprPath,
    replacement: Expr,
    catalog: &operators::Catalog,
) -> Result<Expr, PathError> {
    let allowed = allowed_kinds_with_catalog(expression, path, catalog)
        .ok_or_else(|| PathError::Missing(path.clone()))?;
    if !allowed.contains(replacement.kind()) {
        return Err(PathError::WrongKind {
            path: path.clone(),
            allowed,
            actual: replacement.kind(),
        });
    }
    let candidate = replace_at(expression, &path.segments, replacement)
        .ok_or_else(|| PathError::Missing(path.clone()))?;
    validate_structure(&candidate, catalog).map_err(|message| PathError::InvalidContext {
        path: path.clone(),
        message,
    })?;
    Ok(candidate)
}

fn validate_structure(expression: &Expr, catalog: &operators::Catalog) -> Result<(), String> {
    for child in expression.children() {
        validate_structure(child, catalog)?;
    }
    let rebuilt = match expression {
        Expr::UnaryCall {
            op, arg, kwargs, ..
        } => operators::build_call_with_catalog(
            catalog,
            op,
            vec![arg.as_ref().clone()],
            kwargs.clone(),
        )?,
        Expr::BinaryCall {
            op,
            left,
            right,
            kwargs,
            ..
        } => operators::build_call_with_catalog(
            catalog,
            op,
            vec![left.as_ref().clone(), right.as_ref().clone()],
            kwargs.clone(),
        )?,
        Expr::VariadicCall {
            op, args, kwargs, ..
        } => operators::build_call_with_catalog(catalog, op, args.clone(), kwargs.clone())?,
        Expr::Field { .. } | Expr::Scalar { .. } | Expr::Group { .. } | Expr::Bool { .. } => {
            return Ok(());
        }
    };
    if rebuilt.kind() == expression.kind() {
        Ok(())
    } else {
        Err(format!(
            "operator output kind is {:?}, stored node kind is {:?}",
            rebuilt.kind(),
            expression.kind()
        ))
    }
}

fn replace_at(expression: &Expr, segments: &[PathSegment], replacement: Expr) -> Option<Expr> {
    let Some((head, tail)) = segments.split_first() else {
        return Some(replacement);
    };
    let mut result = expression.clone();
    match (&mut result, head) {
        (Expr::UnaryCall { arg, .. }, PathSegment::UnaryArg) => {
            **arg = replace_at(arg, tail, replacement)?;
        }
        (Expr::BinaryCall { left, .. }, PathSegment::BinaryLeft) => {
            **left = replace_at(left, tail, replacement)?;
        }
        (Expr::BinaryCall { right, .. }, PathSegment::BinaryRight) => {
            **right = replace_at(right, tail, replacement)?;
        }
        (Expr::VariadicCall { args, .. }, PathSegment::VariadicArg { index }) => {
            args[*index] = replace_at(args.get(*index)?, tail, replacement)?;
        }
        (
            Expr::UnaryCall { kwargs, .. }
            | Expr::BinaryCall { kwargs, .. }
            | Expr::VariadicCall { kwargs, .. },
            PathSegment::Keyword { name },
        ) => {
            let current = kwargs.get(name)?;
            let value = replace_at(current, tail, replacement)?;
            kwargs.insert(name.clone(), value);
        }
        _ => return None,
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{canonical, parse_expression};

    #[test]
    fn custom_slots_use_catalog_kinds_and_preserve_operator_constraints() {
        let mut catalog = operators::builtin_catalog().clone();
        let mut custom = catalog.lookup("multiply").unwrap().clone();
        custom.name = "custom_product".into();
        catalog.operators.push(custom);
        catalog.validate().unwrap();
        let expression =
            crate::parse_expression_with_catalog("custom_product(close, open)", &catalog).unwrap();
        let path = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 0 }],
        };
        assert_eq!(
            allowed_kinds_with_catalog(&expression, &path, &catalog),
            Some(KindSet::SIGNAL_OR_SCALAR)
        );
        let child =
            replace_subtree_with_catalog(&expression, &path, Expr::Scalar { value: 2.0 }, &catalog)
                .unwrap();
        assert_eq!(canonical(&child), "custom_product(2, open)");
        let second = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 1 }],
        };
        // Catalog-aware editing must not bypass the signal-presence constraint.
        assert!(matches!(
            replace_subtree_with_catalog(&child, &second, Expr::Scalar { value: 3.0 }, &catalog),
            Err(PathError::InvalidContext { .. })
        ));
        assert!(
            replace_subtree(
                &expression,
                &path,
                Expr::Field {
                    name: "high".into()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_catalog_overrides_embedded_signature_and_enforces_window_domain() {
        let mut catalog = operators::builtin_catalog().clone();
        let mean = catalog
            .operators
            .iter_mut()
            .find(|op| op.name == "ts_mean")
            .unwrap();
        mean.inputs[0].kinds.push(ExprKind::Scalar);
        mean.inputs[1].domain = operators::ValueDomain::WindowSet {
            values: vec![5, 20],
        };
        catalog.validate().unwrap();
        let expression =
            crate::parse_expression_with_catalog("ts_mean(close, 20)", &catalog).unwrap();
        let left = ExprPath {
            segments: vec![PathSegment::BinaryLeft],
        };
        let right = ExprPath {
            segments: vec![PathSegment::BinaryRight],
        };
        assert_eq!(
            allowed_kinds_with_catalog(&expression, &left, &catalog),
            Some(KindSet::SIGNAL_OR_SCALAR)
        );
        assert!(
            replace_subtree_with_catalog(&expression, &left, Expr::Scalar { value: 2.0 }, &catalog)
                .is_ok()
        );
        assert!(
            replace_subtree_with_catalog(
                &expression,
                &right,
                Expr::Scalar { value: 6.0 },
                &catalog
            )
            .is_err()
        );
        assert!(
            replace_subtree_with_catalog(
                &expression,
                &right,
                Expr::Scalar { value: 5.0 },
                &catalog
            )
            .is_ok()
        );
    }

    #[test]
    fn builtin_tree_wrappers_match_explicit_catalog() {
        let expression = parse_expression("winsorize(ts_mean(close, 20), std=2)").unwrap();
        let catalog = operators::builtin_catalog();
        assert_eq!(
            enumerate_paths(&expression),
            enumerate_paths_with_catalog(&expression, catalog)
        );
        for entry in enumerate_paths(&expression) {
            let replacement = subtree_at(&expression, &entry.path).unwrap().clone();
            assert_eq!(
                allowed_kinds(&expression, &entry.path),
                allowed_kinds_with_catalog(&expression, &entry.path, catalog)
            );
            assert_eq!(
                replace_subtree(&expression, &entry.path, replacement.clone()),
                replace_subtree_with_catalog(&expression, &entry.path, replacement, catalog)
            );
        }
    }

    #[test]
    fn enumerates_arguments_keywords_and_allowed_kinds() {
        let expression = parse_expression("winsorize(ts_mean(close, 20), std=2)").unwrap();
        let entries = enumerate_paths(&expression);
        assert_eq!(entries.len(), 5);
        assert!(entries.iter().any(|entry| {
            entry.path.to_string() == "$.arg.right"
                && entry.allowed_kinds == KindSet::singleton(ExprKind::Scalar)
        }));
        assert!(entries.iter().any(|entry| {
            entry.path.to_string() == "$.kwargs.std"
                && entry.allowed_kinds == KindSet::singleton(ExprKind::Scalar)
        }));
    }

    #[test]
    fn replaces_only_a_kind_compatible_slot() {
        let expression = parse_expression("ts_mean(close, 20)").unwrap();
        let path = ExprPath {
            segments: vec![PathSegment::BinaryLeft],
        };
        let replaced = replace_subtree(
            &expression,
            &path,
            Expr::Field {
                name: "open".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(canonical(&replaced), "ts_mean(open, 20)");
        assert!(matches!(
            replace_subtree(&expression, &path, Expr::Scalar { value: 2.0 }),
            Err(PathError::WrongKind { .. })
        ));
    }

    #[test]
    fn multiply_slots_accept_signal_or_scalar_while_preserving_a_signal() {
        let expression = parse_expression("multiply(close, open)").unwrap();
        let first = ExprPath {
            segments: vec![PathSegment::VariadicArg { index: 0 }],
        };
        let first_entry = enumerate_paths(&expression)
            .into_iter()
            .find(|entry| entry.path == first)
            .unwrap();
        assert_eq!(first_entry.allowed_kinds, KindSet::SIGNAL_OR_SCALAR);

        let replaced = replace_subtree(&expression, &first, Expr::Scalar { value: 2.0 }).unwrap();
        assert_eq!(canonical(&replaced), "multiply(2, open)");

        let last_signal = parse_expression("multiply(close, 2)").unwrap();
        assert!(matches!(
            replace_subtree(&last_signal, &first, Expr::Scalar { value: 3.0 }),
            Err(PathError::InvalidContext { .. })
        ));
    }
}
