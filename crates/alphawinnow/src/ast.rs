use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Static value kind used by grammar validation and transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExprKind {
    Signal,
    Scalar,
    Boolean,
    Group,
}

/// Typed expression tree. Call variants make arity explicit after parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
pub enum Expr {
    Field {
        name: String,
    },
    Scalar {
        value: f64,
    },
    Group {
        name: String,
    },
    Bool {
        value: bool,
    },
    UnaryCall {
        op: String,
        arg: Box<Expr>,
        kwargs: BTreeMap<String, Expr>,
        kind: ExprKind,
    },
    BinaryCall {
        op: String,
        left: Box<Expr>,
        right: Box<Expr>,
        kwargs: BTreeMap<String, Expr>,
        kind: ExprKind,
    },
    VariadicCall {
        op: String,
        args: Vec<Expr>,
        kwargs: BTreeMap<String, Expr>,
        kind: ExprKind,
    },
}

impl Expr {
    #[must_use]
    pub const fn kind(&self) -> ExprKind {
        match self {
            Self::Field { .. } => ExprKind::Signal,
            Self::Scalar { .. } => ExprKind::Scalar,
            Self::Group { .. } => ExprKind::Group,
            Self::Bool { .. } => ExprKind::Boolean,
            Self::UnaryCall { kind, .. }
            | Self::BinaryCall { kind, .. }
            | Self::VariadicCall { kind, .. } => *kind,
        }
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        1 + self
            .children()
            .into_iter()
            .map(Self::node_count)
            .sum::<usize>()
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        1 + self
            .children()
            .into_iter()
            .map(Self::depth)
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn operator_count(&self) -> usize {
        usize::from(self.operator().is_some())
            + self
                .children()
                .into_iter()
                .map(Self::operator_count)
                .sum::<usize>()
    }

    #[must_use]
    pub fn operator(&self) -> Option<&str> {
        match self {
            Self::UnaryCall { op, .. }
            | Self::BinaryCall { op, .. }
            | Self::VariadicCall { op, .. } => Some(op),
            _ => None,
        }
    }

    #[must_use]
    pub fn children(&self) -> Vec<&Self> {
        match self {
            Self::UnaryCall { arg, kwargs, .. } => {
                let mut result = vec![arg.as_ref()];
                result.extend(kwargs.values());
                result
            }
            Self::BinaryCall {
                left,
                right,
                kwargs,
                ..
            } => {
                let mut result = vec![left.as_ref(), right.as_ref()];
                result.extend(kwargs.values());
                result
            }
            Self::VariadicCall { args, kwargs, .. } => {
                let mut result: Vec<_> = args.iter().collect();
                result.extend(kwargs.values());
                result
            }
            _ => Vec::new(),
        }
    }
}
