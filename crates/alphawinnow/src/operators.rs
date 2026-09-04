use std::{collections::BTreeMap, sync::OnceLock};

use serde::{Deserialize, Serialize};

use crate::{
    ast::{Expr, ExprKind},
    canonical::digest,
};

pub const CATALOG_SCHEMA: u32 = 1;
const BUILTIN_CATALOG_JSON: &str = include_str!("../catalog/public-v1.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Arity {
    Unary,
    Binary,
    Exact { count: usize },
    Variadic { min: usize, max: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValueDomain {
    Any,
    Window { min: u16, max: u16 },
    WindowSet { values: Vec<u16> },
    NonZeroScalar,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentSpec {
    pub kinds: Vec<ExprKind>,
    #[serde(default = "any_domain")]
    pub domain: ValueDomain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeywordSpec {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    #[serde(default = "required_keyword")]
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KeywordConstraint {
    LessThan { left: String, right: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorSpec {
    pub name: String,
    pub arity: Arity,
    pub inputs: Vec<ArgumentSpec>,
    #[serde(default)]
    pub required_input_kinds: Vec<ExprKind>,
    pub output: ExprKind,
    #[serde(default)]
    pub commutative: bool,
    #[serde(default)]
    pub associative: bool,
    #[serde(default)]
    pub keywords: Vec<KeywordSpec>,
    #[serde(default)]
    pub keyword_constraints: Vec<KeywordConstraint>,
    pub generation_weight: u16,
    #[serde(default)]
    pub parse_only_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldSpec {
    pub name: String,
    pub kind: ExprKind,
    pub family: String,
    pub allowed_roles: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupSpec {
    pub name: String,
    pub kind: ExprKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarDomain {
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub schema: u32,
    pub name: String,
    pub operators: Vec<OperatorSpec>,
    pub fields: Vec<FieldSpec>,
    pub groups: Vec<GroupSpec>,
    pub scalar_domain: ScalarDomain,
}

impl Catalog {
    /// Parse and validate a portable public catalog.
    ///
    /// # Errors
    /// Returns a JSON or catalog-invariant diagnostic.
    pub fn from_json(input: &str) -> Result<Self, String> {
        let catalog: Self = serde_json::from_str(input).map_err(|error| error.to_string())?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Validate names, kinds, domains, weights, and uniqueness.
    ///
    /// # Errors
    /// Returns the first violated catalog invariant.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CATALOG_SCHEMA {
            return Err(format!("unsupported catalog schema {}", self.schema));
        }
        if self.operators.is_empty() || self.fields.is_empty() || self.groups.is_empty() {
            return Err("catalog operators, fields, and groups must be non-empty".to_owned());
        }
        if !valid_domain(
            self.scalar_domain.min,
            self.scalar_domain.max,
            self.scalar_domain.step,
        ) {
            return Err("invalid scalar domain".to_owned());
        }
        unique_names(
            self.operators.iter().map(|item| item.name.as_str()),
            "operator",
        )?;
        unique_names(self.fields.iter().map(|item| item.name.as_str()), "field")?;
        unique_names(self.groups.iter().map(|item| item.name.as_str()), "group")?;
        for field in &self.fields {
            if field.kind != ExprKind::Signal
                || field.family.is_empty()
                || field.allowed_roles.is_empty()
            {
                return Err(format!("invalid public field `{}`", field.name));
            }
        }
        for group in &self.groups {
            if group.kind != ExprKind::Group {
                return Err(format!("group `{}` must have group kind", group.name));
            }
        }
        for operator in &self.operators {
            validate_operator_spec(operator)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&OperatorSpec> {
        self.operators.iter().find(|spec| spec.name == name)
    }

    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldSpec> {
        self.fields.iter().find(|field| field.name == name)
    }

    #[must_use]
    pub fn group(&self, name: &str) -> Option<&GroupSpec> {
        self.groups.iter().find(|group| group.name == name)
    }

    /// SHA-256 of normalized catalog data consumed by the tool.
    #[must_use]
    pub fn checksum(&self) -> String {
        digest(&serde_json::to_string(self).unwrap_or_default())
    }
}

const fn any_domain() -> ValueDomain {
    ValueDomain::Any
}

const fn required_keyword() -> bool {
    true
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().enumerate().all(|(index, ch)| {
            ch == '_' || ch.is_ascii_alphanumeric() && (index > 0 || !ch.is_ascii_digit())
        })
}

fn unique_names<'a>(names: impl Iterator<Item = &'a str>, label: &str) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !valid_name(name) || !seen.insert(name) {
            return Err(format!("invalid or duplicate {label} name `{name}`"));
        }
    }
    Ok(())
}

fn valid_domain(min: f64, max: f64, step: f64) -> bool {
    min.is_finite() && max.is_finite() && step.is_finite() && min < max && step > 0.0
}

fn validate_operator_spec(spec: &OperatorSpec) -> Result<(), String> {
    if !valid_name(&spec.name) {
        return Err(format!("invalid operator name `{}`", spec.name));
    }
    let valid_input_count = match spec.arity {
        Arity::Unary => spec.inputs.len() == 1,
        Arity::Binary => spec.inputs.len() == 2,
        Arity::Exact { count } => count > 0 && spec.inputs.len() == count,
        Arity::Variadic { min, max } => min > 0 && min <= max && spec.inputs.len() == 1,
    };
    if !valid_input_count || spec.inputs.iter().any(|input| input.kinds.is_empty()) {
        return Err(format!("invalid input signature for `{}`", spec.name));
    }
    for input in &spec.inputs {
        match &input.domain {
            ValueDomain::Window { min, max } if *min == 0 || min > max => {
                return Err(format!("invalid window domain for `{}`", spec.name));
            }
            ValueDomain::WindowSet { values }
                if values.is_empty()
                    || values.contains(&0)
                    || values.windows(2).any(|pair| pair[0] >= pair[1]) =>
            {
                return Err(format!("invalid window set for `{}`", spec.name));
            }
            _ => {}
        }
    }
    if spec.generation_weight == 0 && spec.parse_only_reason.is_none() {
        return Err(format!(
            "parse-only operator `{}` requires a documented reason",
            spec.name
        ));
    }
    for keyword in &spec.keywords {
        if !valid_name(&keyword.name) || !valid_domain(keyword.min, keyword.max, keyword.step) {
            return Err(format!("invalid keyword domain for `{}`", spec.name));
        }
    }
    unique_names(
        spec.keywords.iter().map(|item| item.name.as_str()),
        "keyword",
    )?;
    for constraint in &spec.keyword_constraints {
        let KeywordConstraint::LessThan { left, right } = constraint;
        if !spec.keywords.iter().any(|item| item.name == *left)
            || !spec.keywords.iter().any(|item| item.name == *right)
        {
            return Err(format!("invalid keyword constraint for `{}`", spec.name));
        }
    }
    Ok(())
}

/// The embedded, synthetic, license-safe public catalog.
///
/// # Panics
/// Panics only when the catalog embedded at compile time is invalid, which is
/// guarded by tests and prevents shipping an inconsistent binary.
#[must_use]
pub fn builtin_catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        Catalog::from_json(BUILTIN_CATALOG_JSON)
            .expect("the embedded public catalog must remain valid")
    })
}

#[must_use]
pub fn lookup(name: &str) -> Option<&'static OperatorSpec> {
    builtin_catalog().lookup(name)
}

/// Construct a typed call using the embedded public catalog.
///
/// # Errors
/// Returns an error for unknown operators, invalid arity, kinds, or parameters.
pub fn build_call(
    name: &str,
    args: Vec<Expr>,
    kwargs: BTreeMap<String, Expr>,
) -> Result<Expr, String> {
    build_call_with_catalog(builtin_catalog(), name, args, kwargs)
}

/// Construct a typed call using an explicitly supplied catalog.
///
/// # Errors
/// Returns an error for unknown operators, invalid arity, kinds, or parameters.
///
/// # Panics
/// A programmatically constructed catalog must pass [`Catalog::validate`]
/// before use; violating that contract can leave missing signature slots.
pub fn build_call_with_catalog(
    catalog: &Catalog,
    name: &str,
    args: Vec<Expr>,
    kwargs: BTreeMap<String, Expr>,
) -> Result<Expr, String> {
    let spec = catalog
        .lookup(name)
        .ok_or_else(|| format!("unknown operator `{name}`"))?;
    validate_arity(spec, args.len())?;
    validate_args(spec, &args)?;
    validate_kwargs(spec, &kwargs)?;

    Ok(match spec.arity {
        Arity::Unary => {
            let mut args = args.into_iter();
            let Some(arg) = args.next() else {
                return Err(format!("operator `{name}` requires one argument"));
            };
            Expr::UnaryCall {
                op: name.to_owned(),
                arg: Box::new(arg),
                kwargs,
                kind: spec.output,
            }
        }
        Arity::Binary => {
            let mut args = args.into_iter();
            let Some(left) = args.next() else {
                return Err(format!("operator `{name}` requires two arguments"));
            };
            let Some(right) = args.next() else {
                return Err(format!("operator `{name}` requires two arguments"));
            };
            Expr::BinaryCall {
                op: name.to_owned(),
                left: Box::new(left),
                right: Box::new(right),
                kwargs,
                kind: spec.output,
            }
        }
        Arity::Exact { .. } | Arity::Variadic { .. } => Expr::VariadicCall {
            op: name.to_owned(),
            args,
            kwargs,
            kind: spec.output,
        },
    })
}

fn validate_arity(spec: &OperatorSpec, actual: usize) -> Result<(), String> {
    let valid = match spec.arity {
        Arity::Unary => actual == 1,
        Arity::Binary => actual == 2,
        Arity::Exact { count } => actual == count,
        Arity::Variadic { min, max } => (min..=max).contains(&actual),
    };
    valid
        .then_some(())
        .ok_or_else(|| format!("operator `{}` received invalid arity {actual}", spec.name))
}

fn validate_args(spec: &OperatorSpec, args: &[Expr]) -> Result<(), String> {
    for (index, arg) in args.iter().enumerate() {
        let input = match spec.arity {
            Arity::Variadic { .. } => &spec.inputs[0],
            _ => &spec.inputs[index],
        };
        if !input.kinds.contains(&arg.kind()) {
            return Err(format!(
                "operator `{}` argument {} does not accept {:?}",
                spec.name,
                index + 1,
                arg.kind()
            ));
        }
        validate_value_domain(&spec.name, index, arg, &input.domain)?;
    }
    for required in &spec.required_input_kinds {
        if !args.iter().any(|arg| arg.kind() == *required) {
            return Err(format!(
                "operator `{}` requires at least one {required:?} argument",
                spec.name
            ));
        }
    }
    Ok(())
}

fn validate_value_domain(
    operator: &str,
    index: usize,
    value: &Expr,
    domain: &ValueDomain,
) -> Result<(), String> {
    match domain {
        ValueDomain::Any => Ok(()),
        ValueDomain::Window { min, max } => match value {
            Expr::Scalar { value }
                if value.fract() == 0.0 && (f64::from(*min)..=f64::from(*max)).contains(value) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "operator `{operator}` argument {} must be an integer window in {min}..={max}",
                index + 1
            )),
        },
        ValueDomain::WindowSet { values } => match value {
            Expr::Scalar { value }
                if value.fract() == 0.0
                    && values
                        .iter()
                        .any(|window| f64::from(*window).to_bits() == value.to_bits()) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "operator `{operator}` argument {} must be an allowed window",
                index + 1
            )),
        },
        ValueDomain::NonZeroScalar => match value {
            Expr::Scalar { value } if *value != 0.0 => Ok(()),
            _ => Err(format!(
                "operator `{operator}` argument {} must be a non-zero scalar",
                index + 1
            )),
        },
    }
}

fn validate_kwargs(spec: &OperatorSpec, kwargs: &BTreeMap<String, Expr>) -> Result<(), String> {
    for keyword in &spec.keywords {
        if keyword.required && !kwargs.contains_key(&keyword.name) {
            return Err(format!(
                "operator `{}` requires keyword `{}`",
                spec.name, keyword.name
            ));
        }
    }
    if kwargs.len() > spec.keywords.len()
        || kwargs
            .keys()
            .any(|name| !spec.keywords.iter().any(|item| item.name == *name))
    {
        return Err(format!(
            "operator `{}` received unsupported keyword arguments",
            spec.name
        ));
    }
    for (name, value) in kwargs {
        let keyword = spec
            .keywords
            .iter()
            .find(|item| item.name == *name)
            .expect("keyword membership checked");
        let Expr::Scalar { value } = value else {
            return Err(format!("keyword `{name}` must be a scalar literal"));
        };
        if !(keyword.min..=keyword.max).contains(value) {
            return Err(format!(
                "keyword `{name}` must be in {}..={}",
                keyword.min, keyword.max
            ));
        }
    }
    for constraint in &spec.keyword_constraints {
        match constraint {
            KeywordConstraint::LessThan { left, right } => {
                let (Expr::Scalar { value: left_value }, Expr::Scalar { value: right_value }) =
                    (&kwargs[left], &kwargs[right])
                else {
                    unreachable!("keyword literals validated")
                };
                if left_value >= right_value {
                    return Err(format!(
                        "operator `{}` requires keyword constraint {left} < {right}",
                        spec.name
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_is_valid_and_stably_checksummed() {
        let catalog = builtin_catalog();
        catalog.validate().unwrap();
        assert_eq!(catalog.checksum(), catalog.checksum());
        assert_eq!(catalog.schema, CATALOG_SCHEMA);
    }

    #[test]
    fn synthetic_field_is_added_as_data() {
        let mut catalog = builtin_catalog().clone();
        catalog.fields.push(FieldSpec {
            name: "synthetic_quality".to_owned(),
            kind: ExprKind::Signal,
            family: "synthetic".to_owned(),
            allowed_roles: vec!["signal_input".to_owned()],
        });
        catalog.validate().unwrap();
        assert!(catalog.field("synthetic_quality").is_some());
        crate::parse_expression_with_catalog("rank(synthetic_quality)", &catalog).unwrap();
    }
}
