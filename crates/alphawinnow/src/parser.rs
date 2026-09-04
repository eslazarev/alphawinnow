use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    ast::{Expr, ExprKind},
    operators::{self, Catalog},
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExpressionError {
    #[error("expression syntax error at byte {offset}: {message}")]
    Syntax { offset: usize, message: String },
    #[error("expression validation error: {0}")]
    Validation(String),
    #[error("root expression must produce a signal, got {0:?}")]
    NonSignalRoot(ExprKind),
}

/// Parse and fully validate an expression before it can enter a candidate pool.
///
/// # Errors
/// Returns a syntax, operator-validation, or non-signal-root error.
pub fn parse_expression(input: &str) -> Result<Expr, ExpressionError> {
    parse_expression_with_catalog(input, operators::builtin_catalog())
}

/// Parse and validate an expression against an explicit public catalog.
///
/// # Errors
/// Returns a syntax, catalog-validation, or non-signal-root error.
pub fn parse_expression_with_catalog(
    input: &str,
    catalog: &Catalog,
) -> Result<Expr, ExpressionError> {
    let mut parser = Parser {
        input,
        offset: 0,
        catalog,
    };
    let expression = parser.parse_expr()?;
    parser.skip_space();
    if !parser.is_eof() {
        return Err(parser.syntax("unexpected trailing input"));
    }
    if expression.kind() != ExprKind::Signal {
        return Err(ExpressionError::NonSignalRoot(expression.kind()));
    }
    Ok(expression)
}

struct Parser<'a> {
    input: &'a str,
    offset: usize,
    catalog: &'a Catalog,
}

impl Parser<'_> {
    fn parse_expr(&mut self) -> Result<Expr, ExpressionError> {
        self.skip_space();
        let Some(ch) = self.peek() else {
            return Err(self.syntax("expected expression"));
        };
        if ch.is_ascii_digit() || matches!(ch, '-' | '+') {
            return self.parse_number();
        }
        if is_ident_start(ch) {
            return self.parse_identifier_expr();
        }
        Err(self.syntax("expected identifier or scalar literal"))
    }

    fn parse_identifier_expr(&mut self) -> Result<Expr, ExpressionError> {
        let name = self.parse_identifier()?;
        self.skip_space();
        if self.peek() != Some('(') {
            return match name.as_str() {
                "true" => Ok(Expr::Bool { value: true }),
                "false" => Ok(Expr::Bool { value: false }),
                _ if self.catalog.field(&name).is_some() => Ok(Expr::Field { name }),
                _ => Err(ExpressionError::Validation(format!(
                    "unknown field `{name}` in active catalog"
                ))),
            };
        }
        self.bump();
        if name == "group" {
            return self.parse_group();
        }
        let (args, kwargs) = self.parse_call_args()?;
        operators::build_call_with_catalog(self.catalog, &name, args, kwargs)
            .map_err(ExpressionError::Validation)
    }

    fn parse_group(&mut self) -> Result<Expr, ExpressionError> {
        self.skip_space();
        if self.bump() != Some('"') {
            return Err(self.syntax("group literal must be a quoted string"));
        }
        let start = self.offset;
        while let Some(ch) = self.peek() {
            if ch == '"' {
                let name = self.input[start..self.offset].to_owned();
                self.bump();
                self.skip_space();
                if self.bump() != Some(')') {
                    return Err(self.syntax("expected `)` after group literal"));
                }
                if name.is_empty()
                    || !name
                        .chars()
                        .all(|value| value.is_ascii_alphanumeric() || value == '_')
                {
                    return Err(self.syntax("group name must contain only letters, digits, or `_`"));
                }
                if self.catalog.group(&name).is_none() {
                    return Err(ExpressionError::Validation(format!(
                        "unknown group `{name}` in active catalog"
                    )));
                }
                return Ok(Expr::Group { name });
            }
            if ch == '\\' {
                return Err(self.syntax("escapes are not supported in group literals"));
            }
            self.bump();
        }
        Err(self.syntax("unterminated group literal"))
    }

    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, BTreeMap<String, Expr>), ExpressionError> {
        let mut args = Vec::new();
        let mut kwargs = BTreeMap::new();
        let mut saw_keyword = false;
        loop {
            self.skip_space();
            if self.peek() == Some(')') {
                self.bump();
                return Ok((args, kwargs));
            }
            let checkpoint = self.offset;
            let keyword = if self.peek().is_some_and(is_ident_start) {
                let candidate = self.parse_identifier()?;
                self.skip_space();
                if self.peek() == Some('=') {
                    self.bump();
                    Some(candidate)
                } else {
                    self.offset = checkpoint;
                    None
                }
            } else {
                None
            };
            let value = self.parse_expr()?;
            if let Some(keyword) = keyword {
                saw_keyword = true;
                if kwargs.insert(keyword.clone(), value).is_some() {
                    return Err(self.syntax(&format!("duplicate keyword `{keyword}`")));
                }
            } else if saw_keyword {
                return Err(self.syntax("positional argument cannot follow a keyword argument"));
            } else {
                args.push(value);
            }
            self.skip_space();
            match self.bump() {
                Some(',') => {}
                Some(')') => return Ok((args, kwargs)),
                _ => return Err(self.syntax("expected `,` or `)`")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Expr, ExpressionError> {
        let start = self.offset;
        if matches!(self.peek(), Some('-' | '+')) {
            self.bump();
        }
        let mut digits = 0;
        while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
            self.bump();
            digits += 1;
        }
        if self.peek() == Some('.') {
            self.bump();
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump();
                digits += 1;
            }
        }
        if digits == 0 {
            return Err(self.syntax("invalid scalar literal"));
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.bump();
            if matches!(self.peek(), Some('-' | '+')) {
                self.bump();
            }
            let exponent_start = self.offset;
            while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
                self.bump();
            }
            if exponent_start == self.offset {
                return Err(self.syntax("invalid scalar exponent"));
            }
        }
        let raw = &self.input[start..self.offset];
        let mut value: f64 = raw
            .parse()
            .map_err(|_| self.syntax("invalid scalar literal"))?;
        if !value.is_finite() {
            return Err(self.syntax("scalar literal must be finite"));
        }
        if value == 0.0 {
            value = 0.0;
        }
        Ok(Expr::Scalar { value })
    }

    fn parse_identifier(&mut self) -> Result<String, ExpressionError> {
        let start = self.offset;
        if !self.peek().is_some_and(is_ident_start) {
            return Err(self.syntax("expected identifier"));
        }
        self.bump();
        while self.peek().is_some_and(is_ident_continue) {
            self.bump();
        }
        Ok(self.input[start..self.offset].to_owned())
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.offset += ch.len_utf8();
        Some(ch)
    }

    const fn is_eof(&self) -> bool {
        self.offset == self.input.len()
    }

    fn syntax(&self, message: &str) -> ExpressionError {
        ExpressionError::Syntax {
            offset: self.offset,
            message: message.to_owned(),
        }
    }
}

const fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

const fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

#[cfg(test)]
mod tests {
    use crate::{canonical, parse_expression};

    #[test]
    fn parses_documented_operator_families() {
        let cases = [
            "ts_rank(close, 20)",
            "rank(subtract(close, open))",
            "if_else(greater(volume, 100), close, open)",
            "group_rank(close, group(\"sector\"))",
            "winsorize(close, std=3)",
        ];
        for input in cases {
            parse_expression(input).unwrap_or_else(|error| panic!("{input}: {error}"));
        }
    }

    #[test]
    fn normalizes_numeric_spelling() {
        let left = parse_expression("ts_mean(close, 020.0)").unwrap();
        let right = parse_expression("ts_mean(close, 2e1)").unwrap();
        assert_eq!(canonical(&left), canonical(&right));
    }

    #[test]
    fn rejects_invalid_grammar_parameters() {
        for input in [
            "mystery(close)",
            "ts_rank(close, 1)",
            "ts_rank(close, 2.5)",
            "winsorize(close, std=close)",
            "winsorize(close, std=100)",
            "clip(close, other=1)",
            "clip(close, lower=2, upper=1)",
        ] {
            assert!(parse_expression(input).is_err(), "accepted {input}");
        }
    }

    #[test]
    fn rejects_names_absent_from_the_active_catalog() {
        assert!(parse_expression("rank(private_field)").is_err());
        assert!(parse_expression("group_rank(close, group(\"private_group\"))").is_err());
    }
}
