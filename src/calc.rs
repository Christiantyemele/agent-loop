//! # Safe arithmetic for the `calc` tool.
//!
//! A tiny recursive-descent evaluator over a closed grammar. It never evaluates
//! arbitrary code — it parses numbers, operators, parentheses, named functions
//! and constants, and computes the result. This is what the `calc` tool uses so
//! the agent can do arithmetic without executing anything dangerous.

use std::f64::consts::{E, PI, TAU};

/// Evaluate a numeric expression string.
///
/// Returns an error (with a human-readable message) on any parse or domain
/// error (e.g. divide by zero, unknown function, unbalanced parens).
pub fn eval(input: &str) -> Result<f64, String> {
    let mut parser = Parser::new(input);
    let value = parser.parse()?;
    parser.skip_ws();
    if let Some(ch) = parser.peek_char() {
        return Err(format!("unexpected character `{ch}` at position {}", parser.pos));
    }
    Ok(value)
}

// ---------------------------------------------------------------------------
// Tokenizer / parser
// ---------------------------------------------------------------------------

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn eat(&mut self, expected: char) -> Result<(), String> {
        self.skip_ws();
        if self.peek_char() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected `{expected}` at position {}", self.pos))
        }
    }

    fn parse(&mut self) -> Result<f64, String> {
        self.parse_addsub()
    }

    // expression := term (('+' | '-') term)*
    fn parse_addsub(&mut self) -> Result<f64, String> {
        let mut left = self.parse_muldiv()?;
        loop {
            self.skip_ws();
            match self.peek_char() {
                Some('+') => {
                    self.pos += 1;
                    let right = self.parse_muldiv()?;
                    left += right;
                }
                Some('-') => {
                    self.pos += 1;
                    let right = self.parse_muldiv()?;
                    left -= right;
                }
                _ => return Ok(left),
            }
        }
    }

    // term := unary (('*' | '/') unary)*
    fn parse_muldiv(&mut self) -> Result<f64, String> {
        let mut left = self.parse_power()?;
        loop {
            self.skip_ws();
            match self.peek_char() {
                Some('*') => {
                    self.pos += 1;
                    let right = self.parse_power()?;
                    left *= right;
                }
                Some('/') => {
                    self.pos += 1;
                    let right = self.parse_power()?;
                    if right == 0.0 {
                        return Err("division by zero".into());
                    }
                    left /= right;
                }
                _ => return Ok(left),
            }
        }
    }

    // power := unary ('^' unary)?  (right-associative)
    fn parse_power(&mut self) -> Result<f64, String> {
        let base = self.parse_unary()?;
        self.skip_ws();
        if self.peek_char() == Some('^') {
            self.pos += 1;
            let exponent = self.parse_unary()?;
            return Ok(base.powf(exponent));
        }
        Ok(base)
    }

    // unary := ('-'|'+')* unary | atom
    fn parse_unary(&mut self) -> Result<f64, String> {
        self.skip_ws();
        match self.peek_char() {
            Some('-') => {
                self.pos += 1;
                Ok(-self.parse_unary()?)
            }
            Some('+') => {
                self.pos += 1;
                self.parse_unary()
            }
            _ => self.parse_atom(),
        }
    }

    // atom := number | constant | func '(' expr ')' | '(' expr ')'
    fn parse_atom(&mut self) -> Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        let is_number = self
            .peek_char()
            .map(|c| c.is_ascii_digit() || c == '.')
            .unwrap_or(false);
        if is_number {
            // consume digits + optional fraction + optional exponent
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() || c == '.' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            // exponent like 1e5 or 1.5e-3
            if self.peek_char() == Some('e') || self.peek_char() == Some('E') {
                let save = self.pos;
                self.pos += 1;
                if self.peek_char() == Some('-') || self.peek_char() == Some('+') {
                    self.pos += 1;
                }
                let exp_start = self.pos;
                while self.peek_char().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                    self.pos += 1;
                }
                if self.pos == exp_start {
                    self.pos = save;
                }
            }
            let text: String = self.chars[start..self.pos].iter().collect();
            return text
                .parse::<f64>()
                .map_err(|_| format!("invalid number `{text}`"));
        }

        // function name or constant — read alphabetic token
        if self
            .peek_char()
            .map(|c| c.is_ascii_alphabetic() || c == '_')
            .unwrap_or(false)
        {
            while self
                .peek_char()
                .map(|c| c.is_ascii_alphanumeric() || c == '_')
                .unwrap_or(false)
            {
                self.pos += 1;
            }
            let name: String = self.chars[start..self.pos].iter().collect();
            let lower = name.to_lowercase();
            match lower.as_str() {
                "pi" => return Ok(PI),
                "e" => return Ok(E),
                "tau" => return Ok(TAU),
                _ => {
                    self.eat('(')?;
                    let arg = self.parse_addsub()?;
                    self.eat(')')?;
                    let result = apply_function(&lower, arg)?;
                    return Ok(result);
                }
            }
        }

        if self.peek_char() == Some('(') {
            self.pos += 1;
            let inner = self.parse_addsub()?;
            self.eat(')')?;
            return Ok(inner);
        }

        Err(format!(
            "unexpected token at position {}",
            self.pos
        ))
    }
}

/// Apply a single-argument named function.
fn apply_function(name: &str, x: f64) -> Result<f64, String> {
    match name {
        "sqrt" | "root" => {
            if x < 0.0 {
                Err(format!("sqrt of negative {x}"))
            } else {
                Ok(x.sqrt())
            }
        }
        "abs" => Ok(x.abs()),
        "sin" => Ok(x.sin()),
        "cos" => Ok(x.cos()),
        "tan" => Ok(x.tan()),
        "asin" => Ok(x.asin()),
        "acos" => Ok(x.acos()),
        "atan" => Ok(x.atan()),
        "exp" => Ok(x.exp()),
        "ln" | "log" => {
            if x <= 0.0 {
                Err(format!("log of non-positive {x}"))
            } else {
                Ok(x.ln())
            }
        }
        "log10" => {
            if x <= 0.0 {
                Err(format!("log10 of non-positive {x}"))
            } else {
                Ok(x.log10())
            }
        }
        "floor" => Ok(x.floor()),
        "ceil" => Ok(x.ceil()),
        "round" => Ok(x.round()),
        _ => Err(format!("unknown function `{name}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn basic_arithmetic() {
        approx(eval("2 + 3 * 4").unwrap(), 14.0);
        approx(eval("(2 + 3) * 4").unwrap(), 20.0);
        approx(eval("10 / 4").unwrap(), 2.5);
        approx(eval("2 ^ 10").unwrap(), 1024.0);
        approx(eval("-3 + 5").unwrap(), 2.0);
        approx(eval("1.5 + 2.25").unwrap(), 3.75);
    }

    #[test]
    fn functions_and_constants() {
        approx(eval("sqrt(9)").unwrap(), 3.0);
        approx(eval("abs(-7)").unwrap(), 7.0);
        approx(eval("round(2.6)").unwrap(), 3.0);
        approx(eval("floor(2.9)").unwrap(), 2.0);
        approx(eval("ceil(2.1)").unwrap(), 3.0);
        approx(eval("exp(0)").unwrap(), 1.0);
        approx(eval("ln(e)").unwrap(), 1.0);
        approx(eval("pi").unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn errors() {
        assert!(eval("1 / 0").is_err());
        assert!(eval("sqrt(-1)").is_err());
        assert!(eval("bogus(1)").is_err());
        assert!(eval("(1 + 2").is_err());
        assert!(eval("3 # 4").is_err());
    }
}
