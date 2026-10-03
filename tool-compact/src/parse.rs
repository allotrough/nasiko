//! Compact definitions → JSON Schema tools: the inverse of the encoder, and the proof that
//! nothing was lost.

use serde_json::Value;

use crate::encode::CompactTools;
use crate::error::Error;
use crate::schema::{Field, Ty, is_ident_char, is_ident_start};
use crate::types::ToolDef;

/// Rebuild the tool definitions from their compact form.
///
/// For [`crate::Level::Lossless`] output this equals the encoder's input, up to JSON key order
/// and an empty `required` array.
pub fn decode_tools(compact: &CompactTools) -> crate::Result<Vec<ToolDef>> {
    compact.definitions().split('\n').map(parse_tool).collect()
}

fn parse_tool(line: &str) -> crate::Result<ToolDef> {
    let mut p = Parser {
        s: line.chars().collect(),
        i: 0,
    };
    let name = p.ident()?;
    p.expect('(')?;
    let fields = p.fields(')')?;
    p.expect(')')?;
    let description = if p.eat_str(" - ") {
        Some(if p.peek() == Some('"') {
            match p.json()? {
                Value::String(s) => s,
                _ => return Err(p.err("description")),
            }
        } else {
            p.rest()
        })
    } else {
        None
    };
    if p.peek().is_some() {
        return Err(p.err("trailing text"));
    }
    Ok(ToolDef {
        name,
        description,
        parameters: Some(Ty::Object(fields).to_json()),
    })
}

struct Parser {
    s: Vec<char>,
    i: usize,
}

impl Parser {
    fn err(&self, what: &str) -> Error {
        Error::BadDefinitions(format!("{what} at column {}", self.i))
    }

    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, c: char) -> crate::Result<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected `{c}`")))
        }
    }

    fn eat_str(&mut self, lit: &str) -> bool {
        let n = lit.chars().count();
        let matches = self
            .s
            .get(self.i..self.i + n)
            .is_some_and(|w| w.iter().copied().eq(lit.chars()));
        if matches {
            self.i += n;
        }
        matches
    }

    fn rest(&mut self) -> String {
        let r = self.s.get(self.i..).unwrap_or_default().iter().collect();
        self.i = self.s.len();
        r
    }

    fn ident(&mut self) -> crate::Result<String> {
        let start = self.i;
        if !self.peek().is_some_and(is_ident_start) {
            return Err(self.err("expected a name"));
        }
        while self.peek().is_some_and(is_ident_char) {
            self.i += 1;
        }
        Ok(self
            .s
            .get(start..self.i)
            .unwrap_or_default()
            .iter()
            .collect())
    }

    /// One JSON value starting at the cursor. Its extent is found first, because the grammar
    /// puts `|`, `=`, `,` or `)` right after values, which a JSON stream reader rejects.
    fn json(&mut self) -> crate::Result<Value> {
        let end = self.json_end().ok_or_else(|| self.err("invalid JSON"))?;
        let text: String = self.s.get(self.i..end).unwrap_or_default().iter().collect();
        let v = serde_json::from_str(&text).map_err(|_| self.err("invalid JSON"))?;
        self.i = end;
        Ok(v)
    }

    fn json_end(&self) -> Option<usize> {
        let mut i = self.i;
        let first = *self.s.get(i)?;
        if first == '"' || first == '{' || first == '[' {
            let (mut depth, mut in_str, mut esc) = (0usize, false, false);
            loop {
                let c = *self.s.get(i)?;
                i += 1;
                if in_str {
                    if esc {
                        esc = false;
                    } else if c == '\\' {
                        esc = true;
                    } else if c == '"' {
                        in_str = false;
                    }
                } else if c == '"' {
                    in_str = true;
                } else if c == '{' || c == '[' {
                    depth += 1;
                } else if c == '}' || c == ']' {
                    depth = depth.checked_sub(1)?;
                }
                if !in_str && depth == 0 {
                    return Some(i);
                }
            }
        }
        while self
            .s
            .get(i)
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '+' | '.'))
        {
            i += 1;
        }
        (i > self.i).then_some(i)
    }

    fn fields(&mut self, close: char) -> crate::Result<Vec<Field>> {
        let mut fields = Vec::new();
        while self.peek() != Some(close) {
            if !fields.is_empty() && !self.eat_str(", ") {
                return Err(self.err("expected `, `"));
            }
            let name = self.ident()?;
            let required = if self.peek() == Some('?') {
                self.i += 1;
                false
            } else {
                true
            };
            self.expect(':')?;
            let ty = self.ty()?;
            let default = if self.peek() == Some('=') {
                self.i += 1;
                Some(self.json()?)
            } else {
                None
            };
            let description = if self.eat_str(" \"") {
                self.i -= 1;
                match self.json()? {
                    Value::String(s) => Some(s),
                    _ => return Err(self.err("description")),
                }
            } else {
                None
            };
            fields.push(Field {
                name,
                required,
                ty,
                default,
                description,
            });
        }
        Ok(fields)
    }

    fn ty(&mut self) -> crate::Result<Ty> {
        match self.peek() {
            Some('[') => {
                self.i += 1;
                let items = if self.eat_str("any]") {
                    return Ok(Ty::Array(None));
                } else {
                    self.ty()?
                };
                self.expect(']')?;
                Ok(Ty::Array(Some(Box::new(items))))
            }
            Some('{') => {
                self.i += 1;
                let fields = self.fields('}')?;
                self.expect('}')?;
                Ok(Ty::Object(fields))
            }
            _ => self.scalar_or_enum(),
        }
    }

    fn scalar_or_enum(&mut self) -> crate::Result<Ty> {
        let first = self.enum_value()?;
        if self.peek() != Some('|') {
            let Some(word) = first.bare else {
                return Err(self.err("expected a type"));
            };
            return Ok(match word.as_str() {
                "str" if self.peek() == Some('<') => {
                    self.i += 1;
                    let format = self.ident()?;
                    self.expect('>')?;
                    Ty::Str {
                        format: Some(format),
                    }
                }
                "str" => Ty::Str { format: None },
                "datetime" => Ty::Str {
                    format: Some("date-time".into()),
                },
                "int" => Ty::Int,
                "num" => Ty::Num,
                "bool" => Ty::Bool,
                _ => return Err(self.err("unknown type")),
            });
        }
        let mut values = vec![first.value];
        while self.peek() == Some('|') {
            self.i += 1;
            if !self.starts_enum_value() {
                break; // trailing `|`: single-value enum
            }
            values.push(self.enum_value()?.value);
        }
        let integer = values.iter().all(Value::is_number);
        if !integer && !values.iter().all(Value::is_string) {
            return Err(self.err("mixed enum"));
        }
        Ok(Ty::Enum { integer, values })
    }

    fn starts_enum_value(&self) -> bool {
        self.peek()
            .is_some_and(|c| is_ident_start(c) || c == '"' || c == '-' || c.is_ascii_digit())
    }

    fn enum_value(&mut self) -> crate::Result<EnumValue> {
        match self.peek() {
            Some(c) if is_ident_start(c) => {
                let word = self.ident()?;
                Ok(EnumValue {
                    value: Value::String(word.clone()),
                    bare: Some(word),
                })
            }
            _ => Ok(EnumValue {
                value: self.json()?,
                bare: None,
            }),
        }
    }
}

struct EnumValue {
    value: Value,
    /// The word, when written bare. Alone (no `|`) it must be a type keyword.
    bare: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_tools;
    use serde_json::json;

    #[test]
    fn round_trips_every_construct() {
        let params = json!({"type":"object","properties":{
            "s":{"type":"string","description":"a \"quoted\" desc"},
            "f":{"type":"string","format":"uri"},
            "d":{"type":"string","format":"date-time","default":"2026-01-01T00:00:00Z"},
            "i":{"type":"integer","default":5},
            "n":{"type":"number"},
            "b":{"type":"boolean"},
            "e":{"type":"string","enum":["a","str","x y"]},
            "one":{"type":"string","enum":["only"]},
            "ie":{"type":"integer","enum":[-1,2]},
            "any":{"type":"array"},
            "nested":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string"},"v":{"type":"array","items":{"type":"integer"}}},"required":["k"]}}},
          "required":["s","i"]});
        let tools = vec![
            ToolDef {
                name: "t1".into(),
                description: Some("Multi\nline".into()),
                parameters: Some(params),
            },
            ToolDef {
                name: "t2".into(),
                description: None,
                parameters: Some(json!({"type":"object","properties":{}})),
            },
        ];
        assert_eq!(decode_tools(&encode_tools(&tools).unwrap()).unwrap(), tools);
    }
}
