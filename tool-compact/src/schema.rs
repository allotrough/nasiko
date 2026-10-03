//! The one schema model shared by the encoder, the definition parser and the validator.
//!
//! [`Ty::from_json`] accepts exactly the JSON Schema subset the grammar can express and names the
//! first keyword it cannot; [`Ty::to_json`] is its inverse. Because the validator walks this same
//! model, it can never accept a keyword the encoder dropped.

use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Ty {
    Str {
        format: Option<String>,
    },
    Int,
    Num,
    Bool,
    /// `None` = array without `items`.
    Array(Option<Box<Ty>>),
    /// Required fields first, in `required` order, then optional fields.
    Object(Vec<Field>),
    /// Non-empty; all strings (`type: string`) or all integers (`type: integer`).
    Enum {
        integer: bool,
        values: Vec<Value>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Field {
    pub name: String,
    pub required: bool,
    pub ty: Ty,
    pub default: Option<Value>,
    pub description: Option<String>,
}

/// Keywords spelled by the grammar itself, so a bare enum value may not use them.
pub(crate) const KEYWORDS: [&str; 6] = ["str", "int", "num", "bool", "any", "datetime"];

pub(crate) fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

pub(crate) fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')
}

pub(crate) fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(is_ident_start) && chars.all(is_ident_char)
}

impl Ty {
    /// The parameters object of a tool: must be an object schema (or absent).
    pub(crate) fn from_parameters(params: Option<&Value>) -> Result<Ty, String> {
        match params {
            None => Ok(Ty::Object(Vec::new())),
            Some(p) => match Ty::from_json(p)? {
                obj @ Ty::Object(_) => Ok(obj),
                _ => Err("non-object parameters".into()),
            },
        }
    }

    /// Errors carry the offending keyword (or a short description of the shape).
    pub(crate) fn from_json(schema: &Value) -> Result<Ty, String> {
        let obj = schema
            .as_object()
            .ok_or_else(|| "non-object schema".to_string())?;
        let kind = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "type".to_string())?;
        let allowed: &[&str] = match kind {
            "string" => &["type", "format", "enum"],
            "integer" => &["type", "enum"],
            "number" | "boolean" => &["type"],
            "array" => &["type", "items"],
            "object" => &["type", "properties", "required"],
            _ => return Err("type".into()),
        };
        if let Some(k) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
            return Err(k.clone());
        }

        if let Some(values) = obj.get("enum") {
            return enum_from_json(values, kind == "integer");
        }
        Ok(match kind {
            "string" => Ty::Str {
                format: match obj.get("format") {
                    None => None,
                    Some(Value::String(f)) if !f.is_empty() && f.chars().all(is_ident_char) => {
                        Some(f.clone())
                    }
                    Some(_) => return Err("format".into()),
                },
            },
            "integer" => Ty::Int,
            "number" => Ty::Num,
            "boolean" => Ty::Bool,
            "array" => Ty::Array(match obj.get("items") {
                None => None,
                Some(items) => Some(Box::new(Ty::from_json(items)?)),
            }),
            _ => Ty::Object(fields_from_json(obj)?),
        })
    }

    pub(crate) fn to_json(&self) -> Value {
        match self {
            Ty::Str { format: None } => json!({"type": "string"}),
            Ty::Str { format: Some(f) } => json!({"type": "string", "format": f}),
            Ty::Int => json!({"type": "integer"}),
            Ty::Num => json!({"type": "number"}),
            Ty::Bool => json!({"type": "boolean"}),
            Ty::Array(None) => json!({"type": "array"}),
            Ty::Array(Some(items)) => json!({"type": "array", "items": items.to_json()}),
            Ty::Enum { integer, values } => {
                json!({"type": if *integer { "integer" } else { "string" }, "enum": values})
            }
            Ty::Object(fields) => {
                let mut props = Map::new();
                for f in fields {
                    let mut s = f.ty.to_json();
                    if let Value::Object(m) = &mut s {
                        if let Some(d) = &f.default {
                            m.insert("default".into(), d.clone());
                        }
                        if let Some(d) = &f.description {
                            m.insert("description".into(), Value::String(d.clone()));
                        }
                    }
                    props.insert(f.name.clone(), s);
                }
                let mut out = json!({"type": "object", "properties": props});
                let required: Vec<Value> = fields
                    .iter()
                    .filter(|f| f.required)
                    .map(|f| Value::String(f.name.clone()))
                    .collect();
                if !required.is_empty()
                    && let Value::Object(m) = &mut out
                {
                    m.insert("required".into(), Value::Array(required));
                }
                out
            }
        }
    }
}

fn enum_from_json(values: &Value, integer: bool) -> Result<Ty, String> {
    let values = values.as_array().ok_or_else(|| "enum".to_string())?;
    let ok = !values.is_empty()
        && values.iter().all(|v| {
            if integer {
                v.is_i64() || v.is_u64()
            } else {
                v.is_string()
            }
        });
    if !ok {
        return Err("enum".into());
    }
    Ok(Ty::Enum {
        integer,
        values: values.clone(),
    })
}

fn fields_from_json(obj: &Map<String, Value>) -> Result<Vec<Field>, String> {
    let empty = Map::new();
    let props = match obj.get("properties") {
        None => &empty,
        Some(Value::Object(p)) => p,
        Some(_) => return Err("properties".into()),
    };
    let required: Vec<&str> = match obj.get("required") {
        None => Vec::new(),
        Some(Value::Array(r)) => r
            .iter()
            .map(|v| v.as_str().ok_or_else(|| "required".to_string()))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("required".into()),
    };
    if required.iter().any(|r| !props.contains_key(*r)) {
        return Err("required".into());
    }

    let field = |name: &str, required: bool| -> Result<Field, String> {
        if !is_ident(name) {
            return Err(format!("property name `{name}`"));
        }
        let mut schema = props
            .get(name)
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| "non-object schema".to_string())?;
        let description = match schema.remove("description") {
            None => None,
            Some(Value::String(d)) => Some(d),
            Some(_) => return Err("description".into()),
        };
        let default = schema.remove("default");
        Ok(Field {
            name: name.to_string(),
            required,
            ty: Ty::from_json(&Value::Object(schema))?,
            default,
            description,
        })
    };

    let mut fields = Vec::with_capacity(props.len());
    for name in &required {
        if fields.iter().any(|f: &Field| f.name == *name) {
            return Err("required".into());
        }
        fields.push(field(name, true)?);
    }
    for name in props.keys() {
        if !required.contains(&name.as_str()) {
            fields.push(field(name, false)?);
        }
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_fields_come_first_in_required_order() {
        let s = json!({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"},"c":{"type":"boolean"}},"required":["c","a"]});
        let Ty::Object(f) = Ty::from_json(&s).unwrap() else {
            unreachable!()
        };
        let names: Vec<_> = f.iter().map(|f| (f.name.as_str(), f.required)).collect();
        assert_eq!(names, [("c", true), ("a", true), ("b", false)]);
    }

    #[test]
    fn names_the_first_unsupported_keyword() {
        let s = json!({"type":"object","properties":{"a":{"type":"string","pattern":"x"}}});
        assert_eq!(Ty::from_json(&s), Err("pattern".into()));
        assert_eq!(Ty::from_json(&json!({"oneOf": []})), Err("type".into()));
        assert_eq!(
            Ty::from_json(&json!({"type": ["string", "null"]})),
            Err("type".into())
        );
    }

    #[test]
    fn to_json_inverts_from_json() {
        let s = json!({"type":"object","properties":{"t":{"type":"string","format":"date-time","description":"d"},"n":{"type":"integer","enum":[1,2],"default":1},"l":{"type":"array","items":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}}},"required":["t"]});
        assert_eq!(Ty::from_json(&s).unwrap().to_json(), s);
    }
}
