//! Arguments vs. schema. Walks the same [`Ty`] the encoder rendered, so it checks exactly the
//! keywords the model was shown.
//!
//! Stricter than JSON Schema in one place: a property the schema does not declare is rejected.
//! A hallucinated parameter is a guess, and guesses never reach the client.
//! `format` is an annotation, as in JSON Schema, and is not enforced.

use serde_json::Value;

use crate::schema::Ty;

/// `Err((path, reason))` on the first violation.
pub(crate) fn validate(v: &Value, ty: &Ty, path: &str) -> Result<(), (String, String)> {
    let fail = |reason: String| Err((path_or_root(path), reason));
    match ty {
        Ty::Str { .. } if v.is_string() => Ok(()),
        Ty::Int if v.is_i64() || v.is_u64() => Ok(()),
        Ty::Num if v.is_number() => Ok(()),
        Ty::Bool if v.is_boolean() => Ok(()),
        Ty::Enum { values, .. } if values.contains(v) => Ok(()),
        Ty::Enum { .. } => fail(format!("{v} is not an allowed value")),
        Ty::Array(items) => {
            let Some(arr) = v.as_array() else {
                return fail(format!("expected array, got {}", kind(v)));
            };
            match items {
                Some(items) => arr
                    .iter()
                    .enumerate()
                    .try_for_each(|(i, x)| validate(x, items, &format!("{path}[{i}]"))),
                None => Ok(()),
            }
        }
        Ty::Object(fields) => {
            let Some(obj) = v.as_object() else {
                return fail(format!("expected object, got {}", kind(v)));
            };
            if let Some(k) = obj.keys().find(|k| !fields.iter().any(|f| &f.name == *k)) {
                return fail(format!("unknown property `{k}`"));
            }
            for f in fields {
                let child = if path.is_empty() {
                    f.name.clone()
                } else {
                    format!("{path}.{}", f.name)
                };
                match obj.get(&f.name) {
                    Some(x) => validate(x, &f.ty, &child)?,
                    None if f.required => return Err((child, "required".into())),
                    None => {}
                }
            }
            Ok(())
        }
        Ty::Str { .. } => fail(format!("expected string, got {}", kind(v))),
        Ty::Int => fail(format!("expected integer, got {}", kind(v))),
        Ty::Num => fail(format!("expected number, got {}", kind(v))),
        Ty::Bool => fail(format!("expected boolean, got {}", kind(v))),
    }
}

fn path_or_root(path: &str) -> String {
    if path.is_empty() {
        "$".into()
    } else {
        path.into()
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "integer",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
