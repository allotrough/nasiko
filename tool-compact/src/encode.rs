//! JSON Schema tools → one signature line per tool, plus the call-format instructions.

use serde_json::{Map, Value};

use crate::decode::render_call;
use crate::error::Error;
use crate::schema::{Field, KEYWORDS, Ty, is_ident};
use crate::types::{ToolCall, ToolDef};

/// How much of each definition to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Level {
    /// Everything, descriptions verbatim. [`crate::decode_tools`] rebuilds the input exactly.
    #[default]
    Lossless,
    /// Tool descriptions cut to their first sentence; parameter descriptions kept. Lossy.
    Brief,
}

/// Encoder options. The default is lossless, without an example call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Options {
    pub level: Level,
    /// Append one example call to the instructions: the shortest valid call among the tools. Costs
    /// ~25 tokens per request; worth it only if live adherence measurably improves.
    pub example: bool,
}

/// The compact rendering of a tool set.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactTools {
    text: String,
    definitions: String,
}

impl CompactTools {
    /// The full block to inject as a system message: definitions plus instructions.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Only the signature lines, one per tool.
    pub fn definitions(&self) -> &str {
        &self.definitions
    }
}

// Paid on every request, so it is one line and there is no header: on the eval sample a notation
// guide made requests 9% *larger* than native. The signature syntax is TypeScript-like enough
// that models read `?`, `[T]` and `a|b` unaided. Wording measured live (8 models, 7 providers,
// eval sample): "Tool calls: …" 19/24 well-formed; this imperative "write exactly" 22/24 (native
// tool calling: 21/24) for 5 more tokens.
const HOW_TO_CALL: &str = "To call a tool, write exactly: <<call NAME {JSON args}>>";

/// [`encode_tools_with`] using [`Options::default`].
pub fn encode_tools(tools: &[ToolDef]) -> crate::Result<CompactTools> {
    encode_tools_with(tools, &Options::default())
}

/// Encode `tools`, or name the first schema feature the grammar cannot express.
///
/// An empty tool list or a duplicate name is also unsupported: there is nothing to compact, or
/// a call could not be resolved to one schema.
pub fn encode_tools_with(tools: &[ToolDef], opts: &Options) -> crate::Result<CompactTools> {
    let unsupported = |tool: &str, feature: String| Error::Unsupported {
        tool: tool.to_string(),
        feature,
    };
    if tools.is_empty() {
        return Err(unsupported("", "empty tool list".into()));
    }

    let mut definitions = String::new();
    // Shortest rendered valid call: the example teaches the syntax, so fewer tokens is better.
    let mut example: Option<String> = None;
    for (i, tool) in tools.iter().enumerate() {
        if !is_ident(&tool.name) {
            return Err(unsupported(&tool.name, "tool name".into()));
        }
        if tools[..i].iter().any(|t| t.name == tool.name) {
            return Err(unsupported(&tool.name, "duplicate tool name".into()));
        }
        let ty = Ty::from_parameters(tool.parameters.as_ref())
            .map_err(|f| unsupported(&tool.name, f))?;
        let Ty::Object(fields) = &ty else {
            return Err(unsupported(&tool.name, "non-object parameters".into()));
        };

        if i > 0 {
            definitions.push('\n');
        }
        definitions.push_str(&tool.name);
        definitions.push('(');
        write_fields(fields, &mut definitions);
        definitions.push(')');
        if let Some(desc) = &tool.description {
            let desc = match opts.level {
                Level::Lossless => desc.as_str(),
                Level::Brief => first_sentence(desc),
            };
            definitions.push_str(" - ");
            write_description(desc, &mut definitions);
        }
        if opts.example {
            let call = render_call(&ToolCall {
                name: tool.name.clone(),
                arguments: example_value(&ty),
            });
            if example.as_ref().is_none_or(|e| call.len() < e.len()) {
                example = Some(call);
            }
        }
    }

    let mut text = String::with_capacity(definitions.len() + 200);
    text.push_str(&definitions);
    text.push('\n');
    text.push_str(HOW_TO_CALL);
    if let Some(call) = example {
        text.push_str(" Example: ");
        text.push_str(&call);
    }
    Ok(CompactTools { text, definitions })
}

fn write_fields(fields: &[Field], out: &mut String) {
    for (i, f) in fields.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&f.name);
        if !f.required {
            out.push('?');
        }
        out.push(':');
        write_ty(&f.ty, out);
        if let Some(d) = &f.default {
            out.push('=');
            out.push_str(&d.to_string());
        }
        if let Some(d) = &f.description {
            out.push(' ');
            out.push_str(&Value::String(d.clone()).to_string());
        }
    }
}

fn write_ty(ty: &Ty, out: &mut String) {
    match ty {
        Ty::Str { format: None } => out.push_str("str"),
        Ty::Str { format: Some(f) } if f == "date-time" => out.push_str("datetime"),
        Ty::Str { format: Some(f) } => {
            out.push_str("str<");
            out.push_str(f);
            out.push('>');
        }
        Ty::Int => out.push_str("int"),
        Ty::Num => out.push_str("num"),
        Ty::Bool => out.push_str("bool"),
        Ty::Array(None) => out.push_str("[any]"),
        Ty::Array(Some(items)) => {
            out.push('[');
            write_ty(items, out);
            out.push(']');
        }
        Ty::Object(fields) => {
            out.push('{');
            write_fields(fields, out);
            out.push('}');
        }
        Ty::Enum { values, .. } => {
            for (i, v) in values.iter().enumerate() {
                if i > 0 {
                    out.push('|');
                }
                match v {
                    Value::String(s) if is_ident(s) && !KEYWORDS.contains(&s.as_str()) => {
                        out.push_str(s)
                    }
                    other => out.push_str(&other.to_string()),
                }
            }
            if values.len() == 1 {
                out.push('|');
            }
        }
    }
}

/// Plain text unless it would be ambiguous (multi-line, leading quote, empty); then JSON.
fn write_description(desc: &str, out: &mut String) {
    if desc.is_empty() || desc.starts_with('"') || desc.contains(['\n', '\r']) {
        out.push_str(&Value::String(desc.to_string()).to_string());
    } else {
        out.push_str(desc);
    }
}

/// Up to and including the first ". " or line break; the whole text if there is none.
fn first_sentence(desc: &str) -> &str {
    let end = desc
        .char_indices()
        .find(|&(i, c)| {
            c == '\n' || (c == '.' && desc.get(i + 1..).is_some_and(|r| r.starts_with(' ')))
        })
        .map(|(i, c)| if c == '.' { i + 1 } else { i });
    end.and_then(|e| desc.get(..e)).unwrap_or(desc)
}

/// A deterministic, schema-valid placeholder: required fields only.
fn example_value(ty: &Ty) -> Value {
    match ty {
        Ty::Str { format: Some(f) } if f == "date-time" => "2026-01-01T09:00:00Z".into(),
        Ty::Str { .. } => "text".into(),
        Ty::Int => 1.into(),
        Ty::Num => Value::from(1.5),
        Ty::Bool => true.into(),
        Ty::Enum { values, .. } => values.first().cloned().unwrap_or(Value::Null),
        Ty::Array(None) => Value::Array(Vec::new()),
        Ty::Array(Some(items)) => Value::Array(vec![example_value(items)]),
        Ty::Object(fields) => Value::Object(
            fields
                .iter()
                .filter(|f| f.required)
                .map(|f| (f.name.clone(), example_value(&f.ty)))
                .collect::<Map<_, _>>(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, desc: Option<&str>, params: Value) -> ToolDef {
        ToolDef {
            name: name.into(),
            description: desc.map(Into::into),
            parameters: Some(params),
        }
    }

    #[test]
    fn renders_the_calendar_example() {
        let t = tool(
            "create_calendar_event",
            Some("Create an event in the user's calendar."),
            json!({"type":"object","properties":{
                "title":{"type":"string","description":"Event title"},
                "start":{"type":"string","format":"date-time"},
                "attendees":{"type":"array","items":{"type":"string"}},
                "visibility":{"type":"string","enum":["public","private"]}},
              "required":["title","start"]}),
        );
        let c = encode_tools_with(
            &[t],
            &Options {
                level: Level::Lossless,
                example: true,
            },
        )
        .unwrap();
        assert_eq!(
            c.definitions(),
            "create_calendar_event(title:str \"Event title\", start:datetime, attendees?:[str], visibility?:public|private) - Create an event in the user's calendar."
        );
        assert!(c.text().ends_with(
            " Example: <<call create_calendar_event {\"start\":\"2026-01-01T09:00:00Z\",\"title\":\"text\"}>>"
        ));
    }

    #[test]
    fn quotes_enum_values_that_would_be_ambiguous() {
        let t = tool(
            "f",
            None,
            json!({"type":"object","properties":{"k":{"type":"string","enum":["str","a b","ok"]},"one":{"type":"string","enum":["x"]},"n":{"type":"integer","enum":[1,-2]}}}),
        );
        assert_eq!(
            encode_tools(&[t]).unwrap().definitions(),
            "f(k?:\"str\"|\"a b\"|ok, n?:1|-2, one?:x|)"
        );
    }

    #[test]
    fn unsupported_schema_is_named_not_dropped() {
        let t = tool(
            "f",
            None,
            json!({"type":"object","properties":{"a":{"anyOf":[]}}}),
        );
        assert_eq!(
            encode_tools(&[t]),
            Err(Error::Unsupported {
                tool: "f".into(),
                feature: "type".into()
            })
        );
    }

    #[test]
    fn brief_keeps_the_first_sentence() {
        assert_eq!(first_sentence("Does a. Then b."), "Does a.");
        assert_eq!(first_sentence("v1.2 is fine"), "v1.2 is fine");
        assert_eq!(first_sentence("Line one\nline two"), "Line one");
    }
}
