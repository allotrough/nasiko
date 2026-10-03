//! Input-token reduction for tool definitions: compact signatures out, native `tool_calls` back.
//!
//! The request's JSON Schema `tools` are replaced by one leading `system` message of compact
//! signatures plus a one-line call format ([`nasiko_tool_compact`]); the model writes calls as
//! `<<call name {json}>>` text; [`decode_response`] validates them against the **original**
//! schemas and rebuilds OpenAI-shaped `tool_calls`. The client never sees the compact form.
//!
//! # Fail closed, then fall back
//!
//! A response the decoder rejects (unknown tool, schema violation, malformed call) is never
//! repaired. The handler re-sends the untouched request with native tools instead, so the
//! client gets a correct answer and the only cost of a miss is one extra provider call.
//!
//! # When it does not run
//!
//! Anything the text format cannot guarantee is sent natively (see [`Skipped`]): forced
//! `tool_choice` and `parallel_tool_calls: false` are enforced by providers, not by prompts;
//! a transcript that already holds native tool calls would leave Anthropic/Gemini with
//! `tool_use` blocks but no declared tools; a schema outside the grammar would lose meaning.
//! Streaming is not covered yet — the codec supports it (`StreamDecoder`), the wiring does not.
//!
//! # Why a leading system message
//!
//! Tool definitions are the most stable part of a request. Putting their compact form first
//! keeps the provider's cached prefix stable across turns, where they sat natively.

use nasiko_tool_compact::{CompactTools, ToolDef as CompactDef, decode_output, encode_tools};
use serde_json::Value;

use crate::config::GatewayConfig;
use crate::ir::chat::{FunctionCall, Message, ToolCall};
use crate::ir::{ChatRequest, ChatResponse, Usage};
use crate::resolver::ResolvedConfig;

/// Why the request was sent with native tools. Stable labels: they are queryable values.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Skipped {
    Disabled,
    AgentOptedOut,
    NoTools,
    Streaming,
    ToolChoiceForced,
    SequentialCallsRequired,
    HistoryHasToolCalls,
    UnsupportedSchema,
    NotSmaller,
}

impl Skipped {
    pub(crate) fn as_label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::AgentOptedOut => "agent_opted_out",
            Self::NoTools => "no_tools",
            Self::Streaming => "streaming",
            Self::ToolChoiceForced => "tool_choice_forced",
            Self::SequentialCallsRequired => "parallel_tool_calls_false",
            Self::HistoryHasToolCalls => "history_has_tool_calls",
            Self::UnsupportedSchema => "unsupported_schema",
            Self::NotSmaller => "not_smaller",
        }
    }
}

/// A request that went out compacted: what is needed to decode its response or retry it.
#[derive(Debug)]
pub(crate) struct Applied {
    /// The request exactly as the client sent it, for the native retry.
    pub native: ChatRequest,
    defs: Vec<CompactDef>,
    pub bytes_before: usize,
    pub bytes_after: usize,
}

/// Compact `req`'s tools in place, unless a carve-out applies.
pub(crate) fn apply(
    req: &mut ChatRequest,
    cfg: &GatewayConfig,
    resolved: &ResolvedConfig,
) -> Result<Applied, Skipped> {
    if !cfg.compact_tools_enabled {
        return Err(Skipped::Disabled);
    }
    if !resolved.compress_enabled {
        return Err(Skipped::AgentOptedOut);
    }
    let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) else {
        return Err(Skipped::NoTools);
    };
    if req.is_streaming() {
        return Err(Skipped::Streaming);
    }
    if !matches!(&req.tool_choice, None | Some(Value::String(_)))
        || req
            .tool_choice
            .as_ref()
            .and_then(Value::as_str)
            .is_some_and(|c| c != "auto")
    {
        return Err(Skipped::ToolChoiceForced);
    }
    if req.extra.get("parallel_tool_calls") == Some(&Value::Bool(false)) {
        return Err(Skipped::SequentialCallsRequired);
    }
    if req
        .messages
        .iter()
        .any(|m| m.role == "tool" || m.tool_calls.is_some())
    {
        return Err(Skipped::HistoryHasToolCalls);
    }
    // Fields the codec does not model (`type` other than function, provider extensions such as
    // `cache_control`) would be silently dropped by compaction.
    if tools
        .iter()
        .any(|t| t.kind != "function" || !t.extra.is_empty())
    {
        return Err(Skipped::UnsupportedSchema);
    }

    let defs: Vec<CompactDef> = tools
        .iter()
        .map(|t| CompactDef {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            parameters: t.function.parameters.clone(),
        })
        .collect();
    let compact: CompactTools = encode_tools(&defs).map_err(|_| Skipped::UnsupportedSchema)?;
    let bytes_before = serde_json::to_string(tools).map_or(0, |s| s.len());
    let bytes_after = compact.text().len();
    if bytes_after >= bytes_before {
        return Err(Skipped::NotSmaller);
    }

    let native = req.clone();
    req.tools = None;
    req.tool_choice = None;
    req.messages.insert(
        0,
        Message {
            role: "system".into(),
            content: Some(Value::String(compact.text().to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            extra: Default::default(),
        },
    );
    Ok(Applied {
        native,
        defs,
        bytes_before,
        bytes_after,
    })
}

/// Turn the model's text calls back into native `tool_calls`, in place.
///
/// `Err` means the response must not reach the client: re-send [`Applied::native`] instead.
pub(crate) fn decode_response(
    resp: &mut ChatResponse,
    applied: &Applied,
) -> Result<(), nasiko_tool_compact::Error> {
    for choice in &mut resp.choices {
        let Some(Value::String(text)) = &choice.message.content else {
            continue;
        };
        let decoded = decode_output(text, &applied.defs)?;
        if decoded.calls.is_empty() {
            continue;
        }
        let calls = decoded
            .calls
            .into_iter()
            .map(|c| ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                kind: "function".into(),
                function: FunctionCall {
                    name: c.name,
                    arguments: c.arguments.to_string(),
                },
                extra: Default::default(),
            })
            .collect();
        let rest = decoded.text.trim();
        choice.message.content = (!rest.is_empty()).then(|| Value::String(rest.to_string()));
        choice.message.tool_calls = Some(calls);
        choice.finish_reason = Some("tool_calls".into());
    }
    Ok(())
}

/// Usage of a rejected compact attempt plus its native retry: what the provider billed.
pub(crate) fn sum_usage(a: Option<Usage>, b: Option<Usage>) -> Option<Usage> {
    let add = |x: Option<i64>, y: Option<i64>| match (x, y) {
        (None, None) => None,
        (x, y) => Some(x.unwrap_or(0) + y.unwrap_or(0)),
    };
    match (a, b) {
        (Some(a), Some(b)) => Some(Usage {
            prompt_tokens: add(a.prompt_tokens, b.prompt_tokens),
            completion_tokens: add(a.completion_tokens, b.completion_tokens),
            total_tokens: add(a.total_tokens, b.total_tokens),
            ..b
        }),
        (a, b) => b.or(a),
    }
}

/// `token_usage`-style record of the decision, logged per request.
pub(crate) fn to_metadata(outcome: &Result<Applied, Skipped>, retried: bool) -> Value {
    match outcome {
        Ok(a) => serde_json::json!({
            "applied": true,
            "tool_bytes_before": a.bytes_before,
            "tool_bytes_after": a.bytes_after,
            "native_retry": retried,
        }),
        Err(reason) => serde_json::json!({"applied": false, "skipped": reason.as_label()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::chat::Choice;
    use serde_json::json;

    fn cfg(enabled: bool) -> GatewayConfig {
        GatewayConfig {
            compact_tools_enabled: enabled,
            ..Default::default()
        }
    }

    fn resolved(compress_enabled: bool) -> ResolvedConfig {
        ResolvedConfig {
            provider: "openai".into(),
            model: "gpt-4o-mini".into(),
            litellm_model: "openai/gpt-4o-mini".into(),
            api_key: "sk-test".into(),
            fallback_models: vec![],
            temperature: None,
            max_tokens: None,
            has_llm_config: false,
            pinned_model: None,
            tier1_model: None,
            tier2_model: None,
            tier3_model: None,
            platform_paid: true,
            custom_endpoint: None,
            is_coding_agent: false,
            compress_enabled,
        }
    }

    fn request() -> ChatRequest {
        serde_json::from_value(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": {"type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"},
                                   "unit": {"type": "string", "enum": ["c", "f"]}},
                    "required": ["city"]}}}]
        }))
        .unwrap()
    }

    fn response(content: &str) -> ChatResponse {
        ChatResponse {
            id: "x".into(),
            object: "chat.completion".into(),
            created: None,
            model: "m".into(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: Some(Value::String(content.into())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn disabled_leaves_the_request_byte_identical() {
        for (c, r) in [(cfg(false), resolved(true)), (cfg(true), resolved(false))] {
            let mut req = request();
            let before = serde_json::to_string(&req).unwrap();
            assert!(apply(&mut req, &c, &r).is_err());
            assert_eq!(serde_json::to_string(&req).unwrap(), before);
        }
    }

    #[test]
    fn applied_replaces_tools_with_a_leading_system_message() {
        let mut req = request();
        let a = apply(&mut req, &cfg(true), &resolved(true)).unwrap();
        assert!(req.tools.is_none());
        assert_eq!(req.messages[0].role, "system");
        let text = req.messages[0].text().unwrap();
        assert!(text.starts_with("get_weather(city:str \"City name\", unit?:c|f)"));
        assert!(a.bytes_after < a.bytes_before);
        assert!(
            a.native.tools.is_some(),
            "the retry copy keeps the native tools"
        );
    }

    /// A request mutation and the carve-out it must trigger.
    type CarveOut = (fn(&mut ChatRequest), Skipped);

    #[test]
    fn every_carve_out_sends_natively_and_unchanged() {
        let cases: [CarveOut; 6] = [
            (|r| r.tools = None, Skipped::NoTools),
            (|r| r.stream = Some(true), Skipped::Streaming),
            (
                |r| r.tool_choice = Some(json!("required")),
                Skipped::ToolChoiceForced,
            ),
            (
                |r| {
                    r.extra.insert("parallel_tool_calls".into(), json!(false));
                },
                Skipped::SequentialCallsRequired,
            ),
            (
                |r| {
                    r.messages.push(
                        serde_json::from_value(
                            json!({"role":"tool","tool_call_id":"c1","content":"18C"}),
                        )
                        .unwrap(),
                    )
                },
                Skipped::HistoryHasToolCalls,
            ),
            (
                |r| {
                    r.tools.as_mut().unwrap()[0].function.parameters =
                        Some(json!({"type":"object","properties":{"city":{"oneOf":[]}}}))
                },
                Skipped::UnsupportedSchema,
            ),
        ];
        for (mutate, want) in cases {
            let mut req = request();
            mutate(&mut req);
            let before = serde_json::to_string(&req).unwrap();
            assert_eq!(
                apply(&mut req, &cfg(true), &resolved(true)).unwrap_err(),
                want
            );
            assert_eq!(serde_json::to_string(&req).unwrap(), before, "{want:?}");
        }
    }

    #[test]
    fn decoded_calls_become_native_tool_calls() {
        let mut req = request();
        let a = apply(&mut req, &cfg(true), &resolved(true)).unwrap();
        let mut resp = response("Checking.\n<<call get_weather {\"city\":\"Paris\"}>>");
        decode_response(&mut resp, &a).unwrap();
        let msg = &resp.choices[0].message;
        let call = &msg.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.name, "get_weather");
        assert_eq!(call.function.arguments, r#"{"city":"Paris"}"#);
        assert!(call.id.starts_with("call_"));
        assert_eq!(msg.content, Some(json!("Checking.")));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn plain_answers_pass_through_untouched() {
        let mut req = request();
        let a = apply(&mut req, &cfg(true), &resolved(true)).unwrap();
        let mut resp = response("It is sunny in Paris.");
        decode_response(&mut resp, &a).unwrap();
        assert_eq!(
            resp.choices[0].message.content,
            Some(json!("It is sunny in Paris."))
        );
        assert!(resp.choices[0].message.tool_calls.is_none());
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn invalid_calls_are_errors_for_the_retry_path() {
        let mut req = request();
        let a = apply(&mut req, &cfg(true), &resolved(true)).unwrap();
        for bad in [
            "<<call get_weather {\"unit\":\"c\"}>>",
            "<<call get_weather {\"city\":\"Paris\",\"unit\":\"k\"}>>",
            "<<call delete_all {}>>",
            "<<get_weather>> {\"city\":\"Paris\"}",
        ] {
            assert!(decode_response(&mut response(bad), &a).is_err(), "{bad}");
        }
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(
            Skipped::HistoryHasToolCalls.as_label(),
            "history_has_tool_calls"
        );
        assert_eq!(Skipped::Disabled.as_label(), "disabled");
    }
}
