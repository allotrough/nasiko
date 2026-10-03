//! The invariants every change to the grammar, encoder or decoder must keep.
//!
//! Corpus: the eval sample's two tools, the 57 tools of the repo's own agents
//! (`fixtures/agent_tools.json`, extracted from `agents/*/src/tools.rs` and `agents/*/tools.go`),
//! and hand-written adversarial model outputs.

use nasiko_tool_compact::{
    Error, Level, Options, StreamDecoder, ToolCall, ToolDef, decode_calls, decode_output,
    decode_tools, encode_tools, encode_tools_with, render_call,
};
use serde_json::{Value, json};

fn to_defs(tools: &Value) -> Vec<ToolDef> {
    tools
        .as_array()
        .unwrap()
        .iter()
        .map(|t| ToolDef {
            name: t["function"]["name"].as_str().unwrap().into(),
            description: t["function"]["description"].as_str().map(Into::into),
            parameters: t["function"].get("parameters").cloned(),
        })
        .collect()
}

fn agent_tools() -> Vec<(String, Vec<ToolDef>)> {
    let raw: Value = serde_json::from_str(include_str!("fixtures/agent_tools.json")).unwrap();
    raw.as_object()
        .unwrap()
        .iter()
        .map(|(agent, tools)| (agent.clone(), to_defs(tools)))
        .collect()
}

fn eval_tools() -> Vec<ToolDef> {
    to_defs(&json!([
      {"type":"function","function":{"name":"create_calendar_event","description":"Create an event in the user's calendar.",
        "parameters":{"type":"object","properties":{
          "title":{"type":"string","description":"Event title"},
          "start":{"type":"string","format":"date-time","description":"Start time, ISO 8601"},
          "duration_min":{"type":"integer","description":"Duration in minutes"},
          "attendees":{"type":"array","items":{"type":"string"},"description":"Attendee emails"},
          "visibility":{"type":"string","enum":["public","private"]}},
        "required":["title","start"]}}},
      {"type":"function","function":{"name":"send_email","description":"Send an email from the user's account.",
        "parameters":{"type":"object","properties":{
          "to":{"type":"array","items":{"type":"string"},"description":"Recipient emails"},
          "subject":{"type":"string","description":"Subject line"},
          "body":{"type":"string","description":"Plain-text body"},
          "cc":{"type":"array","items":{"type":"string"},"description":"CC emails"}},
        "required":["to","subject","body"]}}}
    ]))
}

/// JSON Schema equality modulo what the grammar normalizes: an empty `required`, a missing
/// `properties` on an object.
fn normalized(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut out: serde_json::Map<String, Value> = m
                .iter()
                .filter(|(k, v)| {
                    !(k.as_str() == "required" && v.as_array().is_some_and(Vec::is_empty))
                })
                .map(|(k, v)| (k.clone(), normalized(v)))
                .collect();
            if out.get("type") == Some(&json!("object")) && !out.contains_key("properties") {
                out.insert("properties".into(), json!({}));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(normalized).collect()),
        other => other.clone(),
    }
}

// ── 1. Lossless ─────────────────────────────────────────────────────────────

#[test]
fn i1_every_agent_tool_round_trips_losslessly() {
    let mut checked = 0;
    for (agent, tools) in agent_tools()
        .into_iter()
        .chain([("eval".into(), eval_tools())])
    {
        let compact = encode_tools(&tools).unwrap_or_else(|e| panic!("{agent}: {e}"));
        let back = decode_tools(&compact).unwrap_or_else(|e| panic!("{agent}: {e}"));
        assert_eq!(back.len(), tools.len());
        for (orig, got) in tools.iter().zip(&back) {
            assert_eq!(got.name, orig.name);
            assert_eq!(got.description, orig.description, "{agent}/{}", orig.name);
            let want = normalized(
                orig.parameters
                    .as_ref()
                    .unwrap_or(&json!({"type":"object"})),
            );
            assert_eq!(
                normalized(got.parameters.as_ref().unwrap()),
                want,
                "{agent}/{}",
                orig.name
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 59);
}

// ── 2. Deterministic ────────────────────────────────────────────────────────

#[test]
fn i2_encoding_is_byte_identical_across_runs() {
    for (_, tools) in agent_tools() {
        assert_eq!(encode_tools(&tools).unwrap(), encode_tools(&tools).unwrap());
    }
}

// ── 3. Fail closed ──────────────────────────────────────────────────────────

#[test]
fn i3_unsupported_schemas_are_refused_whole() {
    let mut tools = eval_tools();
    tools.push(ToolDef {
        name: "x".into(),
        description: None,
        parameters: Some(json!({"type":"object","properties":{"a":{"$ref":"#/defs/a"}}})),
    });
    assert!(matches!(encode_tools(&tools), Err(Error::Unsupported { tool, .. }) if tool == "x"));
}

#[test]
fn i3_invalid_calls_are_errors_never_calls() {
    let tools = eval_tools();
    let cases = [
        ("<<call delete_everything {}>>", "unknown_tool"),
        (
            r#"<<call create_calendar_event {"start":"2026-10-05T15:00:00+05:30","visibility":"secret"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","visibility":"secret"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","duration_min":30.5}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","duration_min":"30"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","attendees":"[\"x\"]"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","attendees":[1]}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b","location":"x"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":null,"start":"b"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a" "start":"b"}>>"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b"}"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event {"title":"a","start":"b"} done"#,
            "invalid_arguments",
        ),
        (
            r#"<<call create_calendar_event ["a"]>>"#,
            "invalid_arguments",
        ),
        // Call attempts with a broken marker (seen live from ministral-3-8b) are errors, not text.
        (
            "<<send_email>>
{\"to\":[\"a@b.c\"],\"subject\":\"s\",\"body\":\"b\"}",
            "invalid_arguments",
        ),
        (
            "I will now <<create_calendar_event {\"title\":\"a\",\"start\":\"b\"}>>",
            "invalid_arguments",
        ),
        ("trailing <<send_email", "invalid_arguments"),
        // Native call markup instead of the compact format (seen live from kimi-k2.5): an
        // intended call, so an error the caller retries — never text handed to the client.
        (
            " I'll read it. <|tool_calls_section_begin|> <|tool_call_begin|> functions.read_file:0",
            "invalid_arguments",
        ),
        (
            "[TOOL_CALLS]create_calendar_event{\"title\":\"a\",\"start\":\"b\"}",
            "invalid_arguments",
        ),
        (
            "<tool_call>
{\"name\": \"send_email\", \"arguments\": {}}
</tool_call>",
            "invalid_arguments",
        ),
        // Seen live from qwen3-32b: `}}>` instead of `}>>`.
        (
            "<<call create_calendar_event {\"title\":\"a\",\"start\":\"b\" }}>",
            "invalid_arguments",
        ),
        // A good call followed by a bad one fails the whole turn: no partial set.
        (
            r#"<<call create_calendar_event {"title":"a","start":"b"}>> <<call nope {}>>"#,
            "unknown_tool",
        ),
    ];
    for (text, code) in cases {
        let err = decode_calls(text, &tools).expect_err(text);
        assert_eq!(err.code(), code, "{text}: {err}");
    }
}

// ── 4. Split invariance: the official decoder cases, split everywhere ────────

fn stream(chunks: &[&str], tools: &[ToolDef]) -> Result<(String, Vec<ToolCall>), Error> {
    let mut d = StreamDecoder::new(tools).unwrap();
    let mut text = String::new();
    for c in chunks {
        text.push_str(&d.push(c));
    }
    let (tail, calls) = d.finish()?;
    text.push_str(&tail);
    Ok((text, calls))
}

/// (chunks, expected calls or error code), as published in `compact-tools-eval@v1-sample`.
type DecoderCase = (Vec<&'static str>, Result<Vec<Value>, &'static str>);

fn official_decoder_cases() -> Vec<DecoderCase> {
    vec![
        (
            vec![
                r#"<<call create_calendar_event {"title":"Design review","start":"2026-10-05T15:00:00+05:30"}>>"#,
            ],
            Ok(vec![
                json!({"name":"create_calendar_event","arguments":{"title":"Design review","start":"2026-10-05T15:00:00+05:30"}}),
            ]),
        ),
        (
            vec![
                "<<ca",
                r#"ll create_calendar_event {"title":"Ret"#,
                r#"ro","start":"2026-10-04T10:00:00+05:30"}>"#,
                ">",
            ],
            Ok(vec![
                json!({"name":"create_calendar_event","arguments":{"title":"Retro","start":"2026-10-04T10:00:00+05:30"}}),
            ]),
        ),
        (
            vec![r#"<<call send_email {"to":["sam@example.com"],"subject":"a >> b","body":"x"}>>"#],
            Ok(vec![
                json!({"name":"send_email","arguments":{"to":["sam@example.com"],"subject":"a >> b","body":"x"}}),
            ]),
        ),
        (vec!["<<call delete_everything {}>>"], Err("unknown_tool")),
        (
            vec![
                r#"<<call create_calendar_event {"start":"2026-10-05T15:00:00+05:30","visibility":"secret"}>>"#,
            ],
            Err("invalid_arguments"),
        ),
    ]
}

fn as_json(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": c.arguments}))
        .collect()
}

#[test]
fn i4_official_decoder_cases_pass_with_their_own_chunking() {
    let tools = eval_tools();
    for (chunks, want) in official_decoder_cases() {
        match (stream(&chunks, &tools), want) {
            (Ok((_, calls)), Ok(want)) => assert_eq!(as_json(&calls), want),
            (Err(e), Err(code)) => assert_eq!(e.code(), code),
            (got, want) => panic!("{chunks:?}: got {got:?}, want {want:?}"),
        }
    }
}

#[test]
fn i4_every_split_point_gives_the_one_shot_result() {
    let tools = eval_tools();
    let mut inputs: Vec<String> = official_decoder_cases()
        .into_iter()
        .map(|(c, _)| c.concat())
        .collect();
    inputs.extend(
        adversarial_outputs()
            .into_iter()
            .map(|(t, _)| t.to_string()),
    );
    for text in inputs {
        let one_shot = stream(&[&text], &tools);
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        for &(i, _) in &chars {
            let (a, b) = text.split_at(i);
            assert_eq!(stream(&[a, b], &tools), one_shot, "split at {i}: {text}");
        }
        // Every char its own chunk: the worst case a provider can produce.
        let singles: Vec<String> = text.chars().map(String::from).collect();
        let singles: Vec<&str> = singles.iter().map(String::as_str).collect();
        assert_eq!(stream(&singles, &tools), one_shot, "char by char: {text}");
    }
}

// ── 5. Real model output shapes ─────────────────────────────────────────────

/// (output, expected tool names). Text around calls must survive verbatim.
fn adversarial_outputs() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("It is sunny, no tool needed.", vec![]),
        ("a < b and a << b, but <<caller is not a call.", vec![]),
        (
            "C++ streams: cout << x << endl; and <<send_emails is another word.",
            vec![],
        ),
        ("trailing marker prefix <<ca", vec![]),
        (
            "<<<call create_calendar_event {\"title\":\"a\",\"start\":\"b\"}>>",
            vec!["create_calendar_event"],
        ),
        (
            "Sure, booking it.\n<<call create_calendar_event {\"title\":\"x\",\"start\":\"y\"}>>\nDone.",
            vec!["create_calendar_event"],
        ),
        (
            "<<call send_email {\"to\":[\"a@b.c\"],\"subject\":\"}>> {tricky\",\"body\":\"line\\n\\\"quoted\\\" \\\\\"}>>",
            vec!["send_email"],
        ),
        (
            "<<call send_email {\"to\":[\"a@b.c\"],\"subject\":\"s\",\"body\":\"b\"}>>\n<<call create_calendar_event {\"title\":\"Retro\",\"start\":\"2026-10-04T10:00:00+05:30\",\"duration_min\":30,\"visibility\":\"private\"}>>",
            vec!["send_email", "create_calendar_event"],
        ),
        (
            "<<call\tcreate_calendar_event\n{\"title\":\"a\",\"start\":\"b\"} >>",
            vec!["create_calendar_event"],
        ),
        (
            "unicode ✓ before <<call create_calendar_event {\"title\":\"réunion 会议\",\"start\":\"b\"}>> after ✓",
            vec!["create_calendar_event"],
        ),
    ]
}

#[test]
fn i5_real_output_shapes_decode_with_text_preserved() {
    let tools = eval_tools();
    for (text, names) in adversarial_outputs() {
        let d = decode_output(text, &tools).unwrap_or_else(|e| panic!("{text}: {e}"));
        let got: Vec<&str> = d.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(got, names, "{text}");
        if names.is_empty() {
            assert_eq!(d.text, text, "plain text must pass through untouched");
        }
    }
}

// ── 6. Render ↔ decode, and the generated example ───────────────────────────

#[test]
fn i6_rendered_calls_decode_back_unchanged() {
    let tools = eval_tools();
    let calls = vec![
        ToolCall {
            name: "send_email".into(),
            arguments: json!({"to":["sam@example.com"],"subject":"a >> b \"q\"","body":"x\ny"}),
        },
        ToolCall {
            name: "create_calendar_event".into(),
            arguments: json!({"title":"Retro","start":"2026-10-04T10:00:00+05:30","duration_min":30,"visibility":"private"}),
        },
    ];
    let text: Vec<String> = calls.iter().map(render_call).collect();
    assert_eq!(decode_calls(&text.join("\n"), &tools).unwrap(), calls);
}

#[test]
fn i6_the_generated_example_is_a_valid_call() {
    for (agent, tools) in agent_tools()
        .into_iter()
        .chain([("eval".into(), eval_tools())])
    {
        let opts = Options {
            level: Level::Lossless,
            example: true,
        };
        let text = encode_tools_with(&tools, &opts).unwrap().text().to_string();
        let example = text.rsplit("Example: ").next().unwrap();
        let calls = decode_calls(example, &tools).unwrap_or_else(|e| panic!("{agent}: {e}"));
        assert_eq!(calls.len(), 1, "{agent}");
    }
}

// ── 7. Levels ───────────────────────────────────────────────────────────────

#[test]
fn i7_brief_is_never_longer_and_keeps_structure() {
    for (_, tools) in agent_tools() {
        let full = encode_tools(&tools).unwrap();
        let brief = encode_tools_with(
            &tools,
            &Options {
                level: Level::Brief,
                example: false,
            },
        )
        .unwrap();
        assert!(brief.text().len() <= full.text().len());
        let (a, b) = (decode_tools(&full).unwrap(), decode_tools(&brief).unwrap());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(
                x.parameters, y.parameters,
                "Brief only touches tool descriptions"
            );
        }
    }
}

// ── 8. Every real agent tool: valid calls decode, broken calls fail ─────────

/// A value satisfying `schema`, with every optional field filled, so each property's type is
/// exercised. Built from the JSON Schema directly, independent of the crate's own model.
fn sample(schema: &Value) -> Value {
    if let Some(first) = schema["enum"].as_array().and_then(|e| e.last()) {
        return first.clone();
    }
    match schema["type"].as_str() {
        Some("string") => json!("some \"text\" with >> and }"),
        Some("integer") => json!(7),
        Some("number") => json!(2.5),
        Some("boolean") => json!(false),
        Some("array") => match schema.get("items") {
            Some(items) => json!([sample(items), sample(items)]),
            None => json!(["anything", 1]),
        },
        _ => Value::Object(
            schema["properties"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.clone(), sample(v)))
                .collect(),
        ),
    }
}

#[test]
fn i8_every_agent_tool_accepts_valid_calls_and_rejects_broken_ones() {
    let (mut valid, mut rejected) = (0, 0);
    for (agent, tools) in agent_tools() {
        for tool in &tools {
            let schema = tool.parameters.clone().unwrap_or(json!({"type":"object"}));
            let call = ToolCall {
                name: tool.name.clone(),
                arguments: sample(&schema),
            };
            let text = format!("Working on it.\n{}\nDone.", render_call(&call));
            assert_eq!(
                decode_calls(&text, &tools).unwrap(),
                vec![call.clone()],
                "{agent}/{}",
                tool.name
            );
            valid += 1;

            let mut broken = Vec::new();
            if let Some(req) = schema["required"].as_array().and_then(|r| r.first()) {
                let mut a = call.arguments.clone();
                a.as_object_mut().unwrap().remove(req.as_str().unwrap());
                broken.push(("missing required", a));
            }
            let mut a = call.arguments.clone();
            a.as_object_mut()
                .unwrap()
                .insert("hallucinated_param".into(), json!(1));
            broken.push(("unknown property", a));
            if let Some((k, _)) = schema["properties"]
                .as_object()
                .and_then(|p| p.iter().next())
            {
                let mut a = call.arguments.clone();
                a[k] = json!({"wrong": "type"});
                broken.push(("wrong type", a));
            }
            for (what, arguments) in broken {
                let bad = render_call(&ToolCall {
                    name: tool.name.clone(),
                    arguments,
                });
                let err =
                    decode_calls(&bad, &tools).expect_err(&format!("{agent}/{} {what}", tool.name));
                assert_eq!(
                    err.code(),
                    "invalid_arguments",
                    "{agent}/{} {what}",
                    tool.name
                );
                rejected += 1;
            }
        }
    }
    assert_eq!(valid, 57);
    assert!(rejected >= 57 * 2, "rejected only {rejected}");
}

#[test]
fn i4_native_markup_is_caught_at_every_split_point() {
    let tools = eval_tools();
    for text in [
        "Sure. <|tool_calls_section_begin|> functions.send_email:0",
        "ok [TOOL_CALLS]send_email{}",
        "<tool_call>{}</tool_call>",
    ] {
        for (i, _) in text.char_indices() {
            let (a, b) = text.split_at(i);
            let err = stream(&[a, b], &tools).expect_err(text);
            assert_eq!(err.code(), "invalid_arguments", "split at {i}: {text}");
        }
    }
}
