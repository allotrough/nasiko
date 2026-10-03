//! Eval harness for compact tool schemas (`nasiko-tool-compact`).
//!
//! ```sh
//! curl -fsSL https://registry.nasiko.dev/r/nasiko/compact-tools-eval -o /tmp/compact-tools-eval.json
//! EVAL_SET=/tmp/compact-tools-eval.json OUT=/tmp/out.jsonl \
//!   cargo run --release -p nasiko-llm-router --example compact_tools_eval
//! ```
//!
//! Writes one JSONL line per case to `OUT` (outputs only; the scorer computes every metric):
//!
//! * `cases` → `{id, compact_request, compacted, rendered_calls, roundtrip_calls}`.
//!   `compact_request` is the OpenAI-shaped body we would send; `compacted: false` means the
//!   tools were sent natively because the schema is outside the grammar.
//! * `decoder_cases` → `{id, decoded}`: the case's chunks fed to `StreamDecoder` in order,
//!   giving `{calls: [...]}` or `{error: "unknown_tool" | "invalid_arguments"}`.
//!
//! Offline and deterministic by default. The call grammar is the one the decoder cases are
//! written in (`<<call name {json}>>`), so no case is converted.
//!
//! # Environment
//!
//! | Var | Default | Meaning |
//! |---|---|---|
//! | `EVAL_SET` | `compact-tools-eval.json` | dataset path |
//! | `OUT` | `compact-tools-out.jsonl` | output path |
//! | `COMPACT_LEVEL` | `lossless` | `lossless` or `brief` (first sentence of tool descriptions) |
//! | `COMPACT_EXAMPLE` | `0` | `1` appends one generated example call to the instructions |
//! | `PROVIDER_BASE_URL`, `MODEL` | unset | live mode: POST each `compact_request` to `{PROVIDER_BASE_URL}/chat/completions` at temperature 0 and add `raw_output`, `live_calls` |
//! | `PROVIDER_API_KEY` | unset | bearer token for live mode |
//! | `LIVE_BASELINE` | `0` | `1` also sends the native request and adds `native_calls` plus both `usage.prompt_tokens` |
//!
//! A token summary (o200k_base, full request body) goes to stderr; it is a convenience, not a
//! claim — the scorer recounts.
//!
//! The fixed reference time (`Today: 2026-10-02 (Asia/Kolkata)`) goes only into live requests,
//! on the compact and the native arm alike, so offline `compact_request` compares like for like
//! with the `{messages, tools}` baseline.

use std::fmt::Write as _;
use std::io::Write as _;

use nasiko_tool_compact::{
    Level, Options, StreamDecoder, ToolCall, ToolDef, decode_calls, decode_output,
    encode_tools_with, render_call,
};
use serde_json::{Value, json};

/// Fixed reference time for every run, so relative dates resolve identically.
const REFERENCE: &str = "Today: 2026-10-02 (Asia/Kolkata)";

type BoxError = Box<dyn std::error::Error>;

fn main() -> Result<(), BoxError> {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let eval_path = env("EVAL_SET", "compact-tools-eval.json");
    let out_path = env("OUT", "compact-tools-out.jsonl");
    let opts = Options {
        level: match env("COMPACT_LEVEL", "lossless").as_str() {
            "lossless" => Level::Lossless,
            "brief" => Level::Brief,
            other => {
                return Err(format!("COMPACT_LEVEL must be lossless|brief, got {other}").into());
            }
        },
        example: env("COMPACT_EXAMPLE", "0") == "1",
    };
    let live = match (std::env::var("PROVIDER_BASE_URL"), std::env::var("MODEL")) {
        (Ok(base), Ok(model)) => Some(Live {
            base: base.trim_end_matches('/').to_string(),
            model,
            key: std::env::var("PROVIDER_API_KEY").ok(),
            baseline: env("LIVE_BASELINE", "0") == "1",
            // One client for the run; the timeout keeps a hung provider from stalling the eval.
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()?,
        }),
        _ => None,
    };

    let set: Value = serde_json::from_str(&std::fs::read_to_string(&eval_path)?)?;
    let catalog = set["tools"]
        .as_array()
        .ok_or("EVAL_SET has no `tools` array")?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    let bpe = tiktoken_rs::o200k_base()?;
    let tokens = |v: &Value| bpe.encode_with_special_tokens(&v.to_string()).len();
    let (mut native_sum, mut compact_sum, mut bypassed) = (0usize, 0usize, 0usize);

    for case in set["cases"].as_array().into_iter().flatten() {
        let id = case["id"].as_str().unwrap_or_default();
        let tools = select(catalog, &case["tools"])?;
        let defs = to_defs(&tools)?;
        let messages = case["messages"].as_array().cloned().unwrap_or_default();

        let native = json!({"messages": messages, "tools": tools});
        let (request, compacted) = match encode_tools_with(&defs, &opts) {
            Ok(compact) => {
                let mut msgs = vec![json!({"role": "system", "content": compact.text()})];
                msgs.extend(messages.iter().cloned());
                (json!({"messages": msgs}), true)
            }
            // Outside the grammar: send natively. Counts as 0% savings, never a lossy guess.
            Err(_) => {
                bypassed += 1;
                (native.clone(), false)
            }
        };
        native_sum += tokens(&native);
        compact_sum += tokens(&request);

        let expected = expected_calls(&case["expected"])?;
        let rendered = expected
            .iter()
            .map(render_call)
            .collect::<Vec<_>>()
            .join("\n");
        // The reference time is context for live runs, not part of compaction: offline, the
        // request is compared like for like with the scorer's `{messages, tools}` baseline; live,
        // both arms carry the same line (see `Live::run`) and `OUT` records what was sent.
        let sent = if live.is_some() {
            with_reference(&request)
        } else {
            request.clone()
        };
        let mut line = json!({
            "id": id,
            "compact_request": sent,
            "compacted": compacted,
            "rendered_calls": rendered,
            "roundtrip_calls": calls_or_error(decode_calls(&rendered, &defs)),
        });

        if let Some(live) = &live {
            live.run(&mut line, &sent, compacted, &native, &defs);
        }
        writeln!(out, "{line}")?;
    }

    for case in set["decoder_cases"].as_array().into_iter().flatten() {
        let tools = select(catalog, &case["tools"])?;
        let mut decoder = StreamDecoder::new(&to_defs(&tools)?)?;
        for chunk in case["chunks"].as_array().into_iter().flatten() {
            decoder.push(chunk.as_str().unwrap_or_default());
        }
        let decoded = match decoder.finish() {
            Ok((_, calls)) => json!({"calls": calls_json(&calls)}),
            Err(e) => json!({"error": e.code()}),
        };
        writeln!(out, "{}", json!({"id": case["id"], "decoded": decoded}))?;
    }
    out.flush()?;

    let mut summary = String::new();
    write!(
        summary,
        "compact_tools_eval: native {native_sum} -> compact {compact_sum} tokens (o200k_base, full body)"
    )?;
    if native_sum > 0 {
        write!(
            summary,
            ", {:.1}% saved",
            100.0 * (1.0 - compact_sum as f64 / native_sum as f64)
        )?;
    }
    eprintln!("{summary}; {bypassed} case(s) bypassed; wrote {out_path}");
    Ok(())
}

/// `body` with the fixed reference time: merged into a leading system message if there is one
/// (the compact definitions), otherwise as a new leading system message.
fn with_reference(body: &Value) -> Value {
    let mut body = body.clone();
    if let Some(msgs) = body["messages"].as_array_mut() {
        match msgs.first_mut() {
            Some(m) if m["role"] == "system" && m["content"].is_string() => {
                let text = m["content"].as_str().unwrap_or_default();
                m["content"] = json!(format!("{REFERENCE}\n{text}"));
            }
            _ => msgs.insert(0, json!({"role": "system", "content": REFERENCE})),
        }
    }
    body
}

/// The case's tools, looked up by name in the file's catalog.
fn select(catalog: &[Value], names: &Value) -> Result<Vec<Value>, BoxError> {
    names
        .as_array()
        .into_iter()
        .flatten()
        .map(|n| {
            catalog
                .iter()
                .find(|t| t["function"]["name"] == *n)
                .cloned()
                .ok_or_else(|| format!("tool {n} is not in the catalog").into())
        })
        .collect()
}

/// OpenAI tool JSON → codec types (what the router's IR conversion does at its seam).
fn to_defs(tools: &[Value]) -> Result<Vec<ToolDef>, BoxError> {
    tools
        .iter()
        .map(|t| {
            let f = &t["function"];
            Ok(ToolDef {
                name: f["name"].as_str().ok_or("tool without a name")?.to_string(),
                description: f["description"].as_str().map(str::to_string),
                parameters: f.get("parameters").cloned(),
            })
        })
        .collect()
}

fn expected_calls(expected: &Value) -> Result<Vec<ToolCall>, BoxError> {
    expected
        .as_array()
        .into_iter()
        .flatten()
        .map(|e| {
            Ok(ToolCall {
                name: e["name"]
                    .as_str()
                    .ok_or("expected call without a name")?
                    .to_string(),
                arguments: e["arguments"].clone(),
            })
        })
        .collect()
}

fn calls_json(calls: &[ToolCall]) -> Vec<Value> {
    calls
        .iter()
        .map(|c| json!({"name": c.name, "arguments": c.arguments}))
        .collect()
}

fn calls_or_error(r: nasiko_tool_compact::Result<Vec<ToolCall>>) -> Value {
    match r {
        Ok(calls) => Value::Array(calls_json(&calls)),
        Err(e) => json!({"error": e.code()}),
    }
}

struct Live {
    base: String,
    model: String,
    key: Option<String>,
    baseline: bool,
    client: reqwest::Client,
}

impl Live {
    /// Adds `raw_output` and `live_calls` (and, with `LIVE_BASELINE=1`, the native comparison).
    /// Network failures are recorded in the line, never fatal: one flaky call must not lose
    /// the rest of the run.
    fn run(
        &self,
        line: &mut Value,
        request: &Value,
        compacted: bool,
        native: &Value,
        defs: &[ToolDef],
    ) {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                line["live_error"] = json!(e.to_string());
                return;
            }
        };
        match rt.block_on(self.send(request)) {
            Ok(resp) => {
                let msg = &resp["choices"][0]["message"];
                line["raw_output"] = msg["content"].clone();
                line["live_calls"] = if compacted {
                    match decode_output(msg["content"].as_str().unwrap_or_default(), defs) {
                        Ok(d) => Value::Array(calls_json(&d.calls)),
                        Err(e) => json!({"error": e.code()}),
                    }
                } else {
                    native_calls(msg)
                };
                line["live_prompt_tokens"] = resp["usage"]["prompt_tokens"].clone();
            }
            Err(e) => line["live_error"] = json!(e.to_string()),
        }
        if self.baseline {
            let req = with_reference(native);
            match rt.block_on(self.send(&req)) {
                Ok(resp) => {
                    line["native_calls"] = native_calls(&resp["choices"][0]["message"]);
                    line["native_prompt_tokens"] = resp["usage"]["prompt_tokens"].clone();
                }
                Err(e) => line["native_error"] = json!(e.to_string()),
            }
        }
    }

    async fn send(&self, body: &Value) -> Result<Value, reqwest::Error> {
        let mut body = body.clone();
        body["model"] = json!(self.model);
        body["temperature"] = json!(0);
        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.base))
            .json(&body);
        if let Some(key) = &self.key {
            req = req.bearer_auth(key);
        }
        req.send().await?.error_for_status()?.json().await
    }
}

/// Native `tool_calls` in the same `{name, arguments}` shape as decoded calls.
fn native_calls(msg: &Value) -> Value {
    Value::Array(
        msg["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tc| {
                let args = tc["function"]["arguments"].as_str().unwrap_or("{}");
                json!({
                    "name": tc["function"]["name"],
                    "arguments": serde_json::from_str::<Value>(args).unwrap_or(json!(args)),
                })
            })
            .collect(),
    )
}
