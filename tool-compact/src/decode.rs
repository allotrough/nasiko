//! Model text → validated tool calls.
//!
//! One character-level state machine serves both entry points: [`decode_calls`] /
//! [`decode_output`] are a [`StreamDecoder`] fed a single chunk. Text outside calls streams out
//! as it arrives (holding back only what may still become `<<call`); calls are released together
//! by [`StreamDecoder::finish`], so one bad call fails the whole turn instead of leaving a
//! partial set behind.

use std::collections::VecDeque;

use serde_json::Value;

use crate::error::Error;
use crate::schema::{Ty, is_ident_char, is_ident_start};
use crate::types::{ToolCall, ToolDef};
use crate::validate::validate;

const OPEN: &str = "<<call";
/// A single call's arguments above this size are refused rather than buffered.
const MAX_CALL_BYTES: usize = 1 << 20;
/// Longest word held after `<<` while deciding whether it is a call.
const MAX_NAME_BYTES: usize = 128;
/// Native tool-call markup that models fall back to when they ignore the compact format
/// (Kimi, Mistral, Hermes/Qwen, Llama). Seen as text, it is a call the client must not receive:
/// it fails the turn so the caller retries with native tools.
const NATIVE_MARKERS: [&str; 5] = [
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "[TOOL_CALLS]",
    "<tool_call>",
    "<|python_tag|>",
];
/// Enough trailing text to recognise the longest marker across chunk boundaries.
const RECENT_KEEP: usize = 32;

/// Write `call` in the compact call grammar.
pub fn render_call(call: &ToolCall) -> String {
    format!("{OPEN} {} {}>>", call.name, call.arguments)
}

/// Text and calls of one complete model response.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    /// Everything outside the calls, verbatim.
    pub text: String,
    pub calls: Vec<ToolCall>,
}

/// Decode a complete response and keep only the calls.
pub fn decode_calls(text: &str, tools: &[ToolDef]) -> crate::Result<Vec<ToolCall>> {
    decode_output(text, tools).map(|d| d.calls)
}

/// Decode a complete response into its text and calls.
pub fn decode_output(text: &str, tools: &[ToolDef]) -> crate::Result<Decoded> {
    let mut d = StreamDecoder::new(tools)?;
    let mut out = d.push(text);
    let (tail, calls) = d.finish()?;
    out.push_str(&tail);
    Ok(Decoded { text: out, calls })
}

#[derive(Debug)]
enum State {
    Text,
    /// Holding `<`, or `<<` plus an identifier: maybe `<<call`, maybe `<<tool_name` (an error).
    Marker(String),
    Name(String),
    BeforeArgs(String),
    Args {
        name: String,
        json: String,
        depth: usize,
        in_str: bool,
        esc: bool,
    },
    Close {
        name: String,
        json: String,
        seen_gt: bool,
    },
    Failed,
}

/// Incremental decoder for streamed model output.
#[derive(Debug)]
pub struct StreamDecoder {
    tools: Vec<(String, Ty)>,
    state: State,
    calls: Vec<ToolCall>,
    error: Option<Error>,
    /// Tail of the text emitted so far, for [`NATIVE_MARKERS`].
    recent: String,
}

impl StreamDecoder {
    /// Fails only if a tool's schema is outside the grammar (it could not have been encoded).
    pub fn new(tools: &[ToolDef]) -> crate::Result<Self> {
        let tools = tools
            .iter()
            .map(|t| {
                Ty::from_parameters(t.parameters.as_ref())
                    .map(|ty| (t.name.clone(), ty))
                    .map_err(|feature| Error::Unsupported {
                        tool: t.name.clone(),
                        feature,
                    })
            })
            .collect::<crate::Result<_>>()?;
        Ok(StreamDecoder {
            tools,
            state: State::Text,
            calls: Vec::new(),
            error: None,
            recent: String::new(),
        })
    }

    /// Feed the next chunk; returns the text that is now certainly not part of a call.
    pub fn push(&mut self, chunk: &str) -> String {
        let mut out = String::new();
        let mut queue: VecDeque<char> = chunk.chars().collect();
        while let Some(c) = queue.pop_front() {
            self.step(c, &mut out, &mut queue);
        }
        out
    }

    /// End of stream: trailing text, and every call — or the first error.
    pub fn finish(self) -> crate::Result<(String, Vec<ToolCall>)> {
        if let Some(e) = self.error {
            return Err(e);
        }
        match self.state {
            State::Text => Ok((String::new(), self.calls)),
            State::Marker(held) if held.strip_prefix("<<").is_some_and(|w| self.is_tool(w)) => {
                Err(Error::Malformed("tool call without the `<<call` marker"))
            }
            // Ended on something like "<<ca": that was text.
            State::Marker(held) => Ok((held, self.calls)),
            _ => Err(Error::Malformed("unterminated call")),
        }
    }

    fn is_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|(n, _)| n == name)
    }

    /// Pass one char of plain text through; `false` if it completed native call markup.
    fn emit(&mut self, c: char, out: &mut String) -> bool {
        out.push(c);
        self.recent.push(c);
        if self.recent.len() > 8 * RECENT_KEEP {
            let skip = self.recent.chars().count().saturating_sub(RECENT_KEEP);
            self.recent = self.recent.chars().skip(skip).collect();
        }
        if NATIVE_MARKERS.iter().any(|m| self.recent.ends_with(m)) {
            self.fail(Error::Malformed(
                "native tool-call markup instead of `<<call`",
            ));
            return false;
        }
        true
    }

    fn fail(&mut self, e: Error) {
        if self.error.is_none() {
            self.error = Some(e);
        }
        self.state = State::Failed;
    }

    fn step(&mut self, c: char, out: &mut String, queue: &mut VecDeque<char>) {
        let state = std::mem::replace(&mut self.state, State::Failed);
        self.state = match state {
            State::Failed => State::Failed,
            State::Text if c == '<' => State::Marker("<".into()),
            State::Text => {
                if !self.emit(c, out) {
                    return;
                }
                State::Text
            }
            State::Marker(mut held) => {
                let word = held.strip_prefix("<<");
                let extends = match word {
                    None => c == '<',
                    Some("") => is_ident_start(c),
                    Some(w) => is_ident_char(c) && w.len() < MAX_NAME_BYTES,
                };
                if extends {
                    held.push(c);
                    State::Marker(held)
                } else if word == Some("call") && c.is_whitespace() {
                    State::Name(String::new())
                } else if word.is_some_and(|w| self.is_tool(w)) {
                    // `<<send_email>>{..}`: a call attempt with a malformed marker. Passing it on
                    // as text would hand the client a half-made call; fail so the caller retries.
                    self.fail(Error::Malformed("tool call without the `<<call` marker"));
                    return;
                } else {
                    // Not a call: emit the first char, re-scan the rest (it may hold a real `<`).
                    held.push(c);
                    let mut chars = held.chars();
                    if let Some(first) = chars.next()
                        && !self.emit(first, out)
                    {
                        return;
                    }
                    for (i, rc) in chars.enumerate() {
                        queue.insert(i, rc);
                    }
                    State::Text
                }
            }
            State::Name(mut name) => {
                if name.is_empty() && c.is_whitespace() {
                    State::Name(name)
                } else if (name.is_empty() && is_ident_start(c))
                    || (!name.is_empty() && is_ident_char(c))
                {
                    name.push(c);
                    State::Name(name)
                } else if !name.is_empty() && c.is_whitespace() {
                    State::BeforeArgs(name)
                } else if !name.is_empty() && c == '{' {
                    args_start(name)
                } else {
                    self.fail(Error::Malformed("expected a tool name"));
                    return;
                }
            }
            State::BeforeArgs(name) if c.is_whitespace() => State::BeforeArgs(name),
            State::BeforeArgs(name) if c == '{' => args_start(name),
            State::BeforeArgs(_) => {
                self.fail(Error::Malformed("expected `{` after the tool name"));
                return;
            }
            State::Args {
                name,
                mut json,
                mut depth,
                mut in_str,
                mut esc,
            } => {
                json.push(c);
                if json.len() > MAX_CALL_BYTES {
                    self.fail(Error::Malformed("call arguments too large"));
                    return;
                }
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
                } else if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                }
                if depth == 0 {
                    State::Close {
                        name,
                        json,
                        seen_gt: false,
                    }
                } else {
                    State::Args {
                        name,
                        json,
                        depth,
                        in_str,
                        esc,
                    }
                }
            }
            State::Close {
                name,
                json,
                seen_gt,
            } => match (c, seen_gt) {
                (c, false) if c.is_whitespace() => State::Close {
                    name,
                    json,
                    seen_gt,
                },
                ('>', false) => State::Close {
                    name,
                    json,
                    seen_gt: true,
                },
                ('>', true) => match self.complete(name, &json) {
                    Ok(call) => {
                        self.calls.push(call);
                        State::Text
                    }
                    Err(e) => {
                        self.fail(e);
                        return;
                    }
                },
                _ => {
                    self.fail(Error::Malformed("expected `>>` after the arguments"));
                    return;
                }
            },
        };
    }

    fn complete(&self, name: String, json: &str) -> crate::Result<ToolCall> {
        let Some((_, ty)) = self.tools.iter().find(|(n, _)| *n == name) else {
            return Err(Error::UnknownTool(name));
        };
        let arguments: Value = serde_json::from_str(json)
            .map_err(|_| Error::Malformed("arguments are not valid JSON"))?;
        validate(&arguments, ty, "").map_err(|(path, reason)| Error::InvalidArguments {
            tool: name.clone(),
            path,
            reason,
        })?;
        Ok(ToolCall { name, arguments })
    }
}

fn args_start(name: String) -> State {
    State::Args {
        name,
        json: "{".into(),
        depth: 1,
        in_str: false,
        esc: false,
    }
}
