//! Compact tool definitions for LLM requests, and a fail-closed decoder for the calls a model
//! writes back.
//!
//! Agents resend their full JSON Schema `tools` array on every call. [`encode_tools`] rewrites it
//! as one signature line per tool plus call-format instructions; the model replies with
//! `<<call name {json}>>`, and [`decode_calls`] / [`StreamDecoder`] turn that text back into
//! [`ToolCall`]s, validated against the original schema.
//!
//! # Grammar
//!
//! ```text
//! tool  = NAME "(" [field {", " field}] ")" [" - " desc]
//! field = NAME ["?"] ":" type ["=" JSON] [" " JSON_STRING]
//! type  = "str" | "int" | "num" | "bool" | "datetime" | "str<" FORMAT ">"
//!       | "[" type "]" | "[any]" | "{" [field {", " field}] "}" | value "|" [value {"|" value}]
//! call  = "<<call" WS NAME WS? JSON_OBJECT WS? ">>"
//! ```
//!
//! Required fields are written first, in `required` order, so the order survives the round trip.
//! Call arguments stay plain JSON: compact *output* formats break models (multi-turn parse
//! cascades, collapsed parallel calls), compact *input* does not.
//!
//! # Invariants
//!
//! * **Lossless** — [`decode_tools`] rebuilds every schema [`encode_tools`] accepted
//!   ([`Level::Lossless`]).
//! * **Fail-closed** — a schema feature outside the grammar is [`Error::Unsupported`] (the caller
//!   sends that request natively); an unknown tool, missing required field, unknown property,
//!   wrong type or enum violation is an error, never a repaired call.
//! * **One validator per grammar** — encoding, decoding and validation share one schema model,
//!   so the validator checks exactly the keywords the encoder accepts.
//! * **Split-invariant** — [`decode_calls`] is a [`StreamDecoder`] fed one chunk, so streamed and
//!   one-shot decoding cannot disagree.
//! * **Deterministic** — no clock, RNG, I/O or environment.

#![forbid(unsafe_code)]
#![deny(
    clippy::string_slice,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

mod decode;
mod encode;
mod error;
mod parse;
mod schema;
mod types;
mod validate;

pub use decode::{Decoded, StreamDecoder, decode_calls, decode_output, render_call};
pub use encode::{CompactTools, Level, Options, encode_tools, encode_tools_with};
pub use error::Error;
pub use parse::decode_tools;
pub use types::{ToolCall, ToolDef};

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;
