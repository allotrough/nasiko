use thiserror::Error;

/// Every error this crate surfaces.
///
/// Note what is absent: a "best guess" outcome. Decoding either returns calls that validate
/// against the original schema or one of these errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// The schema uses something the grammar cannot express; the caller must send the request
    /// with native tools instead.
    #[error("tool `{tool}` uses unsupported schema feature `{feature}`")]
    Unsupported { tool: String, feature: String },

    /// Compact definition text that this crate did not produce.
    #[error("compact definitions are malformed: {0}")]
    BadDefinitions(String),

    /// The model called a tool that was not offered.
    #[error("unknown tool `{0}`")]
    UnknownTool(String),

    /// The call names a real tool but its arguments do not satisfy the schema.
    #[error("invalid arguments for `{tool}` at `{path}`: {reason}")]
    InvalidArguments {
        tool: String,
        path: String,
        reason: String,
    },

    /// Text that started a call but is not a well-formed one (bad JSON, unterminated, oversized).
    #[error("malformed tool call: {0}")]
    Malformed(&'static str),
}

impl Error {
    /// Stable label for telemetry and eval output.
    pub fn code(&self) -> &'static str {
        match self {
            Error::Unsupported { .. } => "unsupported_schema",
            Error::BadDefinitions(_) => "bad_definitions",
            Error::UnknownTool(_) => "unknown_tool",
            // A malformed call is a call with arguments we cannot accept.
            Error::InvalidArguments { .. } | Error::Malformed(_) => "invalid_arguments",
        }
    }
}
