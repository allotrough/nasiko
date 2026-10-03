use serde_json::Value;

/// A tool definition, independent of any router's wire types.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema of the arguments object.
    pub parameters: Option<Value>,
}

/// A decoded, validated tool call. The caller assigns the call id.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Always a JSON object.
    pub arguments: Value,
}
