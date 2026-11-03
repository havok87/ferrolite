//! Request preparation and translation between the OpenAI chat and
//! Anthropic Messages formats.
//!
//! Bodies are handled as `serde_json::Value` so unknown fields can pass
//! through untouched on passthrough paths and errors can name the field at
//! fault. Translated output is built with `json!`.

pub mod anthropic_to_openai;
pub mod openai_to_anthropic;
pub mod sse;

use serde_json::{Map, Value};

/// Client-supplied fields that could redirect a request or inject
/// credentials. Removed before routing, along with any `litellm_*` key.
pub const STRIPPED_FIELDS: &[&str] = &[
    "api_base",
    "base_url",
    "api_key",
    "api_version",
    "custom_llm_provider",
    "extra_headers",
    "mock_response",
];

/// Removes routing and credential fields a client must not control.
pub fn sanitize(body: &mut Map<String, Value>) {
    body.retain(|k, _| !STRIPPED_FIELDS.contains(&k.as_str()) && !k.starts_with("litellm_"));
}

/// Fills in deployment defaults the client did not send.
pub fn apply_defaults(body: &mut Map<String, Value>, defaults: &Map<String, Value>) {
    for (k, v) in defaults {
        body.entry(k.clone()).or_insert_with(|| v.clone());
    }
}

/// Fills in only the defaults whose names mean the same in the Anthropic
/// Messages format.
pub fn apply_shared_defaults(body: &mut Map<String, Value>, defaults: &Map<String, Value>) {
    const SHARED: &[&str] = &["max_tokens", "temperature", "top_p", "top_k"];
    for (k, v) in defaults {
        if SHARED.contains(&k.as_str()) {
            body.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
}

/// A request that cannot be translated as sent. Always the client's fault.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TranslateError(pub String);

impl TranslateError {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// Removes parameters the target cannot accept, or fails naming them,
/// as LiteLLM does when `drop_params` is false.
pub(crate) fn check_params(
    body: &mut Map<String, Value>,
    is_supported: impl Fn(&str, &Value) -> bool,
    drop_params: bool,
    target: &str,
) -> Result<(), TranslateError> {
    let unsupported: Vec<String> = body
        .iter()
        .filter(|(k, v)| !is_supported(k, v))
        .map(|(k, _)| k.clone())
        .collect();
    if unsupported.is_empty() {
        return Ok(());
    }
    if drop_params {
        for k in &unsupported {
            body.remove(k);
        }
        return Ok(());
    }
    Err(TranslateError::new(format!(
        "{target} does not support parameters: {}. Remove them, or set drop_params: true to drop them",
        unsupported.join(", ")
    )))
}

/// Token counts from an upstream response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens, including cached ones.
    pub prompt: u64,
    pub completion: u64,
}

impl Usage {
    pub fn total(self) -> u64 {
        self.prompt.saturating_add(self.completion)
    }

    /// Reads usage from any of the shapes upstreams return: OpenAI
    /// (`prompt_tokens`), Anthropic and Responses (`input_tokens`, with
    /// Anthropic's cache counts added to the input).
    pub fn from_value(usage: &Value) -> Option<Self> {
        let n = |key: &str| usage.get(key).and_then(Value::as_u64);
        if let Some(prompt) = n("prompt_tokens") {
            return Some(Self {
                prompt,
                completion: n("completion_tokens").unwrap_or(0),
            });
        }
        let output = n("output_tokens");
        // Anthropic's message_delta carries only output_tokens.
        let input = match (n("input_tokens"), output) {
            (None, None) => return None,
            (input, _) => input.unwrap_or(0),
        };
        let cached = n("cache_creation_input_tokens")
            .unwrap_or(0)
            .saturating_add(n("cache_read_input_tokens").unwrap_or(0));
        Some(Self {
            prompt: input.saturating_add(cached),
            completion: output.unwrap_or(0),
        })
    }
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The text of a string or of an array of `{"type":"text","text":...}`
/// parts, joined with newlines. `None` for any other part type.
pub(crate) fn plain_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Null => Some(String::new()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).map(str::to_owned),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|t| t.join("\n")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => panic!("not an object"),
        }
    }

    #[test]
    fn sanitize_strips_routing_and_credential_fields() {
        let mut body = map(json!({
            "model": "m", "messages": [],
            "api_base": "http://evil", "base_url": "http://evil", "api_key": "k",
            "api_version": "v", "custom_llm_provider": "openai",
            "extra_headers": {"x": "y"}, "mock_response": "hi",
            "litellm_logging_obj": 1, "litellm_trace_id": "t",
            "temperature": 0.5
        }));
        sanitize(&mut body);
        let mut keys: Vec<&str> = body.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["messages", "model", "temperature"]);
    }

    #[test]
    fn defaults_never_override_the_client() {
        let mut body = map(json!({"temperature": 0.9}));
        let defaults = map(json!({"temperature": 0.1, "max_tokens": 10, "seed": 1}));
        apply_defaults(&mut body, &defaults);
        assert_eq!(
            Value::Object(body),
            json!({"temperature": 0.9, "max_tokens": 10, "seed": 1})
        );

        let mut anthropic = map(json!({}));
        apply_shared_defaults(&mut anthropic, &defaults);
        assert_eq!(
            Value::Object(anthropic),
            json!({"temperature": 0.1, "max_tokens": 10})
        );
    }

    #[test]
    fn check_params_drops_or_fails() {
        let supported = |k: &str, _: &Value| k == "model";
        let mut body = map(json!({"model": "m", "seed": 1, "logprobs": true}));
        let err = check_params(&mut body.clone(), supported, false, "anthropic").unwrap_err();
        assert!(
            err.0
                .contains("anthropic does not support parameters: logprobs, seed"),
            "{err}"
        );
        check_params(&mut body, supported, true, "anthropic").unwrap();
        assert_eq!(Value::Object(body), json!({"model": "m"}));
    }

    #[test]
    fn usage_shapes() {
        assert_eq!(
            Usage::from_value(&json!({"prompt_tokens": 10, "completion_tokens": 5})),
            Some(Usage {
                prompt: 10,
                completion: 5
            })
        );
        assert_eq!(
            Usage::from_value(&json!({
                "input_tokens": 10, "output_tokens": 5,
                "cache_creation_input_tokens": 3, "cache_read_input_tokens": 2
            })),
            Some(Usage {
                prompt: 15,
                completion: 5
            })
        );
        assert_eq!(
            Usage::from_value(&json!({"output_tokens": 9})),
            Some(Usage {
                prompt: 0,
                completion: 9
            })
        );
        assert_eq!(Usage::from_value(&json!({})), None);
    }

    #[test]
    fn plain_text_of_content() {
        assert_eq!(plain_text(&json!("hi")).as_deref(), Some("hi"));
        assert_eq!(
            plain_text(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]))
                .as_deref(),
            Some("a\nb")
        );
        assert_eq!(plain_text(&json!([{"type": "image_url"}])), None);
    }
}
