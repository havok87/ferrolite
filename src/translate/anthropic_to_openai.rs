//! Anthropic Messages API in, OpenAI chat completions out, so
//! Anthropic-format clients can use OpenAI-compatible (often local)
//! backends: the request, then the response or stream back.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::sse::{SseEvent, write_event};
use super::{TranslateError, Usage, check_params, plain_text};
use crate::secrets::Redactor;

fn supported(key: &str, value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match key {
        "model" | "messages" | "system" | "max_tokens" | "temperature" | "top_p" | "top_k"
        | "stop_sequences" | "stream" | "tools" | "tool_choice" | "metadata" => true,
        "thinking" => value.get("type").and_then(Value::as_str) == Some("disabled"),
        _ => false,
    }
}

fn err(msg: impl Into<String>) -> TranslateError {
    TranslateError::new(msg)
}

/// Translates a Messages API body (with `model` already set to the
/// upstream model) into a chat completions body.
pub fn request(body: &Map<String, Value>, drop_params: bool) -> Result<Value, TranslateError> {
    let dropped = check_params(body, supported, drop_params, "OpenAI-compatible backends")?;
    let get = |k: &str| {
        body.get(k)
            .filter(|v| !v.is_null())
            .filter(|_| !dropped.contains(&k))
    };

    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = get("system") {
        let text = plain_text(system).ok_or_else(|| err("`system` must be text"))?;
        messages.push(json!({"role": "system", "content": text}));
    }
    let input = get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| err("`messages` must be a list"))?;
    for (i, m) in input.iter().enumerate() {
        convert_message(i, m, &mut messages)?;
    }

    let mut out = Map::new();
    out.insert("model".into(), get("model").cloned().unwrap_or(Value::Null));
    out.insert("messages".into(), Value::Array(messages));
    for key in ["max_tokens", "temperature", "top_p", "top_k"] {
        if let Some(v) = get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    if let Some(stop) = get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    if get("stream").and_then(Value::as_bool) == Some(true) {
        out.insert("stream".into(), true.into());
        out.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(user) = get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str)
    {
        out.insert("user".into(), user.into());
    }
    if let Some(tools) = get("tools") {
        out.insert("tools".into(), convert_tools(tools)?);
    }
    if let Some(choice) = get("tool_choice") {
        let kind = choice.get("type").and_then(Value::as_str);
        let mapped = match kind {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") => {
                let name = choice
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| err("`tool_choice.name` is required"))?;
                json!({"type": "function", "function": {"name": name}})
            }
            _ => return Err(err("`tool_choice.type` must be auto, any, none or tool")),
        };
        out.insert("tool_choice".into(), mapped);
        if choice
            .get("disable_parallel_tool_use")
            .and_then(Value::as_bool)
            == Some(true)
        {
            out.insert("parallel_tool_calls".into(), false.into());
        }
    }
    Ok(Value::Object(out))
}

fn convert_tools(tools: &Value) -> Result<Value, TranslateError> {
    let list = tools
        .as_array()
        .ok_or_else(|| err("`tools` must be a list"))?;
    list.iter()
        .enumerate()
        .map(|(i, t)| {
            if t.get("type").is_some_and(|k| k != "custom") {
                return Err(err(format!(
                    "tools[{i}]: server tools are not supported by OpenAI-compatible backends"
                )));
            }
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| err(format!("tools[{i}].name is required")))?;
            let mut function = json!({
                "name": name,
                "parameters": t.get("input_schema").cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            });
            if let Some(d) = t.get("description").and_then(Value::as_str) {
                function["description"] = d.into();
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn convert_message(i: usize, m: &Value, out: &mut Vec<Value>) -> Result<(), TranslateError> {
    let role = m
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| err(format!("messages[{i}].role is required")))?;
    let content = m.get("content").unwrap_or(&Value::Null);
    let blocks: Vec<Value> = match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(b) => b.clone(),
        _ => {
            return Err(err(format!(
                "messages[{i}].content must be a string or a list"
            )));
        }
    };
    match role {
        "user" => {
            let mut parts = Vec::new();
            for (j, b) in blocks.iter().enumerate() {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => parts.push(
                        json!({"type": "text", "text": b.get("text").cloned().unwrap_or_default()}),
                    ),
                    Some("image") => parts.push(image_part(i, j, b)?),
                    // Tool results become `tool` messages, which must come
                    // straight after the assistant's tool calls.
                    Some("tool_result") => {
                        let id = b
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                err(format!(
                                    "messages[{i}].content[{j}].tool_use_id is required"
                                ))
                            })?;
                        let text = plain_text(b.get("content").unwrap_or(&Value::Null)).ok_or_else(|| {
                            err(format!(
                                "messages[{i}].content[{j}]: tool results must contain only text for OpenAI-compatible backends"
                            ))
                        })?;
                        out.push(json!({"role": "tool", "tool_call_id": id, "content": text}));
                    }
                    Some(other) => {
                        return Err(err(format!(
                            "messages[{i}].content[{j}]: `{other}` blocks are not supported by OpenAI-compatible backends"
                        )));
                    }
                    None => {
                        return Err(err(format!("messages[{i}].content[{j}].type is required")));
                    }
                }
            }
            if parts.is_empty() {
                return Ok(());
            }
            let content = if parts.iter().all(|p| p["type"] == "text") {
                let texts: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
                Value::String(texts.join("\n"))
            } else {
                Value::Array(parts)
            };
            out.push(json!({"role": "user", "content": content}));
        }
        "assistant" => {
            let mut text = String::new();
            let mut calls = Vec::new();
            for (j, b) in blocks.iter().enumerate() {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("tool_use") => calls.push(json!({
                        "id": b.get("id").cloned().unwrap_or_default(),
                        "type": "function",
                        "function": {
                            "name": b.get("name").cloned().unwrap_or_default(),
                            "arguments": b.get("input").map_or_else(|| "{}".to_owned(), Value::to_string),
                        },
                    })),
                    // Earlier reasoning is not replayed to a different backend.
                    Some("thinking" | "redacted_thinking") => {}
                    Some(other) => {
                        return Err(err(format!(
                            "messages[{i}].content[{j}]: `{other}` blocks are not supported by OpenAI-compatible backends"
                        )));
                    }
                    None => return Err(err(format!("messages[{i}].content[{j}].type is required"))),
                }
            }
            let mut msg = json!({
                "role": "assistant",
                "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { text.into() },
            });
            if !calls.is_empty() {
                msg["tool_calls"] = Value::Array(calls);
            }
            out.push(msg);
        }
        other => {
            return Err(err(format!(
                "messages[{i}]: role `{other}` must be user or assistant"
            )));
        }
    }
    Ok(())
}

fn image_part(i: usize, j: usize, block: &Value) -> Result<Value, TranslateError> {
    let source = &block["source"];
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("");
            let data = source.get("data").and_then(Value::as_str).unwrap_or("");
            format!("data:{media};base64,{data}")
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .filter(|u| u.starts_with("https://") || u.starts_with("http://"))
            .ok_or_else(|| {
                err(format!(
                    "messages[{i}].content[{j}].source.url must be http(s)"
                ))
            })?
            .to_owned(),
        _ => {
            return Err(err(format!(
                "messages[{i}].content[{j}].source.type must be base64 or url"
            )));
        }
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Maps OpenAI's `finish_reason` to Anthropic's `stop_reason`.
pub fn stop_reason(finish_reason: Option<&str>) -> &'static str {
    match finish_reason {
        Some("length") => "max_tokens",
        Some("tool_calls" | "function_call") => "tool_use",
        Some("content_filter") => "refusal",
        _ => "end_turn",
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct OpenAiUsage {
    prompt: u64,
    completion: u64,
    cached: u64,
}

impl OpenAiUsage {
    fn update(&mut self, u: &Value) {
        if let Some(p) = u.get("prompt_tokens").and_then(Value::as_u64) {
            self.prompt = p;
        }
        if let Some(c) = u.get("completion_tokens").and_then(Value::as_u64) {
            self.completion = c;
        }
        if let Some(c) = u
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
        {
            self.cached = c;
        }
    }

    fn usage(self) -> Usage {
        Usage {
            prompt: self.prompt,
            completion: self.completion,
        }
    }

    /// Anthropic counts cache reads separately from `input_tokens`.
    fn to_anthropic(self) -> Value {
        json!({
            "input_tokens": self.prompt.saturating_sub(self.cached),
            "output_tokens": self.completion,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": self.cached,
        })
    }
}

fn message_id(openai_id: Option<&str>) -> String {
    match openai_id.filter(|s| !s.is_empty()) {
        Some(id) => format!("msg_{id}"),
        None => format!("msg_{:x}", super::unix_now()),
    }
}

/// Translates a chat completion into a Messages API response.
pub fn response(completion: &Value) -> Option<(Value, Usage)> {
    let choice = completion.get("choices")?.as_array()?.first()?;
    let message = choice.get("message")?;
    let mut content = Vec::new();
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty())
    {
        content.push(json!({"type": "text", "text": text}));
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let args = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        content.push(json!({
            "type": "tool_use",
            "id": call.get("id").cloned().unwrap_or_default(),
            "name": call.pointer("/function/name").cloned().unwrap_or_default(),
            "input": serde_json::from_str::<Value>(args).ok().filter(Value::is_object).unwrap_or_else(|| json!({})),
        }));
    }
    let mut usage = OpenAiUsage::default();
    if let Some(u) = completion.get("usage") {
        usage.update(u);
    }
    let out = json!({
        "id": message_id(completion.get("id").and_then(Value::as_str)),
        "type": "message",
        "role": "assistant",
        "model": completion.get("model").cloned().unwrap_or_default(),
        "content": content,
        "stop_reason": stop_reason(choice.get("finish_reason").and_then(Value::as_str)),
        "stop_sequence": null,
        "usage": usage.to_anthropic(),
    });
    Some((out, usage.usage()))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Open {
    Text(u64),
    Tool(u64),
}

/// Turns chat completion chunks into an Anthropic event stream.
pub struct StreamTranslator {
    scrub: Redactor,
    started: bool,
    open: Option<Open>,
    next_index: u64,
    /// OpenAI tool call index to Anthropic content block index.
    tools: HashMap<u64, u64>,
    stop_reason: Option<&'static str>,
    usage: OpenAiUsage,
    done: bool,
    failed: bool,
}

impl StreamTranslator {
    pub fn new(scrub: Redactor) -> Self {
        Self {
            scrub,
            started: false,
            open: None,
            next_index: 0,
            tools: HashMap::new(),
            stop_reason: None,
            usage: OpenAiUsage::default(),
            done: false,
            failed: false,
        }
    }

    /// Whether the stream ended with an error event.
    pub fn failed(&self) -> bool {
        self.failed
    }

    pub fn usage(&self) -> Usage {
        self.usage.usage()
    }

    fn emit(out: &mut Vec<u8>, data: &Value) {
        let kind = data["type"].as_str().unwrap_or("");
        write_event(out, Some(kind), &data.to_string());
    }

    fn start(&mut self, out: &mut Vec<u8>, chunk: &Value) {
        if self.started {
            return;
        }
        self.started = true;
        Self::emit(
            out,
            &json!({"type": "message_start", "message": {
                "id": message_id(chunk.get("id").and_then(Value::as_str)),
                "type": "message",
                "role": "assistant",
                "model": chunk.get("model").cloned().unwrap_or_default(),
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }}),
        );
    }

    fn close(&mut self, out: &mut Vec<u8>) {
        if let Some(Open::Text(i) | Open::Tool(i)) = self.open.take() {
            Self::emit(out, &json!({"type": "content_block_stop", "index": i}));
        }
    }

    fn open_block(&mut self, out: &mut Vec<u8>, block: Value) -> u64 {
        self.close(out);
        let index = self.next_index;
        self.next_index += 1;
        Self::emit(
            out,
            &json!({"type": "content_block_start", "index": index, "content_block": block}),
        );
        index
    }

    pub fn event(&mut self, ev: &SseEvent, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        if ev.data.trim() == "[DONE]" {
            self.complete(out);
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&ev.data) else {
            return;
        };
        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .unwrap_or("upstream stream error")
                .to_owned();
            self.fail(out, &message);
            return;
        }
        self.start(out, &chunk);
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage.update(u);
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            return;
        };
        let delta = &choice["delta"];
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
        {
            let index = match self.open {
                Some(Open::Text(i)) => i,
                _ => {
                    let i = self.open_block(out, json!({"type": "text", "text": ""}));
                    self.open = Some(Open::Text(i));
                    i
                }
            };
            Self::emit(
                out,
                &json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text}}),
            );
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let tool = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let index = match self.tools.get(&tool) {
                Some(&i) => i,
                None => {
                    let id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .map_or_else(|| format!("toolu_{tool}"), str::to_owned);
                    let name = call.pointer("/function/name").cloned().unwrap_or_default();
                    let i = self.open_block(
                        out,
                        json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                    );
                    self.open = Some(Open::Tool(i));
                    self.tools.insert(tool, i);
                    i
                }
            };
            if let Some(args) = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .filter(|a| !a.is_empty())
            {
                Self::emit(
                    out,
                    &json!({"type": "content_block_delta", "index": index,
                            "delta": {"type": "input_json_delta", "partial_json": args}}),
                );
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.stop_reason = Some(stop_reason(Some(reason)));
        }
    }

    fn complete(&mut self, out: &mut Vec<u8>) {
        self.start(out, &Value::Null);
        self.close(out);
        let mut usage = self.usage.to_anthropic();
        if let Some(map) = usage.as_object_mut() {
            map.remove("cache_creation_input_tokens");
        }
        Self::emit(
            out,
            &json!({"type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason.unwrap_or("end_turn"), "stop_sequence": null},
                    "usage": usage}),
        );
        Self::emit(out, &json!({"type": "message_stop"}));
        self.done = true;
    }

    fn fail(&mut self, out: &mut Vec<u8>, message: &str) {
        Self::emit(
            out,
            &json!({"type": "error", "error": {"type": "api_error", "message": self.scrub.redact(message)}}),
        );
        self.done = true;
        self.failed = true;
    }

    /// Ends the stream. A stream that stops after its finish reason but
    /// without `[DONE]` is completed normally; anything else is an error.
    pub fn finish(&mut self, error: Option<&str>, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        match error {
            Some(message) => self.fail(out, message),
            None if self.stop_reason.is_some() => self.complete(out),
            None => self.fail(out, "upstream stream ended before it was complete"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::translate::sse::SseParser;

    fn map(v: Value) -> Map<String, Value> {
        match v {
            Value::Object(m) => m,
            _ => panic!("not an object"),
        }
    }

    fn translate(body: Value) -> Result<Value, TranslateError> {
        request(&map(body), false)
    }

    #[test]
    fn request_translation() {
        let out = translate(json!({
            "model": "qwen3",
            "system": [{"type": "text", "text": "Be brief."}],
            "max_tokens": 512,
            "temperature": 0.2,
            "top_k": 40,
            "stop_sequences": ["END"],
            "stream": true,
            "metadata": {"user_id": "u-1"},
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR"}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "Looking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "cat"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a cat"}]},
                    {"type": "text", "text": "Thanks"},
                ]},
                {"role": "user", "content": "And?"},
            ],
            "tools": [{"name": "lookup", "description": "Look up", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
        }))
        .unwrap();
        assert_eq!(
            out,
            json!({
                "model": "qwen3",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
                    ]},
                    {"role": "assistant", "content": "Looking.", "tool_calls": [
                        {"id": "toolu_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"cat\"}"}},
                    ]},
                    {"role": "tool", "tool_call_id": "toolu_1", "content": "a cat"},
                    {"role": "user", "content": "Thanks"},
                    {"role": "user", "content": "And?"},
                ],
                "max_tokens": 512,
                "temperature": 0.2,
                "top_k": 40,
                "stop": ["END"],
                "stream": true,
                "stream_options": {"include_usage": true},
                "user": "u-1",
                "tools": [{"type": "function", "function": {
                    "name": "lookup", "description": "Look up", "parameters": {"type": "object"},
                }}],
                "tool_choice": "required",
                "parallel_tool_calls": false,
            })
        );
    }

    #[test]
    fn tool_choice_mapping() {
        for (choice, expected) in [
            (json!({"type": "auto"}), json!("auto")),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "f"}),
                json!({"type": "function", "function": {"name": "f"}}),
            ),
        ] {
            let out =
                translate(json!({"model": "m", "messages": [], "tool_choice": choice})).unwrap();
            assert_eq!(out["tool_choice"], expected);
            assert!(out.get("parallel_tool_calls").is_none());
        }
    }

    #[test]
    fn rejects_what_openai_backends_cannot_take() {
        for (body, needle) in [
            (
                json!({"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]}),
                "`document` blocks",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "file:///x"}}]}]}),
                "must be http(s)",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image"}]}]}]}),
                "only text",
            ),
            (
                json!({"messages": [{"role": "system", "content": "x"}]}),
                "user or assistant",
            ),
            (
                json!({"messages": [], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
                "server tools",
            ),
        ] {
            let mut b = map(body);
            b.insert("model".into(), "m".into());
            let e = request(&b, false).unwrap_err();
            assert!(e.0.contains(needle), "{needle}: {e}");
        }
    }

    #[test]
    fn thinking_and_unknown_params() {
        let body = map(
            json!({"model": "m", "messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}, "service_tier": "auto"}),
        );
        let e = request(&body, false).unwrap_err();
        assert!(e.0.contains("service_tier, thinking"), "{e}");
        assert!(request(&body, true).unwrap().get("thinking").is_none());
        let disabled = map(json!({"model": "m", "messages": [], "thinking": {"type": "disabled"}}));
        assert!(request(&disabled, false).is_ok());
    }

    #[test]
    fn response_translation() {
        let (out, usage) = response(&json!({
            "id": "chatcmpl-1", "object": "chat.completion", "model": "qwen3",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Calling.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"cat\"}"}}],
            }}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 5}},
        }))
        .unwrap();
        assert_eq!(
            usage,
            Usage {
                prompt: 20,
                completion: 8
            }
        );
        assert_eq!(
            out,
            json!({
                "id": "msg_chatcmpl-1", "type": "message", "role": "assistant", "model": "qwen3",
                "content": [
                    {"type": "text", "text": "Calling."},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "cat"}},
                ],
                "stop_reason": "tool_use", "stop_sequence": null,
                "usage": {"input_tokens": 15, "output_tokens": 8, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5},
            })
        );
        for (finish, stop) in [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("content_filter", "refusal"),
        ] {
            let (out, _) = response(
                &json!({"choices": [{"message": {"content": "x"}, "finish_reason": finish}]}),
            )
            .unwrap();
            assert_eq!(out["stop_reason"], stop);
        }
        assert!(response(&json!({"choices": []})).is_none());
    }

    pub(crate) const OPENAI_STREAM: &str = concat!(
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"q\\\":\"}}]}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"cat\\\"}\"}}]}}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"id\":\"c1\",\"model\":\"qwen3\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":9}}\n\n",
        "data: [DONE]\n\n",
    );

    fn run(input: &str, split: usize) -> (Vec<(String, Value)>, Usage) {
        let mut parser = SseParser::default();
        let mut t = StreamTranslator::new(Redactor::default());
        let mut out = Vec::new();
        for part in [&input.as_bytes()[..split], &input.as_bytes()[split..]] {
            for ev in parser.push(part).unwrap() {
                t.event(&ev, &mut out);
            }
        }
        t.finish(None, &mut out);
        let events = SseParser::default()
            .push(&out)
            .unwrap()
            .into_iter()
            .map(|e| {
                (
                    e.event.unwrap_or_default(),
                    serde_json::from_str(&e.data).unwrap(),
                )
            })
            .collect();
        (events, t.usage())
    }

    #[test]
    fn stream_translation() {
        let (events, usage) = run(OPENAI_STREAM, 0);
        assert_eq!(
            usage,
            Usage {
                prompt: 11,
                completion: 9
            }
        );
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        for (name, data) in &events {
            assert_eq!(data["type"], name.as_str());
        }
        assert_eq!(events[0].1["message"]["id"], "msg_c1");
        assert_eq!(events[0].1["message"]["model"], "qwen3");
        assert_eq!(
            events[1].1["content_block"],
            json!({"type": "text", "text": ""})
        );
        assert_eq!(
            events[2].1["delta"],
            json!({"type": "text_delta", "text": "Hel"})
        );
        assert_eq!(
            events[5].1,
            json!({"type": "content_block_start", "index": 1,
                   "content_block": {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}})
        );
        assert_eq!(
            events[6].1["delta"],
            json!({"type": "input_json_delta", "partial_json": "{\"q\":"})
        );
        assert_eq!(events[8].1["index"], 1);
        assert_eq!(
            events[9].1,
            json!({"type": "message_delta",
                   "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                   "usage": {"input_tokens": 11, "output_tokens": 9, "cache_read_input_tokens": 0}})
        );
    }

    #[test]
    fn stream_translation_is_independent_of_chunking() {
        let (whole, _) = run(OPENAI_STREAM, 0);
        for split in (0..OPENAI_STREAM.len()).step_by(5) {
            assert_eq!(run(OPENAI_STREAM, split).0, whole, "split at {split}");
        }
    }

    #[test]
    fn stream_endings() {
        // No [DONE] but a finish reason: completed normally.
        let no_done = OPENAI_STREAM.replace("data: [DONE]\n\n", "");
        let (events, _) = run(&no_done, 0);
        assert_eq!(events.last().unwrap().0, "message_stop");

        // Cut off mid-answer: an error event, no message_stop.
        let cut = &OPENAI_STREAM[..OPENAI_STREAM.find("{\"index\":0,\"delta\":{},").unwrap() - 40];
        let cut = &cut[..cut.rfind("\n\n").unwrap() + 2];
        let (events, _) = run(cut, 0);
        assert_eq!(events.last().unwrap().0, "error");
        assert!(events.iter().all(|(n, _)| n != "message_stop"));

        // Upstream error chunk.
        let (events, _) = run("data: {\"error\":{\"message\":\"model crashed\"}}\n\n", 0);
        assert_eq!(
            events,
            [(
                "error".to_owned(),
                json!({"type": "error", "error": {"type": "api_error", "message": "model crashed"}})
            )]
        );

        // Empty stream that still says [DONE].
        let (events, _) = run("data: [DONE]\n\n", 0);
        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["message_start", "message_delta", "message_stop"]);
    }
}
