//! OpenAI chat completions in, Anthropic Messages API out: the request,
//! then the response or stream back into chat format.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use super::sse::{SseEvent, write_event};
use super::{TranslateError, Usage, check_params, plain_text, unix_now};
use crate::config::ModelInfo;
use crate::secrets::Redactor;

/// `max_tokens` when neither the request nor `model_info` gives one.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;
const IMAGE_TYPES: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];

fn supported(key: &str, value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    match key {
        "model"
        | "messages"
        | "max_tokens"
        | "max_completion_tokens"
        | "temperature"
        | "top_p"
        | "top_k"
        | "stop"
        | "stream"
        | "stream_options"
        | "tools"
        | "tool_choice"
        | "parallel_tool_calls"
        | "user"
        | "reasoning_effort" => true,
        "n" => value.as_u64() == Some(1),
        "response_format" => value.get("type").and_then(Value::as_str) == Some("text"),
        _ => false,
    }
}

fn err(msg: impl Into<String>) -> TranslateError {
    TranslateError::new(msg)
}

/// Translates a chat completions body (with `model` already set to the
/// upstream model) into a Messages API body.
pub fn request(
    body: &Map<String, Value>,
    model_info: &ModelInfo,
    drop_params: bool,
) -> Result<Value, TranslateError> {
    let mut body = body.clone();
    check_params(&mut body, supported, drop_params, "anthropic")?;
    let get = |k: &str| body.get(k).filter(|v| !v.is_null());

    let messages = get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| err("`messages` must be a list"))?;
    let (system, messages) = convert_messages(messages)?;

    let mut out = Map::new();
    out.insert("model".into(), get("model").cloned().unwrap_or(Value::Null));
    if let Some(system) = system {
        out.insert("system".into(), Value::String(system));
    }
    out.insert("messages".into(), Value::Array(messages));

    let requested = match get("max_completion_tokens").or_else(|| get("max_tokens")) {
        Some(v) => Some(
            v.as_u64()
                .ok_or_else(|| err("`max_tokens` must be a non-negative integer"))?,
        ),
        None => None,
    };
    let mut max_tokens = requested
        .or(model_info.max_output_tokens)
        .or(model_info.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);

    if let Some(effort) = get("reasoning_effort") {
        let budget = match effort.as_str() {
            Some("low") => Some(1024),
            Some("medium") => Some(2048),
            Some("high") => Some(4096),
            Some("minimal" | "none") => None,
            _ => {
                return Err(err(
                    "`reasoning_effort` must be one of none, minimal, low, medium, high",
                ));
            }
        };
        if let Some(budget) = budget {
            // Anthropic requires max_tokens > budget_tokens; keep the
            // requested room for the visible answer on top of the budget.
            if max_tokens <= budget {
                max_tokens += budget;
            }
            out.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
    }
    out.insert("max_tokens".into(), max_tokens.into());

    for key in ["temperature", "top_p", "top_k"] {
        if let Some(v) = get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    match get("stop") {
        None => {}
        Some(Value::String(s)) => {
            out.insert("stop_sequences".into(), json!([s]));
        }
        Some(Value::Array(list)) if list.iter().all(Value::is_string) => {
            out.insert("stop_sequences".into(), Value::Array(list.clone()));
        }
        Some(_) => return Err(err("`stop` must be a string or a list of strings")),
    }
    if get("stream").and_then(Value::as_bool) == Some(true) {
        out.insert("stream".into(), true.into());
    }
    if let Some(user) = get("user") {
        let user = user
            .as_str()
            .ok_or_else(|| err("`user` must be a string"))?;
        out.insert("metadata".into(), json!({"user_id": user}));
    }
    if let Some(tools) = get("tools") {
        out.insert("tools".into(), convert_tools(tools)?);
    }
    let mut tool_choice = match get("tool_choice") {
        None => None,
        Some(Value::String(s)) => Some(match s.as_str() {
            "auto" => json!({"type": "auto"}),
            "required" => json!({"type": "any"}),
            "none" => json!({"type": "none"}),
            _ => {
                return Err(err(
                    "`tool_choice` must be auto, required, none or a function",
                ));
            }
        }),
        Some(v) => {
            let name = v
                .pointer("/function/name")
                .and_then(Value::as_str)
                .ok_or_else(|| err("`tool_choice.function.name` is required"))?;
            Some(json!({"type": "tool", "name": name}))
        }
    };
    if get("parallel_tool_calls").and_then(Value::as_bool) == Some(false) {
        let choice = tool_choice.get_or_insert_with(|| json!({"type": "auto"}));
        if choice["type"] != "none" {
            choice["disable_parallel_tool_use"] = true.into();
        }
    }
    if let Some(choice) = tool_choice {
        out.insert("tool_choice".into(), choice);
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
            if t.get("type").and_then(Value::as_str) != Some("function") {
                return Err(err(format!(
                    "tools[{i}]: only function tools are supported by anthropic"
                )));
            }
            let f = t
                .get("function")
                .ok_or_else(|| err(format!("tools[{i}].function is required")))?;
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| err(format!("tools[{i}].function.name is required")))?;
            let mut tool = json!({
                "name": name,
                "input_schema": f.get("parameters").filter(|p| !p.is_null()).cloned()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
            });
            if let Some(d) = f.get("description").and_then(Value::as_str) {
                tool["description"] = d.into();
            }
            Ok(tool)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

/// System text plus Anthropic messages, with consecutive same-role
/// messages merged (Anthropic requires roles to alternate).
fn convert_messages(messages: &[Value]) -> Result<(Option<String>, Vec<Value>), TranslateError> {
    let mut system: Vec<String> = Vec::new();
    let mut out: Vec<(&'static str, Vec<Value>)> = Vec::new();
    let mut push = |role: &'static str, blocks: Vec<Value>| {
        if blocks.is_empty() {
            return;
        }
        match out.last_mut() {
            Some((last, existing)) if *last == role => existing.extend(blocks),
            _ => out.push((role, blocks)),
        }
    };

    for (i, m) in messages.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| err(format!("messages[{i}].role is required")))?;
        let content = m.get("content").unwrap_or(&Value::Null);
        match role {
            "system" | "developer" => system.push(plain_text(content).ok_or_else(|| {
                err(format!(
                    "messages[{i}]: system messages must contain only text"
                ))
            })?),
            "user" => push("user", user_blocks(i, content)?),
            "assistant" => push("assistant", assistant_blocks(i, m, content)?),
            "tool" => {
                let id = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| err(format!("messages[{i}].tool_call_id is required")))?;
                let text = plain_text(content).ok_or_else(|| {
                    err(format!(
                        "messages[{i}]: tool messages must contain only text"
                    ))
                })?;
                push(
                    "user",
                    vec![json!({"type": "tool_result", "tool_use_id": id, "content": text})],
                );
            }
            other => {
                return Err(err(format!(
                    "messages[{i}]: role `{other}` is not supported by anthropic"
                )));
            }
        }
    }
    let system = (!system.is_empty()).then(|| system.join("\n"));
    let messages = out
        .into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    Ok((system, messages))
}

fn user_blocks(i: usize, content: &Value) -> Result<Vec<Value>, TranslateError> {
    match content {
        Value::String(s) => Ok(vec![json!({"type": "text", "text": s})]),
        Value::Array(parts) => parts
            .iter()
            .enumerate()
            .map(|(j, part)| match part.get("type").and_then(Value::as_str) {
                Some("text") => Ok(json!({"type": "text", "text": part.get("text").cloned().unwrap_or_default()})),
                Some("image_url") => {
                    let url = part
                        .get("image_url")
                        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))
                        .ok_or_else(|| err(format!("messages[{i}].content[{j}].image_url.url is required")))?;
                    image_block(i, j, url)
                }
                Some(other) => Err(err(format!(
                    "messages[{i}].content[{j}]: content type `{other}` is not supported by anthropic"
                ))),
                None => Err(err(format!("messages[{i}].content[{j}].type is required"))),
            })
            .collect(),
        Value::Null => Ok(Vec::new()),
        _ => Err(err(format!("messages[{i}].content must be a string or a list"))),
    }
}

/// `data:` URLs become base64 sources; http(s) URLs become URL sources that
/// Anthropic fetches. The proxy never fetches anything itself.
fn image_block(i: usize, j: usize, url: &str) -> Result<Value, TranslateError> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (media_type, data) = rest.split_once(";base64,").ok_or_else(|| {
            err(format!(
                "messages[{i}].content[{j}]: data URLs must be base64-encoded"
            ))
        })?;
        if !IMAGE_TYPES.contains(&media_type) {
            return Err(err(format!(
                "messages[{i}].content[{j}]: image type `{media_type}` is not supported \
                 (use jpeg, png, gif or webp)"
            )));
        }
        return Ok(json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }));
    }
    if url.starts_with("https://") || url.starts_with("http://") {
        return Ok(json!({"type": "image", "source": {"type": "url", "url": url}}));
    }
    Err(err(format!(
        "messages[{i}].content[{j}]: image URLs must be data:, http:// or https://"
    )))
}

fn assistant_blocks(
    i: usize,
    message: &Value,
    content: &Value,
) -> Result<Vec<Value>, TranslateError> {
    let mut blocks = Vec::new();
    match content {
        Value::String(s) if !s.is_empty() => blocks.push(json!({"type": "text", "text": s})),
        Value::Array(parts) => {
            for p in parts {
                if let Some(text) = p
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|t| !t.is_empty())
                {
                    blocks.push(json!({"type": "text", "text": text}));
                }
            }
        }
        _ => {}
    }
    let calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (j, call) in calls.iter().enumerate() {
        let path = format!("messages[{i}].tool_calls[{j}]");
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| err(format!("{path}.id is required")))?;
        let name = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .ok_or_else(|| err(format!("{path}.function.name is required")))?;
        let args = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        let input: Value = if args.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(args)
                .map_err(|_| err(format!("{path}.function.arguments must be valid JSON")))?
        };
        if !input.is_object() {
            return Err(err(format!(
                "{path}.function.arguments must be a JSON object"
            )));
        }
        blocks.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
    }
    Ok(blocks)
}

/// Maps Anthropic's `stop_reason` to OpenAI's `finish_reason`.
pub fn finish_reason(stop_reason: Option<&str>) -> &'static str {
    match stop_reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        Some("refusal") => "content_filter",
        _ => "stop",
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AnthropicUsage {
    input: u64,
    output: u64,
    cache_creation: u64,
    cache_read: u64,
}

impl AnthropicUsage {
    fn update(&mut self, u: &Value) {
        let n = |k: &str| u.get(k).and_then(Value::as_u64);
        if let Some(v) = n("input_tokens") {
            self.input = v;
        }
        if let Some(v) = n("output_tokens") {
            self.output = v;
        }
        if let Some(v) = n("cache_creation_input_tokens") {
            self.cache_creation = v;
        }
        if let Some(v) = n("cache_read_input_tokens") {
            self.cache_read = v;
        }
    }

    fn usage(self) -> Usage {
        Usage {
            prompt: self.input + self.cache_creation + self.cache_read,
            completion: self.output,
        }
    }

    fn to_openai(self) -> Value {
        let u = self.usage();
        json!({
            "prompt_tokens": u.prompt,
            "completion_tokens": u.completion,
            "total_tokens": u.total(),
            "prompt_tokens_details": {"cached_tokens": self.cache_read},
            "cache_creation_input_tokens": self.cache_creation,
            "cache_read_input_tokens": self.cache_read,
        })
    }
}

/// Translates a Messages API response into a chat completion.
pub fn response(message: &Value) -> Option<(Value, Usage)> {
    let blocks = message.get("content")?.as_array()?;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("thinking") => {
                reasoning.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or(""));
            }
            Some("tool_use") => tool_calls.push(json!({
                "id": b.get("id").cloned().unwrap_or_default(),
                "type": "function",
                "function": {
                    "name": b.get("name").cloned().unwrap_or_default(),
                    "arguments": b.get("input").map_or_else(|| "{}".to_owned(), Value::to_string),
                },
            })),
            _ => {}
        }
    }
    let mut msg = json!({
        "role": "assistant",
        "content": if text.is_empty() && !tool_calls.is_empty() { Value::Null } else { text.into() },
    });
    if !tool_calls.is_empty() {
        msg["tool_calls"] = Value::Array(tool_calls);
    }
    if !reasoning.is_empty() {
        msg["reasoning_content"] = reasoning.into();
    }
    let mut usage = AnthropicUsage::default();
    if let Some(u) = message.get("usage") {
        usage.update(u);
    }
    let out = json!({
        "id": message.get("id").cloned().unwrap_or_default(),
        "object": "chat.completion",
        "created": unix_now(),
        "model": message.get("model").cloned().unwrap_or_default(),
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": finish_reason(message.get("stop_reason").and_then(Value::as_str)),
        }],
        "usage": usage.to_openai(),
    });
    Some((out, usage.usage()))
}

#[derive(Debug, Clone, Copy)]
enum Block {
    Text,
    Tool(u64),
    Thinking,
    Other,
}

/// Turns an Anthropic event stream into chat completion chunks.
pub struct StreamTranslator {
    include_usage: bool,
    scrub: Redactor,
    id: String,
    model: String,
    created: u64,
    blocks: HashMap<u64, Block>,
    next_tool: u64,
    usage: AnthropicUsage,
    finish: Option<&'static str>,
    done: bool,
}

impl StreamTranslator {
    /// `include_usage` mirrors the client's `stream_options.include_usage`.
    pub fn new(include_usage: bool, scrub: Redactor) -> Self {
        Self {
            include_usage,
            scrub,
            id: String::new(),
            model: String::new(),
            created: unix_now(),
            blocks: HashMap::new(),
            next_tool: 0,
            usage: AnthropicUsage::default(),
            finish: None,
            done: false,
        }
    }

    pub fn usage(&self) -> Usage {
        self.usage.usage()
    }

    fn chunk(&self, out: &mut Vec<u8>, delta: Value, finish: Option<&str>) {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        write_event(out, None, &chunk.to_string());
    }

    fn fail(&mut self, out: &mut Vec<u8>, kind: &str, message: &str) {
        let error = json!({"error": {"message": self.scrub.redact(message), "type": kind}});
        write_event(out, None, &error.to_string());
        write_event(out, None, "[DONE]");
        self.done = true;
    }

    pub fn event(&mut self, ev: &SseEvent, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        let Ok(data) = serde_json::from_str::<Value>(&ev.data) else {
            return;
        };
        let kind = data
            .get("type")
            .and_then(Value::as_str)
            .or(ev.event.as_deref())
            .unwrap_or("");
        let index = data.get("index").and_then(Value::as_u64).unwrap_or(0);
        match kind {
            "message_start" => {
                let m = &data["message"];
                self.id = m.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
                self.model = m
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                if let Some(u) = m.get("usage") {
                    self.usage.update(u);
                }
                self.chunk(out, json!({"role": "assistant", "content": ""}), None);
            }
            "content_block_start" => {
                let block = &data["content_block"];
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        self.blocks.insert(index, Block::Text);
                        if let Some(t) = block
                            .get("text")
                            .and_then(Value::as_str)
                            .filter(|t| !t.is_empty())
                        {
                            self.chunk(out, json!({"content": t}), None);
                        }
                    }
                    Some("tool_use") => {
                        let tool = self.next_tool;
                        self.next_tool += 1;
                        self.blocks.insert(index, Block::Tool(tool));
                        self.chunk(
                            out,
                            json!({"tool_calls": [{
                                "index": tool,
                                "id": block.get("id").cloned().unwrap_or_default(),
                                "type": "function",
                                "function": {"name": block.get("name").cloned().unwrap_or_default(), "arguments": ""},
                            }]}),
                            None,
                        );
                    }
                    Some("thinking") => {
                        self.blocks.insert(index, Block::Thinking);
                    }
                    _ => {
                        self.blocks.insert(index, Block::Other);
                    }
                }
            }
            "content_block_delta" => {
                let delta = &data["delta"];
                let block = self.blocks.get(&index).copied().unwrap_or(Block::Other);
                match (delta.get("type").and_then(Value::as_str), block) {
                    (Some("text_delta"), Block::Text) => {
                        let t = delta.get("text").cloned().unwrap_or_default();
                        self.chunk(out, json!({"content": t}), None);
                    }
                    (Some("input_json_delta"), Block::Tool(tool)) => {
                        let partial = delta.get("partial_json").cloned().unwrap_or_default();
                        self.chunk(
                            out,
                            json!({"tool_calls": [{"index": tool, "function": {"arguments": partial}}]}),
                            None,
                        );
                    }
                    (Some("thinking_delta"), Block::Thinking) => {
                        let t = delta.get("thinking").cloned().unwrap_or_default();
                        self.chunk(out, json!({"reasoning_content": t}), None);
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(u) = data.get("usage") {
                    self.usage.update(u);
                }
                let stop = data.pointer("/delta/stop_reason").and_then(Value::as_str);
                if stop.is_some() {
                    let finish = finish_reason(stop);
                    self.finish = Some(finish);
                    self.chunk(out, json!({}), Some(finish));
                }
            }
            "message_stop" => {
                if self.finish.is_none() {
                    self.chunk(out, json!({}), Some("stop"));
                }
                if self.include_usage {
                    let chunk = json!({
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": self.usage.to_openai(),
                    });
                    write_event(out, None, &chunk.to_string());
                }
                write_event(out, None, "[DONE]");
                self.done = true;
            }
            "error" => {
                let kind = data
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("upstream_error")
                    .to_owned();
                let message = data
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("upstream stream error")
                    .to_owned();
                self.fail(out, &kind, &message);
            }
            _ => {}
        }
    }

    /// Ends the stream. `error` is set when reading the upstream failed;
    /// a stream that ends without `message_stop` is also an error.
    pub fn finish(&mut self, error: Option<&str>, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        let message = error.unwrap_or("upstream stream ended before it was complete");
        self.fail(out, "upstream_error", message);
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
        request(&map(body), &ModelInfo::default(), false)
    }

    #[test]
    fn basic_request() {
        let out = translate(json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": [{"type": "text", "text": "No emoji."}]},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 0.3,
            "stop": "END",
            "user": "u-1",
            "stream": true,
            "stream_options": {"include_usage": true},
            "n": 1,
            "response_format": {"type": "text"},
        }))
        .unwrap();
        assert_eq!(
            out,
            json!({
                "model": "claude-sonnet-4-5",
                "system": "Be brief.\nNo emoji.",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": 0.3,
                "stop_sequences": ["END"],
                "stream": true,
                "metadata": {"user_id": "u-1"},
            })
        );
    }

    #[test]
    fn max_tokens_sources() {
        let body = |extra: Value| {
            let mut b = map(json!({"model": "m", "messages": []}));
            b.extend(map(extra));
            b
        };
        let info = ModelInfo {
            max_output_tokens: Some(8192),
            ..ModelInfo::default()
        };
        let mt = |b: Map<String, Value>, info: &ModelInfo| {
            request(&b, info, false).unwrap()["max_tokens"].clone()
        };
        assert_eq!(mt(body(json!({"max_tokens": 100})), &info), 100);
        assert_eq!(
            mt(
                body(json!({"max_completion_tokens": 200, "max_tokens": 100})),
                &info
            ),
            200
        );
        assert_eq!(mt(body(json!({})), &info), 8192);
        assert_eq!(
            mt(body(json!({})), &ModelInfo::default()),
            DEFAULT_MAX_TOKENS
        );
    }

    #[test]
    fn images_tools_and_tool_results() {
        let out = translate(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K"}},
                    {"type": "image_url", "image_url": "https://example.com/cat.jpg"},
                ]},
                {"role": "assistant", "content": "Let me look.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"cat\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "noargs", "arguments": ""}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a cat"},
                {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": "Thanks"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "description": "Look things up",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }}, {"type": "function", "function": {"name": "noargs"}}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "parallel_tool_calls": false,
        }))
        .unwrap();
        assert_eq!(
            out["messages"],
            json!([
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me look."},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "cat"}},
                    {"type": "tool_use", "id": "call_2", "name": "noargs", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "a cat"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "ok"},
                    {"type": "text", "text": "Thanks"},
                ]},
            ])
        );
        assert_eq!(
            out["tools"],
            json!([
                {"name": "lookup", "description": "Look things up",
                 "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
                {"name": "noargs", "input_schema": {"type": "object", "properties": {}}},
            ])
        );
        assert_eq!(
            out["tool_choice"],
            json!({"type": "tool", "name": "lookup", "disable_parallel_tool_use": true})
        );
    }

    #[test]
    fn tool_choice_strings() {
        for (choice, expected) in [
            ("auto", json!({"type": "auto"})),
            ("required", json!({"type": "any"})),
            ("none", json!({"type": "none"})),
        ] {
            let out =
                translate(json!({"model": "m", "messages": [], "tool_choice": choice})).unwrap();
            assert_eq!(out["tool_choice"], expected);
        }
    }

    #[test]
    fn reasoning_effort_becomes_thinking() {
        let out = translate(
            json!({"model": "m", "messages": [], "reasoning_effort": "medium", "max_tokens": 1000}),
        )
        .unwrap();
        assert_eq!(
            out["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert_eq!(out["max_tokens"], 3048);
        let out = translate(
            json!({"model": "m", "messages": [], "reasoning_effort": "high", "max_tokens": 10000}),
        )
        .unwrap();
        assert_eq!(out["max_tokens"], 10000);
        let out =
            translate(json!({"model": "m", "messages": [], "reasoning_effort": "none"})).unwrap();
        assert!(out.get("thinking").is_none());
        assert!(
            translate(json!({"model": "m", "messages": [], "reasoning_effort": "max"})).is_err()
        );
    }

    #[test]
    fn rejects_what_anthropic_cannot_take() {
        for (body, needle) in [
            (
                json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}]}]}),
                "data:, http:// or https://",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,AA"}}]}]}),
                "image/svg+xml",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,raw"}}]}]}),
                "base64",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]}),
                "input_audio",
            ),
            (
                json!({"messages": [{"role": "function", "content": "x"}]}),
                "role `function`",
            ),
            (
                json!({"messages": [{"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{bad"}}]}]}),
                "valid JSON",
            ),
            (
                json!({"messages": [{"role": "tool", "content": "x"}]}),
                "tool_call_id",
            ),
            (json!({"messages": "hi"}), "`messages` must be a list"),
            (
                json!({"messages": [], "tools": [{"type": "code_interpreter"}]}),
                "only function tools",
            ),
        ] {
            let mut b = map(body);
            b.insert("model".into(), "m".into());
            let e = request(&b, &ModelInfo::default(), false).unwrap_err();
            assert!(e.0.contains(needle), "{needle}: {e}");
        }
    }

    #[test]
    fn unsupported_params_fail_or_drop() {
        let body = map(
            json!({"model": "m", "messages": [], "logprobs": true, "seed": 1, "n": 2, "presence_penalty": null}),
        );
        let e = request(&body, &ModelInfo::default(), false).unwrap_err();
        assert!(e.0.contains("logprobs, n, seed"), "{e}");
        let out = request(&body, &ModelInfo::default(), true).unwrap();
        assert!(out.get("logprobs").is_none() && out.get("seed").is_none());
    }

    #[test]
    fn response_translation() {
        let (out, usage) = response(&json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Calling."},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "cat"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 7, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 5},
        }))
        .unwrap();
        assert_eq!(
            usage,
            Usage {
                prompt: 18,
                completion: 7
            }
        );
        assert_eq!(out["id"], "msg_1");
        assert_eq!(out["object"], "chat.completion");
        assert_eq!(out["model"], "claude-x");
        assert_eq!(
            out["choices"][0],
            json!({
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Calling.",
                    "reasoning_content": "hmm",
                    "tool_calls": [{"id": "toolu_1", "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"q\":\"cat\"}"}}],
                },
            })
        );
        assert_eq!(
            out["usage"],
            json!({
                "prompt_tokens": 18, "completion_tokens": 7, "total_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 5},
                "cache_creation_input_tokens": 3, "cache_read_input_tokens": 5,
            })
        );
        for (stop, finish) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("refusal", "content_filter"),
        ] {
            let (out, _) = response(&json!({"content": [], "stop_reason": stop})).unwrap();
            assert_eq!(out["choices"][0]["finish_reason"], finish);
        }
        assert!(response(&json!({"nope": 1})).is_none());
    }

    pub(crate) const ANTHROPIC_STREAM: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":12,\"output_tokens\":1,\"cache_read_input_tokens\":4}}}\n\n",
        "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"lookup\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"cat\\\"}\"}}\n\n",
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":30}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );

    /// Runs a raw stream through the translator, cutting the input at `split`.
    fn run(input: &str, split: usize, include_usage: bool) -> (Vec<Value>, usize, Usage) {
        let mut parser = SseParser::default();
        let mut t = StreamTranslator::new(include_usage, Redactor::default());
        let mut out = Vec::new();
        for part in [&input.as_bytes()[..split], &input.as_bytes()[split..]] {
            for ev in parser.push(part).unwrap() {
                t.event(&ev, &mut out);
            }
        }
        t.finish(None, &mut out);
        let mut chunks = Vec::new();
        let mut done = 0;
        for ev in SseParser::default().push(&out).unwrap() {
            if ev.data == "[DONE]" {
                done += 1;
            } else {
                chunks.push(serde_json::from_str(&ev.data).unwrap());
            }
        }
        (chunks, done, t.usage())
    }

    #[test]
    fn stream_translation() {
        let (chunks, done, usage) = run(ANTHROPIC_STREAM, 0, true);
        assert_eq!(done, 1);
        assert_eq!(
            usage,
            Usage {
                prompt: 16,
                completion: 30
            }
        );
        let deltas: Vec<&Value> = chunks.iter().map(|c| &c["choices"][0]["delta"]).collect();
        assert_eq!(chunks[0]["id"], "msg_1");
        assert_eq!(chunks[0]["object"], "chat.completion.chunk");
        assert_eq!(chunks[0]["model"], "claude-x");
        assert_eq!(*deltas[0], json!({"role": "assistant", "content": ""}));
        assert_eq!(*deltas[1], json!({"reasoning_content": "plan"}));
        assert_eq!(*deltas[2], json!({"content": "Hel"}));
        assert_eq!(*deltas[3], json!({"content": "lo"}));
        assert_eq!(
            *deltas[4],
            json!({"tool_calls": [{"index": 0, "id": "toolu_1", "type": "function", "function": {"name": "lookup", "arguments": ""}}]})
        );
        assert_eq!(
            *deltas[5],
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"q\":"}}]})
        );
        assert_eq!(
            *deltas[6],
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"cat\"}"}}]})
        );
        assert_eq!(chunks[7]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[8]["choices"], json!([]));
        assert_eq!(chunks[8]["usage"]["prompt_tokens"], 16);
        assert_eq!(chunks[8]["usage"]["completion_tokens"], 30);
        assert_eq!(chunks.len(), 9);
    }

    #[test]
    fn stream_translation_is_independent_of_chunking() {
        let (whole, _, _) = run(ANTHROPIC_STREAM, 0, false);
        assert!(whole.iter().all(|c| c.get("usage").is_none()));
        let strip = |cs: Vec<Value>| -> Vec<Value> {
            cs.into_iter().map(|c| c["choices"].clone()).collect()
        };
        let whole = strip(whole);
        for split in (0..ANTHROPIC_STREAM.len()).step_by(7) {
            let (chunks, done, _) = run(ANTHROPIC_STREAM, split, false);
            assert_eq!(done, 1);
            assert_eq!(strip(chunks), whole, "split at {split}");
        }
    }

    #[test]
    fn stream_errors_and_truncation() {
        let secret = Secret::new("sk-ant-secret-value-123");
        let errored = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{}}}\n\n",
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded sk-ant-secret-value-123\"}}\n\n",
        );
        let mut parser = SseParser::default();
        let mut t = StreamTranslator::new(false, Redactor::new([&secret]));
        let mut out = Vec::new();
        for ev in parser.push(errored.as_bytes()).unwrap() {
            t.event(&ev, &mut out);
        }
        t.finish(None, &mut out);
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("overloaded_error") && text.contains("[REDACTED]"),
            "{text}"
        );
        assert!(!text.contains("sk-ant-secret"));
        assert_eq!(text.matches("[DONE]").count(), 1);

        let truncated = &ANTHROPIC_STREAM[..ANTHROPIC_STREAM.find("event: message_delta").unwrap()];
        let (chunks, done, _) = run(truncated, 0, false);
        assert_eq!(done, 1);
        assert!(
            chunks.last().unwrap()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("ended")
        );
    }

    use crate::secrets::Secret;
}
