//! Streaming response bodies: upstream SSE in, client-format SSE out.

use std::time::Duration;

use bytes::Bytes;
use serde_json::{Value, json};

use super::{ByteStream, UsageHandle};
use crate::router::Lease;
use crate::secrets::Redactor;
use crate::translate::sse::{SseEvent, SseParser, write_event};
use crate::translate::{Usage, anthropic_to_openai, openai_to_anthropic};

/// The event format the client expects, for error events on passthrough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ClientFormat {
    /// `data:` chunks ending in `[DONE]` (chat and text completions).
    OpenAiChat,
    /// Named events (the Responses API).
    OpenAiEvents,
    Anthropic,
}

pub(super) enum Transform {
    Pass {
        format: ClientFormat,
        /// Drop the usage-only chunk that ferrolite asked for itself.
        hide_usage: bool,
        scrub: Redactor,
        usage: Usage,
        /// Events seen; a stream that ends with none is an error.
        events: u64,
        /// The upstream reported an error, or the stream broke.
        failed: bool,
    },
    ToOpenAi(openai_to_anthropic::StreamTranslator),
    ToAnthropic(anthropic_to_openai::StreamTranslator),
}

impl Transform {
    pub(super) fn pass(format: ClientFormat, hide_usage: bool, scrub: Redactor) -> Self {
        Self::Pass {
            format,
            hide_usage,
            scrub,
            usage: Usage::default(),
            events: 0,
            failed: false,
        }
    }

    fn event(&mut self, ev: &SseEvent, out: &mut Vec<u8>) {
        match self {
            Self::ToOpenAi(t) => t.event(ev, out),
            Self::ToAnthropic(t) => t.event(ev, out),
            Self::Pass {
                hide_usage,
                usage,
                events,
                failed,
                ..
            } => {
                *events += 1;
                if let Ok(data) = serde_json::from_str::<Value>(&ev.data) {
                    // OpenAI-style `{"error": ...}` chunks, and `type: error`
                    // events from Anthropic and the Responses API.
                    if data.get("error").is_some_and(|e| !e.is_null()) || data["type"] == "error" {
                        *failed = true;
                    }
                    let found = data
                        .get("usage")
                        .or_else(|| data.pointer("/response/usage"))
                        .or_else(|| data.pointer("/message/usage"))
                        .and_then(Usage::from_value);
                    if let Some(u) = found {
                        // Anthropic splits usage over message_start and
                        // message_delta; keep the largest of each.
                        usage.prompt = usage.prompt.max(u.prompt);
                        usage.completion = usage.completion.max(u.completion);
                    }
                    let usage_only = data
                        .get("choices")
                        .and_then(Value::as_array)
                        .is_some_and(Vec::is_empty)
                        && data.get("usage").is_some_and(|u| !u.is_null());
                    if *hide_usage && usage_only {
                        return;
                    }
                }
                write_event(out, ev.event.as_deref(), &ev.data);
            }
        }
    }

    fn finish(&mut self, error: Option<&str>, out: &mut Vec<u8>) {
        match self {
            Self::ToOpenAi(t) => t.finish(error, out),
            Self::ToAnthropic(t) => t.finish(error, out),
            Self::Pass {
                format,
                scrub,
                events,
                failed,
                ..
            } => {
                let message = match error {
                    Some(m) => m,
                    None if *events == 0 => "the upstream stream ended without sending any events",
                    None => return,
                };
                *failed = true;
                let message = scrub.redact(message);
                match format {
                    ClientFormat::OpenAiChat => {
                        let e = json!({"error": {"message": message, "type": "upstream_error"}});
                        write_event(out, None, &e.to_string());
                        write_event(out, None, "[DONE]");
                    }
                    ClientFormat::OpenAiEvents => {
                        let e =
                            json!({"type": "error", "code": "upstream_error", "message": message});
                        write_event(out, Some("error"), &e.to_string());
                    }
                    ClientFormat::Anthropic => {
                        let e = json!({"type": "error", "error": {"type": "api_error", "message": message}});
                        write_event(out, Some("error"), &e.to_string());
                    }
                }
            }
        }
    }

    fn usage(&self) -> Usage {
        match self {
            Self::ToOpenAi(t) => t.usage(),
            Self::ToAnthropic(t) => t.usage(),
            Self::Pass { usage, .. } => *usage,
        }
    }

    /// Whether the stream carried an upstream error or ended badly.
    fn failed(&self) -> bool {
        match self {
            Self::ToOpenAi(t) => t.failed(),
            Self::ToAnthropic(t) => t.failed(),
            Self::Pass { failed, .. } => *failed,
        }
    }
}

struct State {
    response: reqwest::Response,
    parser: SseParser,
    transform: Transform,
    idle: Duration,
    lease: Option<Lease>,
    usage: UsageHandle,
    done: bool,
}

impl Drop for State {
    /// Runs when the stream ends or the client goes away: either way the
    /// tokens seen so far count against tpm and the lease is released.
    fn drop(&mut self) {
        let usage = self.transform.usage();
        self.usage.set(usage, self.transform.failed());
        if let Some(lease) = self.lease.take() {
            lease.record_tokens(usage.total());
        }
    }
}

/// Reads the upstream stream, translating as it goes. A read that takes
/// longer than `idle` ends the stream with an error event in the client's
/// format, as does a broken connection or an oversized event.
pub(super) fn body(
    response: reqwest::Response,
    idle: Duration,
    transform: Transform,
    lease: Lease,
    usage: UsageHandle,
) -> ByteStream {
    let state = State {
        response,
        parser: SseParser::default(),
        transform,
        idle,
        lease: Some(lease),
        usage,
        done: false,
    };
    Box::pin(futures_util::stream::unfold(state, |mut st| async move {
        while !st.done {
            let mut out = Vec::new();
            match tokio::time::timeout(st.idle, st.response.chunk()).await {
                Ok(Ok(Some(chunk))) => match st.parser.push(&chunk) {
                    Ok(events) => {
                        for ev in &events {
                            st.transform.event(ev, &mut out);
                        }
                    }
                    Err(e) => {
                        st.transform.finish(Some(&e.to_string()), &mut out);
                        st.done = true;
                    }
                },
                Ok(Ok(None)) => {
                    st.transform.finish(None, &mut out);
                    st.done = true;
                }
                Ok(Err(_)) => {
                    st.transform
                        .finish(Some("the upstream connection failed mid-stream"), &mut out);
                    st.done = true;
                }
                Err(_) => {
                    let msg = format!("the upstream sent nothing for {}s", st.idle.as_secs_f64());
                    st.transform.finish(Some(&msg), &mut out);
                    st.done = true;
                }
            }
            if !out.is_empty() {
                return Some((Bytes::from(out), st));
            }
        }
        None
    }))
}
