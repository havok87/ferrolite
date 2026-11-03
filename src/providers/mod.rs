//! Upstream calls: one shared HTTP client, request preparation and
//! translation, error classification, and response bodies (whole or
//! streamed) in the format the client asked for.
//!
//! [`Upstream::send`] is the call the router runs for each attempt.

mod anthropic;
mod openai_compat;
mod stream;
#[cfg(test)]
mod tests;

use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bytes::Bytes;
use futures_util::Stream;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};
use url::Position;

use crate::config::{Config, Deployment, HeaderValue as ConfigHeader, Provider};
use crate::router::{Attempt, Classify, ErrorClass, Lease};
use crate::secrets::Redactor;
use crate::translate::{
    self, TranslateError, Usage, anthropic_to_openai, openai_to_anthropic, truncate_chars,
};
use stream::{ClientFormat, Transform};

/// Default cap on a non-streaming upstream response body.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How much of an upstream error body is read to classify it.
const MAX_ERROR_BYTES: usize = 64 * 1024;
/// Longest upstream error message passed on.
const MAX_ERROR_MESSAGE_CHARS: usize = 1024;

/// The client-facing endpoint a request arrived on. It decides the
/// response format: OpenAI for everything but `/v1/messages`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    Completions,
    Embeddings,
    Responses,
    Messages,
}

impl Endpoint {
    /// Path relative to an OpenAI-style `.../v1` base.
    pub fn path(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat/completions",
            Self::Completions => "completions",
            Self::Embeddings => "embeddings",
            Self::Responses => "responses",
            Self::Messages => "messages",
        }
    }
}

/// A client request, after the server has parsed and authenticated it.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    pub endpoint: Endpoint,
    pub body: Map<String, Value>,
    /// Forwarded only on Anthropic passthrough.
    pub anthropic_version: Option<String>,
    /// Forwarded only on Anthropic passthrough.
    pub anthropic_beta: Option<String>,
}

impl ProxyRequest {
    pub fn new(endpoint: Endpoint, body: Map<String, Value>) -> Self {
        Self {
            endpoint,
            body,
            anthropic_version: None,
            anthropic_beta: None,
        }
    }

    pub fn stream(&self) -> bool {
        self.body.get("stream").and_then(Value::as_bool) == Some(true)
    }

    fn include_usage(&self) -> bool {
        self.body
            .get("stream_options")
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            == Some(true)
    }
}

/// A response body stream, already in the client's format.
pub type ByteStream = Pin<Box<dyn Stream<Item = Bytes> + Send>>;

/// How a stream went, filled in when the stream ends.
#[derive(Debug, Clone, Default)]
pub struct UsageHandle(Arc<Mutex<StreamOutcome>>);

#[derive(Debug, Clone, Copy, Default)]
struct StreamOutcome {
    usage: Option<Usage>,
    failed: bool,
}

impl UsageHandle {
    /// Token counts, once the stream has ended.
    pub fn get(&self) -> Option<Usage> {
        self.outcome().usage
    }

    /// Whether the stream carried an upstream error, broke, stalled or
    /// ended without completing.
    pub fn failed(&self) -> bool {
        self.outcome().failed
    }

    fn outcome(&self) -> StreamOutcome {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn set(&self, usage: Usage, failed: bool) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = StreamOutcome {
            usage: Some(usage),
            failed,
        };
    }
}

pub enum ProxyResponse {
    Json {
        body: Bytes,
        usage: Option<Usage>,
    },
    Stream {
        body: ByteStream,
        usage: UsageHandle,
    },
}

impl std::fmt::Debug for ProxyResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json { body, usage } => f
                .debug_struct("Json")
                .field("bytes", &body.len())
                .field("usage", usage)
                .finish(),
            Self::Stream { .. } => f.write_str("Stream"),
        }
    }
}

/// A failed attempt. `message` is safe to show a client: secrets and
/// upstream hosts are scrubbed from it.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct ProviderError {
    pub class: ErrorClass,
    /// The HTTP status to answer the client with.
    pub status: u16,
    pub message: String,
}

impl ProviderError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::BadRequest,
            status: 400,
            message: message.into(),
        }
    }

    fn bad_gateway(class: ErrorClass, message: impl Into<String>) -> Self {
        Self {
            class,
            status: 502,
            message: message.into(),
        }
    }
}

impl From<TranslateError> for ProviderError {
    fn from(e: TranslateError) -> Self {
        Self::bad_request(e.0)
    }
}

impl Classify for ProviderError {
    fn classify(&self) -> ErrorClass {
        self.class
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UpstreamOptions {
    pub max_response_bytes: usize,
    pub connect_timeout: Duration,
}

impl Default for UpstreamOptions {
    fn default() -> Self {
        Self {
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("cannot build the HTTP client: {0}")]
pub struct ClientBuildError(String);

/// Shared upstream HTTP client plus what every call needs.
pub struct Upstream {
    client: reqwest::Client,
    scrub: Redactor,
    drop_params: bool,
    max_response_bytes: usize,
}

impl Upstream {
    /// `redactor` holds the config's secrets; upstream hosts are added to it
    /// so neither reaches a client in an error message.
    pub fn new(
        config: &Config,
        redactor: &Redactor,
        options: UpstreamOptions,
    ) -> Result<Self, ClientBuildError> {
        let client = reqwest::Client::builder()
            // Never re-send credentials to wherever a redirect points.
            .redirect(reqwest::redirect::Policy::none())
            // Ignore HTTP(S)_PROXY: keys must not be routed through a proxy
            // that the environment happens to name.
            .no_proxy()
            .connect_timeout(options.connect_timeout)
            .http1_only()
            .user_agent(concat!("ferrolite/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ClientBuildError(e.to_string()))?;
        let hosts = config.deployments.iter().flat_map(|d| {
            let url = &d.api_base;
            [
                url[Position::BeforeHost..Position::AfterPort].to_owned(),
                url.host_str().unwrap_or("").to_owned(),
            ]
        });
        Ok(Self {
            client,
            scrub: redactor.with_patterns(hosts),
            drop_params: config.drop_params,
            max_response_bytes: options.max_response_bytes,
        })
    }

    /// Performs one attempt. Returns once the upstream has answered with a
    /// status: a stream's body is read later, by whoever consumes it.
    pub async fn send(
        &self,
        attempt: Attempt,
        request: &ProxyRequest,
    ) -> Result<ProxyResponse, ProviderError> {
        let d = Arc::clone(&attempt.deployment);
        let drop_params = d.drop_params.unwrap_or(self.drop_params);
        let stream = request.stream();
        let mut body = request.body.clone();
        translate::sanitize(&mut body);
        translate::reject_upstream_resources(&body)?;
        drop_listed(&mut body, &d);
        body.insert("model".into(), attempt.upstream_model.clone().into());

        let (url, headers, body, format, transform) = match (d.provider, request.endpoint) {
            (Provider::Anthropic, Endpoint::Messages) => {
                translate::apply_shared_defaults(&mut body, &d.defaults);
                (
                    anthropic::messages_url(&d.api_base),
                    anthropic::headers(
                        &d,
                        request.anthropic_version.as_deref(),
                        request.anthropic_beta.as_deref(),
                    ),
                    Value::Object(body),
                    Format::Pass,
                    Transform::pass(ClientFormat::Anthropic, false, self.scrub.clone()),
                )
            }
            (Provider::Anthropic, Endpoint::ChatCompletions) => {
                translate::apply_defaults(&mut body, &d.defaults);
                let mut out = openai_to_anthropic::request(&body, &d.model_info, drop_params)?;
                drop_listed_value(&mut out, &d);
                (
                    anthropic::messages_url(&d.api_base),
                    anthropic::headers(&d, None, None),
                    out,
                    Format::AnthropicToOpenAi,
                    Transform::ToOpenAi(openai_to_anthropic::StreamTranslator::new(
                        request.include_usage(),
                        self.scrub.clone(),
                    )),
                )
            }
            (Provider::Anthropic, other) => {
                return Err(ProviderError::bad_request(format!(
                    "/v1/{} is not supported for anthropic/ deployments",
                    other.path()
                )));
            }
            (Provider::OpenAiCompat(_), Endpoint::Messages) => {
                let mut out = anthropic_to_openai::request(&body, drop_params)?;
                if let Value::Object(map) = &mut out {
                    translate::apply_defaults(map, &d.defaults);
                    drop_listed(map, &d);
                }
                (
                    openai_compat::url(&d.api_base, Endpoint::ChatCompletions),
                    openai_compat::headers(&d),
                    out,
                    Format::OpenAiToAnthropic,
                    Transform::ToAnthropic(anthropic_to_openai::StreamTranslator::new(
                        self.scrub.clone(),
                    )),
                )
            }
            (Provider::OpenAiCompat(_), endpoint) => {
                translate::apply_defaults(&mut body, &d.defaults);
                // To count tokens against tpm, ask for usage and hide the
                // extra chunk again if the client did not ask for it.
                let hide_usage = endpoint == Endpoint::ChatCompletions
                    && stream
                    && d.tpm.is_some()
                    && !request.include_usage();
                if hide_usage {
                    let options = body
                        .entry("stream_options")
                        .or_insert_with(|| Value::Object(Map::new()));
                    if let Value::Object(o) = options {
                        o.insert("include_usage".into(), true.into());
                    }
                }
                let format = if endpoint == Endpoint::Responses {
                    ClientFormat::OpenAiEvents
                } else {
                    ClientFormat::OpenAiChat
                };
                (
                    openai_compat::url(&d.api_base, endpoint),
                    openai_compat::headers(&d),
                    Value::Object(body),
                    Format::Pass,
                    Transform::pass(format, hide_usage, self.scrub.clone()),
                )
            }
        };

        let payload = serde_json::to_vec(&body)
            .map_err(|_| ProviderError::bad_request("request body cannot be serialized"))?;
        let response = self
            .client
            .post(url)
            .headers(headers)
            .header(CONTENT_TYPE, "application/json")
            .header(
                ACCEPT,
                if stream {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            )
            .body(payload)
            .send()
            .await
            .map_err(|e| connect_error(&e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(self.http_error(response).await);
        }
        let Attempt { lease, timeout, .. } = attempt;
        if stream && !is_event_stream(response.headers()) {
            // Some backends ignore `stream` or report errors with a 200 and
            // a JSON body; either way there is nothing to stream.
            return Err(self.not_a_stream(response).await);
        }
        if stream {
            let usage = UsageHandle::default();
            let body = stream::body(response, timeout, transform, lease, usage.clone());
            return Ok(ProxyResponse::Stream { body, usage });
        }
        let raw = self.read_body(response).await?;
        self.json_response(raw, format, &lease)
    }

    /// Checks that a deployment answers `GET .../models` with its configured
    /// credentials, within `timeout`.
    pub async fn probe(&self, d: &Deployment, timeout: Duration) -> Result<(), ProviderError> {
        let (url, headers) = match d.provider {
            Provider::Anthropic => (
                anthropic::models_url(&d.api_base),
                anthropic::headers(d, None, None),
            ),
            Provider::OpenAiCompat(_) => (
                openai_compat::models_url(&d.api_base),
                openai_compat::headers(d),
            ),
        };
        let request = self.client.get(url).headers(headers).send();
        match tokio::time::timeout(timeout, request).await {
            Err(_) => Err(ProviderError {
                class: ErrorClass::Timeout,
                status: 504,
                message: "the upstream did not respond in time".to_owned(),
            }),
            Ok(Err(e)) => Err(connect_error(&e)),
            Ok(Ok(response)) if response.status().is_success() => Ok(()),
            Ok(Ok(response)) => Err(self.http_error(response).await),
        }
    }

    /// Reads a non-streaming body, refusing anything over the size cap.
    async fn read_body(&self, mut response: reqwest::Response) -> Result<Bytes, ProviderError> {
        let mut buf = Vec::new();
        loop {
            match response.chunk().await {
                Ok(Some(chunk)) => {
                    if buf.len() + chunk.len() > self.max_response_bytes {
                        return Err(ProviderError::bad_gateway(
                            ErrorClass::BadRequest,
                            format!(
                                "the upstream response is larger than {} bytes",
                                self.max_response_bytes
                            ),
                        ));
                    }
                    buf.extend_from_slice(&chunk);
                }
                Ok(None) => return Ok(buf.into()),
                Err(_) => {
                    return Err(ProviderError::bad_gateway(
                        ErrorClass::Connection,
                        "the upstream connection failed while reading the response",
                    ));
                }
            }
        }
    }

    fn json_response(
        &self,
        raw: Bytes,
        format: Format,
        lease: &Lease,
    ) -> Result<ProxyResponse, ProviderError> {
        let invalid = || {
            ProviderError::bad_gateway(
                ErrorClass::Server,
                "the upstream returned an unexpected response",
            )
        };
        let parsed: Option<Value> = serde_json::from_slice(&raw).ok();
        let (body, usage) = match format {
            // Passthrough bodies go back byte for byte.
            Format::Pass => {
                let usage = parsed.as_ref().and_then(|v| {
                    v.get("usage")
                        .or_else(|| v.pointer("/response/usage"))
                        .and_then(Usage::from_value)
                });
                (raw, usage)
            }
            Format::AnthropicToOpenAi => {
                let (out, usage) = parsed
                    .as_ref()
                    .and_then(openai_to_anthropic::response)
                    .ok_or_else(invalid)?;
                (Bytes::from(out.to_string()), Some(usage))
            }
            Format::OpenAiToAnthropic => {
                let (out, usage) = parsed
                    .as_ref()
                    .and_then(anthropic_to_openai::response)
                    .ok_or_else(invalid)?;
                (Bytes::from(out.to_string()), Some(usage))
            }
        };
        if let Some(u) = usage {
            lease.record_tokens(u.total());
        }
        Ok(ProxyResponse::Json { body, usage })
    }

    /// A 2xx answer to a streaming request that is not an event stream.
    async fn not_a_stream(&self, response: reqwest::Response) -> ProviderError {
        let mut buf = Vec::new();
        let mut response = response;
        while buf.len() < MAX_ERROR_BYTES {
            match response.chunk().await {
                Ok(Some(chunk)) => buf.extend_from_slice(&chunk),
                _ => break,
            }
        }
        let detail = serde_json::from_slice::<Value>(&buf)
            .ok()
            .and_then(|b| error_fields(&b).0)
            .map(|m| {
                format!(
                    ": {}",
                    truncate_chars(&self.scrub.redact(&m), MAX_ERROR_MESSAGE_CHARS)
                )
            })
            .unwrap_or_default();
        ProviderError::bad_gateway(
            ErrorClass::Server,
            format!("the upstream answered a streaming request without an event stream{detail}"),
        )
    }

    /// Classifies a non-2xx response and extracts a client-safe message.
    async fn http_error(&self, mut response: reqwest::Response) -> ProviderError {
        let status = response.status().as_u16();
        if status < 400 {
            // Redirects are not followed, so one means api_base is wrong.
            return ProviderError::bad_gateway(
                ErrorClass::NotFound,
                format!("the upstream answered with HTTP {status}; redirects are not followed"),
            );
        }
        let retry_after = retry_after(response.headers());
        let mut buf = Vec::new();
        while buf.len() < MAX_ERROR_BYTES {
            match response.chunk().await {
                Ok(Some(chunk)) => buf.extend_from_slice(&chunk),
                _ => break,
            }
        }
        buf.truncate(MAX_ERROR_BYTES);
        let body: Option<Value> = serde_json::from_slice(&buf).ok();
        let (message, code) = body.as_ref().map(error_fields).unwrap_or_default();
        let class = classify(
            status,
            code.as_deref(),
            message.as_deref().unwrap_or(""),
            retry_after,
        );
        let message = match message {
            Some(m) => {
                let m = self.scrub.redact(&m).into_owned();
                truncate_chars(&m, MAX_ERROR_MESSAGE_CHARS)
            }
            None => format!("the upstream returned HTTP {status}"),
        };
        ProviderError {
            class,
            status,
            message,
        }
    }
}

/// How a non-streaming response body is turned into the client's format.
#[derive(Debug, Clone, Copy)]
enum Format {
    Pass,
    AnthropicToOpenAi,
    OpenAiToAnthropic,
}

fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| {
            ct.trim_start()
                .to_ascii_lowercase()
                .starts_with("text/event-stream")
        })
}

fn drop_listed(body: &mut Map<String, Value>, d: &Deployment) {
    for key in &d.additional_drop_params {
        body.remove(key);
    }
}

fn drop_listed_value(body: &mut Value, d: &Deployment) {
    if let Value::Object(map) = body {
        drop_listed(map, d);
    }
}

/// The deployment's `extra_headers`, marked sensitive where they are secrets.
fn extra_headers(d: &Deployment, headers: &mut HeaderMap) {
    for (name, value) in &d.extra_headers {
        let (Ok(name), Ok(mut v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value.expose()),
        ) else {
            continue;
        };
        v.set_sensitive(matches!(value, ConfigHeader::Secret(_)));
        headers.insert(name, v);
    }
}

fn connect_error(e: &reqwest::Error) -> ProviderError {
    // reqwest's error text includes the URL, so it is not passed on.
    if e.is_timeout() && !e.is_connect() {
        return ProviderError {
            class: ErrorClass::Timeout,
            status: 504,
            message: "the upstream did not respond in time".to_owned(),
        };
    }
    ProviderError::bad_gateway(ErrorClass::Connection, "could not connect to the upstream")
}

/// `Retry-After` in seconds, or `retry-after-ms` (sent by OpenAI).
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let number = |name: &str| {
        headers
            .get(name)?
            .to_str()
            .ok()?
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite() && *n >= 0.0)
    };
    if let Some(ms) = number("retry-after-ms") {
        return Duration::try_from_secs_f64(ms / 1000.0).ok();
    }
    number("retry-after").and_then(|s| Duration::try_from_secs_f64(s).ok())
}

/// The message and code from the error shapes upstreams use: OpenAI
/// `{"error":{"message","code"}}`, Anthropic `{"error":{"type","message"}}`,
/// vLLM `{"message"}` and Ollama `{"error":"..."}`.
fn error_fields(body: &Value) -> (Option<String>, Option<String>) {
    let text = |v: Option<&Value>| v.and_then(Value::as_str).map(str::to_owned);
    match body.get("error") {
        Some(Value::String(s)) => (Some(s.clone()), None),
        Some(e) => (
            text(e.get("message")),
            text(e.get("code")).or_else(|| text(e.get("type"))),
        ),
        None => (
            text(body.get("message")),
            text(body.get("code")).or_else(|| text(body.get("type"))),
        ),
    }
}

fn classify(
    status: u16,
    code: Option<&str>,
    message: &str,
    retry_after: Option<Duration>,
) -> ErrorClass {
    let m = message.to_ascii_lowercase();
    let context = code == Some("context_length_exceeded")
        || [
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many tokens",
            "input is too long",
        ]
        .iter()
        .any(|needle| m.contains(needle));
    let policy = matches!(code, Some("content_filter" | "content_policy_violation"))
        || m.contains("content management policy")
        || m.contains("content policy");
    match status {
        429 => ErrorClass::RateLimited { retry_after },
        408 => ErrorClass::Timeout,
        401 | 403 => ErrorClass::Auth,
        404 => ErrorClass::NotFound,
        s if s >= 500 => ErrorClass::Server,
        _ if context => ErrorClass::ContextWindowExceeded,
        _ if policy => ErrorClass::ContentPolicy,
        _ => ErrorClass::BadRequest,
    }
}
