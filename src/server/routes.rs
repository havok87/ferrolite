//! HTTP routes. Every handler authenticates (except liveliness and
//! readiness), and every response carries an `x-request-id` and is
//! logged and counted once.

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Response, Uri, header};
use axum::routing::{get, post};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::OwnedSemaphorePermit;
use tokio::time::Instant;

use super::errors::{ApiError, ErrorFormat};
use super::limits::{BodyError, read_body};
use super::metrics::{Route, class_label};
use super::{AppState, auth, health};
use crate::providers::{Endpoint, ProxyRequest, ProxyResponse, UsageHandle};
use crate::router::{Hints, RouterError};
use crate::translate::{self, Usage};

type St = State<Arc<AppState>>;

pub(super) fn router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/responses", post(responses))
        .route("/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(list_models))
        .route("/models", get(list_models))
        .route("/v1/models/{*id}", get(get_model))
        .route("/models/{*id}", get(get_model))
        .route("/model/info", get(model_info))
        .route("/v1/model/info", get(model_info))
        .route("/health", get(health_check))
        .route("/health/readiness", get(readiness))
        .route("/health/liveliness", get(liveliness))
        .route("/metrics", get(metrics))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .with_state(state)
}

/// What gets logged for a request. Never prompts, completions or headers.
struct Log {
    id: String,
    route: Route,
    started: Instant,
    model_group: Option<String>,
    deployment: Option<String>,
    attempts: u32,
    fallback: Option<String>,
    usage: Option<Usage>,
    emitted: bool,
    /// Set for proxied requests, so one dropped before it finished (the
    /// client went away) is still logged and counted.
    on_cancel: Option<Arc<AppState>>,
}

/// nginx's "client closed request": logged when the client disconnects
/// before a response is ready.
const CLIENT_CLOSED: u16 = 499;

impl Log {
    fn new(st: &AppState, route: Route) -> Self {
        Self {
            id: st.request_id(),
            route,
            started: Instant::now(),
            model_group: None,
            deployment: None,
            attempts: 0,
            fallback: None,
            usage: None,
            emitted: false,
            on_cancel: None,
        }
    }

    fn log_if_cancelled(&mut self, st: &Arc<AppState>) {
        self.on_cancel = Some(Arc::clone(st));
    }

    fn emit(&mut self, st: &AppState, status: u16) {
        self.emitted = true;
        let elapsed = self.started.elapsed();
        st.metrics.request(self.route, status, elapsed);
        let latency_ms = elapsed.as_millis() as u64;
        let prompt_tokens = self.usage.map(|u| u.prompt);
        let completion_tokens = self.usage.map(|u| u.completion);
        macro_rules! event {
            ($level:ident) => {
                tracing::$level!(
                    request_id = %self.id,
                    route = self.route.label(),
                    model_group = self.model_group.as_deref(),
                    deployment = self.deployment.as_deref(),
                    status,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    attempts = self.attempts,
                    fallback = self.fallback.as_deref(),
                    "request"
                )
            };
        }
        if status >= 500 {
            event!(warn);
        } else {
            event!(info);
        }
    }

    /// Logs, counts and stamps the request id on a finished response.
    fn finish(mut self, st: &AppState, mut resp: Response<Body>) -> Response<Body> {
        if let Ok(v) = HeaderValue::from_str(&self.id) {
            resp.headers_mut().insert("x-request-id", v);
        }
        self.emit(st, resp.status().as_u16());
        resp
    }

    fn error(self, st: &AppState, err: ApiError, format: ErrorFormat) -> Response<Body> {
        let mut err = err;
        err.message = st.scrub.redact(&err.message).into_owned();
        self.finish(st, err.response(format))
    }
}

impl Drop for Log {
    fn drop(&mut self) {
        if !self.emitted
            && let Some(st) = self.on_cancel.take()
        {
            self.emit(&st, CLIENT_CLOSED);
        }
    }
}

fn json_response(status: u16, value: &Value) -> Response<Body> {
    super::json_response(status, value.to_string())
}

/// `Err` with the 401 response if the request lacks the master key.
fn authenticate(
    st: &AppState,
    headers: &HeaderMap,
    log: Log,
    format: ErrorFormat,
) -> Result<Log, Box<Response<Body>>> {
    if auth::authorized(st.master_key.as_ref(), headers) {
        return Ok(log);
    }
    st.metrics.auth_failure();
    tracing::warn!(
        request_id = %log.id,
        route = log.route.label(),
        "rejected a request with a missing or wrong key"
    );
    Err(Box::new(log.error(
        st,
        ApiError::new(401, "missing or invalid API key"),
        format,
    )))
}

// ---- proxied endpoints ----

async fn chat_completions(State(st): St, headers: HeaderMap, body: Body) -> Response<Body> {
    proxy(
        st,
        headers,
        body,
        Endpoint::ChatCompletions,
        Route::ChatCompletions,
    )
    .await
}

async fn completions(State(st): St, headers: HeaderMap, body: Body) -> Response<Body> {
    proxy(st, headers, body, Endpoint::Completions, Route::Completions).await
}

async fn embeddings(State(st): St, headers: HeaderMap, body: Body) -> Response<Body> {
    proxy(st, headers, body, Endpoint::Embeddings, Route::Embeddings).await
}

async fn responses(State(st): St, headers: HeaderMap, body: Body) -> Response<Body> {
    proxy(st, headers, body, Endpoint::Responses, Route::Responses).await
}

async fn messages(State(st): St, headers: HeaderMap, body: Body) -> Response<Body> {
    proxy(st, headers, body, Endpoint::Messages, Route::Messages).await
}

async fn proxy(
    st: Arc<AppState>,
    headers: HeaderMap,
    body: Body,
    endpoint: Endpoint,
    route: Route,
) -> Response<Body> {
    let format = if endpoint == Endpoint::Messages {
        ErrorFormat::Anthropic
    } else {
        ErrorFormat::OpenAi
    };
    let mut log = match authenticate(&st, &headers, Log::new(&st, route), format) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    log.log_if_cancelled(&st);
    let Some(permit) = st.concurrency.try_acquire() else {
        let e = ApiError::new(
            503,
            "the gateway is at its concurrency limit; retry shortly",
        )
        .with_retry_after(1);
        return log.error(&st, e, format);
    };

    let bytes = match read_body(
        body,
        st.options.max_request_bytes,
        st.options.body_read_timeout,
    )
    .await
    {
        Ok(b) => b,
        Err(e) => {
            let e = match e {
                BodyError::TooLarge => ApiError::new(
                    413,
                    format!(
                        "the request body is larger than {} bytes",
                        st.options.max_request_bytes
                    ),
                ),
                BodyError::Timeout => {
                    ApiError::new(408, "the request body was not received in time")
                }
                BodyError::Read => ApiError::new(400, "the request body could not be read"),
            };
            return log.error(&st, e, format);
        }
    };
    let Ok(Value::Object(mut body)) = serde_json::from_slice::<Value>(&bytes) else {
        return log.error(
            &st,
            ApiError::new(400, "the request body must be a JSON object"),
            format,
        );
    };
    // Routing and credential fields are never taken from a client.
    translate::sanitize(&mut body);
    let Some(model) = body.get("model").and_then(Value::as_str).map(str::to_owned) else {
        return log.error(&st, ApiError::new(400, "`model` is required"), format);
    };
    let requested_group = st.router.resolve(&model).map(|r| r.group);
    log.model_group.clone_from(&requested_group);

    let mut request = ProxyRequest::new(endpoint, body);
    if endpoint == Endpoint::Messages {
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        request.anthropic_version = header("anthropic-version");
        request.anthropic_beta = header("anthropic-beta");
    }
    let hints = Hints {
        stream: request.stream(),
    };
    let result = st
        .router
        .execute(&model, hints, |attempt| st.upstream.send(attempt, &request))
        .await;

    let routed = match result {
        Ok(r) => r,
        Err(err) => {
            let class = match &err {
                RouterError::UnknownModel { .. } => "unknown_model",
                RouterError::NoDeployment { .. } => "no_deployment",
                RouterError::Timeout { .. } => "timeout",
                RouterError::Upstream { error, .. } => class_label(error.class),
            };
            if let RouterError::Upstream { deployment, .. }
            | RouterError::Timeout { deployment, .. } = &err
            {
                log.deployment = Some(deployment.clone());
            }
            if let Some(group) = &requested_group {
                st.metrics.upstream_error(group, class);
            }
            return log.error(&st, ApiError::from_router(&err), format);
        }
    };

    log.deployment = Some(routed.deployment_id);
    log.attempts = routed.attempts;
    if routed.fallback.is_some()
        && let Some(group) = &requested_group
    {
        st.metrics.fallback(group);
    }
    log.fallback = routed.fallback;
    let served_group = routed.model_group;

    match routed.value {
        ProxyResponse::Json { body, usage } => {
            if let Some(u) = usage {
                st.metrics.tokens(&served_group, u);
            }
            log.usage = usage;
            drop(permit);
            log.finish(&st, super::json_response(200, body))
        }
        ProxyResponse::Stream { body, usage } => {
            let mut resp = Response::new(Body::empty());
            if let Ok(v) = HeaderValue::from_str(&log.id) {
                resp.headers_mut().insert("x-request-id", v);
            }
            // Logged and counted when the stream ends or the client leaves.
            let guard = StreamGuard {
                st: Arc::clone(&st),
                log: Some(log),
                usage,
                group: served_group,
                _permit: permit,
            };
            let stream = body.map(move |chunk| {
                let _ = &guard;
                Ok::<_, Infallible>(chunk)
            });
            *resp.body_mut() = Body::from_stream(stream);
            let h = resp.headers_mut();
            h.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            h.insert("x-accel-buffering", HeaderValue::from_static("no"));
            resp
        }
    }
}

/// Keeps a stream's concurrency permit and logs it when the stream is dropped.
struct StreamGuard {
    st: Arc<AppState>,
    log: Option<Log>,
    usage: UsageHandle,
    group: String,
    _permit: OwnedSemaphorePermit,
}

impl Drop for StreamGuard {
    /// The response stream is dropped before this guard (it is the first
    /// field of the `Map` combinator), so the stream's outcome is final here.
    fn drop(&mut self) {
        if let Some(mut log) = self.log.take() {
            log.usage = self.usage.get();
            if let Some(u) = log.usage {
                self.st.metrics.tokens(&self.group, u);
            }
            // The 200 headers are long gone, but a stream that carried an
            // upstream error or broke off is logged and counted as a failure.
            let status = if self.usage.failed() {
                self.st.metrics.upstream_error(&self.group, "stream_failed");
                502
            } else {
                200
            };
            log.emit(&self.st, status);
        }
    }
}

// ---- models ----

/// Model groups plus aliases that are not hidden, sorted.
fn model_ids(st: &AppState) -> BTreeSet<&str> {
    let groups = st.config.deployments.iter().map(|d| d.model_name.as_str());
    let aliases = st
        .config
        .router
        .model_group_alias
        .iter()
        .filter(|(_, a)| !a.hidden)
        .map(|(name, _)| name.as_str());
    groups.chain(aliases).collect()
}

fn model_object(st: &AppState, id: &str) -> Value {
    json!({"id": id, "object": "model", "created": st.started, "owned_by": "ferrolite"})
}

async fn list_models(State(st): St, headers: HeaderMap) -> Response<Body> {
    let log = match authenticate(
        &st,
        &headers,
        Log::new(&st, Route::Models),
        ErrorFormat::OpenAi,
    ) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    let data: Vec<Value> = model_ids(&st)
        .into_iter()
        .map(|id| model_object(&st, id))
        .collect();
    log.finish(
        &st,
        json_response(200, &json!({"object": "list", "data": data})),
    )
}

async fn get_model(State(st): St, headers: HeaderMap, uri: Uri) -> Response<Body> {
    let log = match authenticate(
        &st,
        &headers,
        Log::new(&st, Route::Models),
        ErrorFormat::OpenAi,
    ) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    let path = uri.path();
    let id = path
        .strip_prefix("/v1/models/")
        .or_else(|| path.strip_prefix("/models/"))
        .unwrap_or("");
    if model_ids(&st).contains(id) {
        return log.finish(&st, json_response(200, &model_object(&st, id)));
    }
    let e = ApiError::new(404, "model is not configured").with_code("model_not_found");
    log.error(&st, e, ErrorFormat::OpenAi)
}

/// Per-deployment metadata in LiteLLM's shape, without keys, `api_base`
/// or headers.
async fn model_info(State(st): St, headers: HeaderMap) -> Response<Body> {
    let log = match authenticate(
        &st,
        &headers,
        Log::new(&st, Route::ModelInfo),
        ErrorFormat::OpenAi,
    ) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    let data: Vec<Value> = st
        .config
        .deployments
        .iter()
        .map(|d| {
            let mut info = d.model_info.extra.clone();
            info.insert("id".into(), d.id.clone().into());
            for (k, v) in [
                ("mode", d.model_info.mode.clone().map(Value::from)),
                ("max_tokens", d.model_info.max_tokens.map(Value::from)),
                ("max_input_tokens", d.model_info.max_input_tokens.map(Value::from)),
                ("max_output_tokens", d.model_info.max_output_tokens.map(Value::from)),
            ] {
                if let Some(v) = v {
                    info.insert(k.into(), v);
                }
            }
            json!({
                "model_name": d.model_name,
                "litellm_params": {"model": format!("{}/{}", d.provider.prefix(), d.upstream_model)},
                "model_info": info,
            })
        })
        .collect();
    log.finish(&st, json_response(200, &json!({"data": data})))
}

// ---- health and metrics ----

async fn health_check(State(st): St, headers: HeaderMap) -> Response<Body> {
    let log = match authenticate(
        &st,
        &headers,
        Log::new(&st, Route::Health),
        ErrorFormat::OpenAi,
    ) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    let value = health::backends(&st).await;
    log.finish(&st, json_response(200, &value))
}

async fn readiness(State(st): St) -> Response<Body> {
    let log = Log::new(&st, Route::Readiness);
    let (status, value) = health::readiness(&st);
    log.finish(&st, json_response(status, &value))
}

async fn liveliness(State(st): St) -> Response<Body> {
    let log = Log::new(&st, Route::Liveliness);
    log.finish(&st, json_response(200, &json!("I'm alive!")))
}

async fn metrics(State(st): St, headers: HeaderMap) -> Response<Body> {
    let log = match authenticate(
        &st,
        &headers,
        Log::new(&st, Route::Metrics),
        ErrorFormat::OpenAi,
    ) {
        Ok(log) => log,
        Err(resp) => return *resp,
    };
    let text = st
        .metrics
        .render(st.concurrency.in_flight(), &st.router.snapshot());
    let mut resp = Response::new(Body::from(text));
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    log.finish(&st, resp)
}

async fn not_found(State(st): St) -> Response<Body> {
    let log = Log::new(&st, Route::Other);
    log.error(
        &st,
        ApiError::new(404, "no such endpoint"),
        ErrorFormat::OpenAi,
    )
}

async fn method_not_allowed(State(st): St) -> Response<Body> {
    let log = Log::new(&st, Route::Other);
    log.error(
        &st,
        ApiError::new(405, "method not allowed"),
        ErrorFormat::OpenAi,
    )
}
