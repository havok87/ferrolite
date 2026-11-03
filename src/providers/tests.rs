//! Provider tests against mock upstreams (axum on ephemeral ports), driven
//! through the router so attempts, leases and timeouts are real.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap as AxHeaders, Response, Uri};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{Value, json};

use super::*;
use crate::config::{self, LoadOptions};
use crate::router::{Hints, Router, RouterError};
use crate::secrets::{KeychainRunner, Resolver, Secret};
use crate::translate::anthropic_to_openai::tests::OPENAI_STREAM;
use crate::translate::openai_to_anthropic::tests::ANTHROPIC_STREAM;

const UPSTREAM_KEY: &str = "sk-upstream-key-0001";
const ANTHROPIC_KEY: &str = "sk-ant-key-0000001";
const HEADER_SECRET: &str = "hdr-secret-0000001";

#[derive(Clone)]
enum Reply {
    Json(u16, Value, Vec<(&'static str, &'static str)>),
    Sse(Vec<String>, bool),
    Redirect(&'static str),
}

#[derive(Debug, Clone)]
struct Seen {
    path: String,
    headers: AxHeaders,
    body: Value,
}

struct Mock {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Mock {
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    fn last(&self) -> Seen {
        self.seen().pop().expect("the mock was not called")
    }
}

async fn mock(reply: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Mock {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let reply = Arc::new(reply);
    let log = Arc::clone(&seen);
    let app = axum::Router::new().fallback(move |uri: Uri, headers: AxHeaders, body: Bytes| {
        let log = Arc::clone(&log);
        let reply = Arc::clone(&reply);
        async move {
            let req = Seen {
                path: uri.path().to_owned(),
                headers,
                body: serde_json::from_slice(&body).unwrap_or(Value::Null),
            };
            let r = reply(&req);
            log.lock().unwrap().push(req);
            respond(r)
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Mock { port, seen }
}

fn respond(reply: Reply) -> Response<Body> {
    match reply {
        Reply::Json(status, v, headers) => {
            let mut b = Response::builder()
                .status(status)
                .header("content-type", "application/json");
            for (k, v) in headers {
                b = b.header(k, v);
            }
            b.body(Body::from(v.to_string())).unwrap()
        }
        Reply::Sse(chunks, stall) => {
            let items = futures_util::stream::iter(
                chunks
                    .into_iter()
                    .map(|c| Ok::<_, std::io::Error>(Bytes::from(c))),
            );
            let body = if stall {
                Body::from_stream(items.chain(futures_util::stream::pending()))
            } else {
                Body::from_stream(items)
            };
            Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(body)
                .unwrap()
        }
        Reply::Redirect(location) => Response::builder()
            .status(302)
            .header("location", location)
            .body(Body::empty())
            .unwrap(),
    }
}

/// Splits a stream fixture into awkward network-sized pieces.
fn chunks(s: &str, size: usize) -> Vec<String> {
    s.as_bytes()
        .chunks(size)
        .map(|c| String::from_utf8(c.to_vec()).unwrap())
        .collect()
}

struct NoKeychain;

impl KeychainRunner for NoKeychain {
    fn find_generic_password(&self, _: &str, _: &str) -> Result<Secret, String> {
        Err("no keychain in tests".to_owned())
    }
}

struct Gateway {
    router: Router,
    upstream: Upstream,
}

fn gateway(port: u16, options: UpstreamOptions) -> Gateway {
    let yaml = format!(
        "
model_list:
  - model_name: oa
    litellm_params:
      model: hosted_vllm/served-model
      api_base: http://127.0.0.1:{port}/v1
      api_key: os.environ/UP_KEY
      temperature: 0.1
      extra_headers:
        X-Team: blue
        X-Upstream-Token: os.environ/HDR
  - model_name: oa-tpm
    litellm_params:
      model: hosted_vllm/served-model
      api_base: http://127.0.0.1:{port}/v1
      tpm: 1000000
  - model_name: oa-slow
    litellm_params:
      model: hosted_vllm/served-model
      api_base: http://127.0.0.1:{port}/v1
      stream_timeout: 1
  - model_name: claude
    litellm_params:
      model: anthropic/claude-up
      api_base: http://127.0.0.1:{port}
      api_key: os.environ/ANT_KEY
  - model_name: closed
    litellm_params:
      model: hosted_vllm/x
      api_base: http://127.0.0.1:9/v1
"
    );
    let env: HashMap<String, String> = [
        ("UP_KEY", UPSTREAM_KEY),
        ("ANT_KEY", ANTHROPIC_KEY),
        ("HDR", HEADER_SECRET),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let resolver = Resolver::new(&env, &NoKeychain);
    let opts = LoadOptions {
        bind: "127.0.0.1".parse().unwrap(),
        allow_no_auth: true,
    };
    let loaded =
        config::load_str(&yaml, Path::new("t.yaml"), Path::new("/"), &opts, &resolver).unwrap();
    Gateway {
        router: Router::with_seed(&loaded.config, 1),
        upstream: Upstream::new(&loaded.config, &loaded.redactor, options).unwrap(),
    }
}

impl Gateway {
    async fn call(
        &self,
        model: &str,
        request: ProxyRequest,
    ) -> Result<ProxyResponse, RouterError<ProviderError>> {
        let hints = Hints {
            stream: request.stream(),
        };
        self.router
            .execute(model, hints, |attempt| {
                self.upstream.send(attempt, &request)
            })
            .await
            .map(|routed| routed.value)
    }
}

fn request(endpoint: Endpoint, body: Value) -> ProxyRequest {
    let Value::Object(map) = body else {
        panic!("body must be an object")
    };
    ProxyRequest::new(endpoint, map)
}

fn json_body(r: ProxyResponse) -> (Value, Option<Usage>) {
    match r {
        ProxyResponse::Json { body, usage } => (serde_json::from_slice(&body).unwrap(), usage),
        ProxyResponse::Stream { .. } => panic!("expected JSON"),
    }
}

async fn stream_text(r: ProxyResponse) -> (String, UsageHandle) {
    match r {
        ProxyResponse::Stream { body, usage } => {
            let parts: Vec<Bytes> = body.collect().await;
            (String::from_utf8(parts.concat()).unwrap(), usage)
        }
        ProxyResponse::Json { .. } => panic!("expected a stream"),
    }
}

fn upstream_error(e: RouterError<ProviderError>) -> ProviderError {
    match e {
        RouterError::Upstream { error, .. } => error,
        other => panic!("expected an upstream error, got {other:?}"),
    }
}

fn header<'a>(seen: &'a Seen, name: &str) -> Option<&'a str> {
    seen.headers.get(name).and_then(|v| v.to_str().ok())
}

const CHAT: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","model":"served-model","choices":[{"index":0,"message":{"role":"assistant","content":"Hi!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;

fn chat_completion() -> Value {
    serde_json::from_str(CHAT).unwrap()
}

fn anthropic_message() -> Value {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-up",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 3},
    })
}

// ---- passthrough ----

#[tokio::test]
async fn openai_passthrough_uses_server_side_auth_only() {
    let m = mock(|_| Reply::Json(200, chat_completion(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let resp = g
        .call(
            "oa",
            request(
                Endpoint::ChatCompletions,
                json!({
                    "model": "oa",
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_p": 0.5,
                    "api_base": "http://evil.example/v1",
                    "api_key": "sk-client-supplied",
                    "extra_headers": {"X-Evil": "1"},
                    "litellm_logging_obj": {},
                }),
            ),
        )
        .await
        .unwrap();
    let (body, usage) = json_body(resp);
    assert_eq!(body, chat_completion());
    assert_eq!(usage.map(Usage::total), Some(7));

    let seen = m.last();
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(
        header(&seen, "authorization"),
        Some(&*format!("Bearer {UPSTREAM_KEY}"))
    );
    assert_eq!(header(&seen, "x-team"), Some("blue"));
    assert_eq!(header(&seen, "x-upstream-token"), Some(HEADER_SECRET));
    assert!(header(&seen, "x-evil").is_none());
    assert!(
        header(&seen, "user-agent")
            .unwrap()
            .starts_with("ferrolite/")
    );
    assert_eq!(
        seen.body,
        json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.5,
            "temperature": 0.1,
        })
    );
}

#[tokio::test]
async fn passthrough_endpoints_use_their_paths() {
    let m = mock(|_| {
        Reply::Json(
            200,
            json!({"object": "list", "data": [], "usage": {"prompt_tokens": 3}}),
            vec![],
        )
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    for (endpoint, path) in [
        (Endpoint::Completions, "/v1/completions"),
        (Endpoint::Embeddings, "/v1/embeddings"),
        (Endpoint::Responses, "/v1/responses"),
    ] {
        g.call(
            "oa",
            request(endpoint, json!({"model": "oa", "input": "x"})),
        )
        .await
        .unwrap();
        assert_eq!(m.last().path, path);
    }
}

#[tokio::test]
async fn anthropic_passthrough_forwards_only_allowed_headers() {
    let m = mock(|_| Reply::Json(200, anthropic_message(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let mut req = request(
        Endpoint::Messages,
        json!({"model": "claude", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
    );
    req.anthropic_version = Some("2024-01-01".into());
    req.anthropic_beta = Some("prompt-caching-2024-07-31".into());
    let (body, usage) = json_body(g.call("claude", req).await.unwrap());
    assert_eq!(body, anthropic_message());
    assert_eq!(
        usage,
        Some(Usage {
            prompt: 9,
            completion: 3
        })
    );

    let seen = m.last();
    assert_eq!(seen.path, "/v1/messages");
    assert_eq!(header(&seen, "x-api-key"), Some(ANTHROPIC_KEY));
    assert_eq!(header(&seen, "anthropic-version"), Some("2024-01-01"));
    assert_eq!(
        header(&seen, "anthropic-beta"),
        Some("prompt-caching-2024-07-31")
    );
    assert!(header(&seen, "authorization").is_none());
    assert_eq!(seen.body["model"], "claude-up");
}

// ---- translation ----

#[tokio::test]
async fn chat_to_anthropic_and_back() {
    let m = mock(|_| Reply::Json(200, anthropic_message(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let (body, usage) = json_body(
        g.call(
            "claude",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "claude", "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"},
                ]}),
            ),
        )
        .await
        .unwrap(),
    );
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        usage,
        Some(Usage {
            prompt: 9,
            completion: 3
        })
    );

    let seen = m.last();
    assert_eq!(seen.path, "/v1/messages");
    assert_eq!(
        header(&seen, "anthropic-version"),
        Some(anthropic::DEFAULT_ANTHROPIC_VERSION)
    );
    assert_eq!(seen.body["system"], "Be brief.");
    assert_eq!(seen.body["model"], "claude-up");
    assert_eq!(seen.body["max_tokens"], 4096);
}

#[tokio::test]
async fn messages_to_openai_and_back() {
    let m = mock(|_| Reply::Json(200, chat_completion(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let (body, _) = json_body(
        g.call(
            "oa",
            request(
                Endpoint::Messages,
                json!({"model": "oa", "max_tokens": 50, "system": "Be brief.",
                       "messages": [{"role": "user", "content": "hi"}]}),
            ),
        )
        .await
        .unwrap(),
    );
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"], json!([{"type": "text", "text": "Hi!"}]));
    assert_eq!(body["usage"]["input_tokens"], 5);

    let seen = m.last();
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(
        seen.body,
        json!({
            "model": "served-model",
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "temperature": 0.1,
        })
    );
}

#[tokio::test]
async fn translation_errors_never_reach_the_upstream() {
    let m = mock(|_| Reply::Json(200, anthropic_message(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let e = upstream_error(
        g.call(
            "claude",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "claude", "messages": [], "logprobs": true}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert_eq!((e.class, e.status), (ErrorClass::BadRequest, 400));
    assert!(e.message.contains("logprobs"));

    let e = upstream_error(
        g.call(
            "claude",
            request(
                Endpoint::Embeddings,
                json!({"model": "claude", "input": "x"}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert!(e.message.contains("not supported for anthropic/"), "{e}");
    assert!(m.seen().is_empty());
}

// ---- streaming ----

#[tokio::test]
async fn anthropic_stream_becomes_chat_chunks() {
    let m = mock(|_| Reply::Sse(chunks(ANTHROPIC_STREAM, 37), false)).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let resp = g
        .call(
            "claude",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "claude", "stream": true, "stream_options": {"include_usage": true},
                       "messages": [{"role": "user", "content": "hi"}]}),
            ),
        )
        .await
        .unwrap();
    let (text, usage) = stream_text(resp).await;
    assert!(text.contains("\"content\":\"Hel\""), "{text}");
    assert!(text.contains("\"finish_reason\":\"tool_calls\""));
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        usage.get(),
        Some(Usage {
            prompt: 16,
            completion: 30
        })
    );
    assert_eq!(m.last().body["stream"], true);
    assert_eq!(header(&m.last(), "accept"), Some("text/event-stream"));
}

#[tokio::test]
async fn openai_stream_becomes_anthropic_events() {
    let m = mock(|_| Reply::Sse(chunks(OPENAI_STREAM, 23), false)).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let resp = g
        .call(
            "oa",
            request(
                Endpoint::Messages,
                json!({"model": "oa", "max_tokens": 10, "stream": true,
                       "messages": [{"role": "user", "content": "hi"}]}),
            ),
        )
        .await
        .unwrap();
    let (text, usage) = stream_text(resp).await;
    assert!(text.starts_with("event: message_start\n"), "{text}");
    assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
    assert_eq!(
        usage.get(),
        Some(Usage {
            prompt: 11,
            completion: 9
        })
    );
    assert_eq!(
        m.last().body["stream_options"],
        json!({"include_usage": true})
    );
}

#[tokio::test]
async fn passthrough_stream_counts_usage_without_showing_it() {
    let m = mock(|_| Reply::Sse(chunks(OPENAI_STREAM, 41), false)).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let body =
        json!({"model": "oa-tpm", "stream": true, "messages": [{"role": "user", "content": "hi"}]});
    let (text, usage) = stream_text(
        g.call("oa-tpm", request(Endpoint::ChatCompletions, body))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(
        m.last().body["stream_options"],
        json!({"include_usage": true})
    );
    assert!(!text.contains("\"usage\""), "{text}");
    assert!(text.contains("\"content\":\"Hel\""));
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert_eq!(
        usage.get(),
        Some(Usage {
            prompt: 11,
            completion: 9
        })
    );

    // A client that asks for usage gets the upstream's chunk unchanged.
    let body = json!({"model": "oa-tpm", "stream": true, "stream_options": {"include_usage": true},
                      "messages": [{"role": "user", "content": "hi"}]});
    let (text, _) = stream_text(
        g.call("oa-tpm", request(Endpoint::ChatCompletions, body))
            .await
            .unwrap(),
    )
    .await;
    assert!(text.contains("\"usage\""));
}

#[tokio::test]
async fn stalled_streams_are_cut_off() {
    let first = OPENAI_STREAM[..OPENAI_STREAM.find("\n\n").unwrap() + 2].to_owned();
    let m = mock(move |_| Reply::Sse(vec![first.clone()], true)).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let started = Instant::now();
    let resp = g
        .call(
            "oa-slow",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "oa-slow", "stream": true, "messages": []}),
            ),
        )
        .await
        .unwrap();
    let (text, _) = stream_text(resp).await;
    assert!(text.contains("sent nothing for 1s"), "{text}");
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert!(started.elapsed() < Duration::from_secs(5));
}

// ---- errors ----

#[tokio::test]
async fn classifies_upstream_errors() {
    let m = mock(|req| match req.body["messages"][0]["content"].as_str().unwrap_or("") {
        "429" => Reply::Json(429, json!({"error": {"message": "slow down"}}), vec![("retry-after", "3")]),
        "429ms" => Reply::Json(429, json!({"error": {"message": "slow down"}}), vec![("retry-after-ms", "250")]),
        "529" => Reply::Json(529, json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}), vec![]),
        "ctx" => Reply::Json(400, json!({"error": {"message": "too long", "code": "context_length_exceeded"}}), vec![]),
        "ctx-anthropic" => Reply::Json(400, json!({"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 300000 tokens > 200000 maximum"}}), vec![]),
        "policy" => Reply::Json(400, json!({"error": {"message": "blocked", "code": "content_filter"}}), vec![]),
        "auth" => Reply::Json(401, json!({"error": "bad key"}), vec![]),
        "missing" => Reply::Json(404, json!({"object": "error", "message": "model not found"}), vec![]),
        _ => Reply::Json(400, json!({"nothing": "useful"}), vec![]),
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    for (content, class, status) in [
        (
            "429",
            ErrorClass::RateLimited {
                retry_after: Some(Duration::from_secs(3)),
            },
            429,
        ),
        (
            "429ms",
            ErrorClass::RateLimited {
                retry_after: Some(Duration::from_millis(250)),
            },
            429,
        ),
        ("529", ErrorClass::Server, 529),
        ("ctx", ErrorClass::ContextWindowExceeded, 400),
        ("ctx-anthropic", ErrorClass::ContextWindowExceeded, 400),
        ("policy", ErrorClass::ContentPolicy, 400),
        ("auth", ErrorClass::Auth, 401),
        ("missing", ErrorClass::NotFound, 404),
        ("other", ErrorClass::BadRequest, 400),
    ] {
        let e = upstream_error(
            g.call(
                "oa",
                request(
                    Endpoint::ChatCompletions,
                    json!({"model": "oa", "messages": [{"role": "user", "content": content}]}),
                ),
            )
            .await
            .unwrap_err(),
        );
        assert_eq!((e.class, e.status), (class, status), "{content}: {e}");
    }
}

#[tokio::test]
async fn error_messages_are_scrubbed_and_truncated() {
    let m = mock(|_| {
        Reply::Json(
            500,
            json!({"error": {"message": format!(
                "auth {UPSTREAM_KEY} failed calling http://127.0.0.1:1234/v1 {}", "x".repeat(5000)
            )}}),
            vec![],
        )
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    let e = upstream_error(
        g.call(
            "oa",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "oa", "messages": []}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert!(!e.message.contains(UPSTREAM_KEY), "{e}");
    assert!(!e.message.contains("127.0.0.1"), "{e}");
    assert!(e.message.contains("[REDACTED]"));
    assert!(e.message.chars().count() <= 1025);
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let m = mock(|req| {
        if req.path == "/stolen" {
            Reply::Json(200, chat_completion(), vec![])
        } else {
            Reply::Redirect("/stolen")
        }
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    let e = upstream_error(
        g.call(
            "oa",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "oa", "messages": []}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert_eq!((e.class, e.status), (ErrorClass::NotFound, 502));
    assert!(e.message.contains("redirects are not followed"), "{e}");
    assert!(m.seen().iter().all(|s| s.path != "/stolen"));
}

#[tokio::test]
async fn oversized_responses_are_refused() {
    let m = mock(|_| Reply::Json(200, json!({"data": "x".repeat(10_000)}), vec![])).await;
    let g = gateway(
        m.port,
        UpstreamOptions {
            max_response_bytes: 1024,
            ..UpstreamOptions::default()
        },
    );
    let e = upstream_error(
        g.call(
            "oa",
            request(Endpoint::Embeddings, json!({"model": "oa", "input": "x"})),
        )
        .await
        .unwrap_err(),
    );
    assert_eq!(e.status, 502);
    assert!(e.message.contains("larger than 1024 bytes"), "{e}");
}

#[tokio::test]
async fn unreachable_upstreams_are_connection_errors() {
    let g = gateway(1, UpstreamOptions::default());
    let e = upstream_error(
        g.call(
            "closed",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "closed", "messages": []}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert_eq!((e.class, e.status), (ErrorClass::Connection, 502));
    assert!(!e.message.contains("127.0.0.1"));
}

#[test]
fn url_building() {
    let base = |s: &str| url::Url::parse(s).unwrap();
    assert_eq!(
        openai_compat::url(&base("http://h:1/v1"), Endpoint::ChatCompletions).as_str(),
        "http://h:1/v1/chat/completions"
    );
    assert_eq!(
        openai_compat::url(&base("http://h:1/v1/"), Endpoint::Embeddings).as_str(),
        "http://h:1/v1/embeddings"
    );
    for (b, expected) in [
        (
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://proxy.example/v1",
            "https://proxy.example/v1/messages",
        ),
        (
            "https://proxy.example/v1/messages",
            "https://proxy.example/v1/messages",
        ),
        (
            "https://proxy.example/anthropic/",
            "https://proxy.example/anthropic/v1/messages",
        ),
    ] {
        assert_eq!(anthropic::messages_url(&base(b)).as_str(), expected);
    }
    assert_eq!(
        anthropic::models_url(&base("https://proxy.example/v1/messages")).as_str(),
        "https://proxy.example/v1/models"
    );
    assert_eq!(
        openai_compat::models_url(&base("http://h:1/v1")).as_str(),
        "http://h:1/v1/models"
    );
}

#[tokio::test]
async fn json_answers_to_stream_requests_are_errors() {
    let m = mock(|_| {
        Reply::Json(
            200,
            json!({"error": {"message": "streaming is not supported"}}),
            vec![],
        )
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    let e = upstream_error(
        g.call(
            "oa",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "oa", "stream": true, "messages": []}),
            ),
        )
        .await
        .unwrap_err(),
    );
    assert_eq!((e.class, e.status), (ErrorClass::Server, 502));
    assert!(
        e.message
            .contains("without an event stream: streaming is not supported"),
        "{e}"
    );
}

#[tokio::test]
async fn empty_and_failed_streams_are_reported() {
    let m = mock(|req| match req.body["model"].as_str() {
        Some("served-model") if req.body["messages"][0]["content"] == "empty" => {
            Reply::Sse(vec![], false)
        }
        _ => Reply::Sse(
            vec![
                OPENAI_STREAM[..OPENAI_STREAM.find("\n\n").unwrap() + 2].to_owned(),
                "data: {\"error\":{\"message\":\"model crashed\"}}\n\n".to_owned(),
            ],
            false,
        ),
    })
    .await;
    let g = gateway(m.port, UpstreamOptions::default());
    let body = |content: &str| json!({"model": "oa", "stream": true, "messages": [{"role": "user", "content": content}]});

    let (text, outcome) = stream_text(
        g.call("oa", request(Endpoint::ChatCompletions, body("empty")))
            .await
            .unwrap(),
    )
    .await;
    assert!(text.contains("ended without sending any events"), "{text}");
    assert!(text.ends_with("data: [DONE]\n\n"));
    assert!(outcome.failed());

    let (text, outcome) = stream_text(
        g.call("oa", request(Endpoint::ChatCompletions, body("crash")))
            .await
            .unwrap(),
    )
    .await;
    assert!(text.contains("model crashed"), "{text}");
    assert!(outcome.failed());

    let (_, outcome) = stream_text(
        g.call(
            "claude",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "claude", "stream": true, "messages": []}),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    // The claude deployment's mock answer is not an Anthropic stream, so
    // the translator ends it with an error.
    assert!(outcome.failed());
}

#[tokio::test]
async fn completed_streams_are_not_failures() {
    let m = mock(|_| Reply::Sse(chunks(OPENAI_STREAM, 50), false)).await;
    let g = gateway(m.port, UpstreamOptions::default());
    let (_, outcome) = stream_text(
        g.call(
            "oa",
            request(
                Endpoint::ChatCompletions,
                json!({"model": "oa", "stream": true, "messages": []}),
            ),
        )
        .await
        .unwrap(),
    )
    .await;
    assert!(!outcome.failed());
    assert_eq!(
        outcome.get(),
        Some(Usage {
            prompt: 11,
            completion: 9
        })
    );
}

#[tokio::test]
async fn client_steered_upstream_resources_never_reach_the_upstream() {
    let m = mock(|_| Reply::Json(200, anthropic_message(), vec![])).await;
    let g = gateway(m.port, UpstreamOptions::default());
    for (model, endpoint, body) in [
        (
            "claude",
            Endpoint::Messages,
            json!({"model": "claude", "max_tokens": 5, "messages": [],
            "mcp_servers": [{"type": "url", "url": "https://attacker.example/mcp", "name": "x"}]}),
        ),
        (
            "oa",
            Endpoint::Responses,
            json!({"model": "oa", "input": "hi",
            "tools": [{"type": "mcp", "server_label": "x", "server_url": "https://attacker.example"}]}),
        ),
        (
            "oa",
            Endpoint::ChatCompletions,
            json!({"model": "oa", "messages": [{"role": "user",
            "content": [{"type": "file", "file": {"file_id": "file-abc"}}]}]}),
        ),
    ] {
        let e = upstream_error(g.call(model, request(endpoint, body)).await.unwrap_err());
        assert_eq!((e.class, e.status), (ErrorClass::BadRequest, 400), "{e}");
    }
    assert!(m.seen().is_empty());
}
