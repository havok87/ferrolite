//! End-to-end tests: a real ferrolite server on an ephemeral port in front
//! of mock OpenAI-compatible and Anthropic upstreams.

#![forbid(unsafe_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::Method;
use common::*;
use ferrolite::server::{ServeOptions, ServerOptions, tls};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const OPENAI_SSE: &[&str] = &[
    "data: {\"id\":\"c1\",\"model\":\"up\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    "data: {\"id\":\"c1\",\"model\":\"up\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
    "data: {\"id\":\"c1\",\"model\":\"up\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"c1\",\"model\":\"up\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
    "data: [DONE]\n\n",
];

const ANTHROPIC_SSE: &str = concat!(
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"up-claude\",\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n",
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
);

fn chat(content: &str) -> Value {
    json!({
        "id": "chatcmpl-1", "object": "chat.completion", "model": "up",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    })
}

fn sse(chunks: &[&str], delay: Duration, stall: bool) -> Reply {
    Reply::Sse {
        chunks: chunks.iter().map(|c| (*c).to_owned()).collect(),
        delay,
        stall,
    }
}

/// The working upstream: answers by the upstream model in the body.
fn good(req: &Seen) -> Reply {
    if req.method == Method::GET {
        return Reply::Json(200, json!({"object": "list", "data": []}));
    }
    let stream = req.body["stream"] == true;
    match req.body["model"].as_str().unwrap_or("") {
        "up-claude" if stream => sse(&[ANTHROPIC_SSE], Duration::ZERO, false),
        "up-claude" => Reply::Json(
            200,
            json!({
                "id": "msg_1", "type": "message", "role": "assistant", "model": "up-claude",
                "content": [{"type": "text", "text": "Hello"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }),
        ),
        "up-small" => Reply::Json(
            400,
            json!({"error": {"message": "maximum context length is 10 tokens", "code": "context_length_exceeded"}}),
        ),
        "up-big" => Reply::Json(200, chat("from the big model")),
        "up-leak" => Reply::Json(
            500,
            json!({"error": {"message": format!("key {UPSTREAM_KEY} rejected by http://127.0.0.1:1/v1")}}),
        ),
        "up-slow" => sse(OPENAI_SSE, Duration::from_millis(200), false),
        "up-stall" => sse(&OPENAI_SSE[..1], Duration::ZERO, true),
        "up-crash" => sse(
            &[
                OPENAI_SSE[0],
                "data: {\"error\":{\"message\":\"model crashed\"}}\n\n",
            ],
            Duration::ZERO,
            false,
        ),
        "up-slowjson" => Reply::JsonAfter(Duration::from_secs(2), 200, chat("late")),
        _ if stream => sse(OPENAI_SSE, Duration::ZERO, false),
        _ => Reply::Json(200, chat("Hi")),
    }
}

/// The broken upstream.
fn down(_: &Seen) -> Reply {
    Reply::Json(500, json!({"error": {"message": "down"}}))
}

fn config_yaml(a: u16, b: u16) -> String {
    format!(
        r#"
model_list:
  - {{model_name: gpt, litellm_params: {{model: hosted_vllm/up-gpt, api_base: "http://127.0.0.1:{a}/v1", api_key: os.environ/UP_KEY}}}}
  - {{model_name: claude, litellm_params: {{model: anthropic/up-claude, api_base: "http://127.0.0.1:{a}", api_key: os.environ/ANT_KEY}}}}
  - {{model_name: flaky, litellm_params: {{model: hosted_vllm/up-flaky, api_base: "http://127.0.0.1:{b}/v1", order: 1}}}}
  - {{model_name: flaky, litellm_params: {{model: hosted_vllm/up-flaky-good, api_base: "http://127.0.0.1:{a}/v1", order: 2}}}}
  - {{model_name: primary, litellm_params: {{model: hosted_vllm/up-primary, api_base: "http://127.0.0.1:{b}/v1"}}}}
  - {{model_name: small, litellm_params: {{model: hosted_vllm/up-small, api_base: "http://127.0.0.1:{a}/v1"}}}}
  - {{model_name: big, litellm_params: {{model: hosted_vllm/up-big, api_base: "http://127.0.0.1:{a}/v1"}}}}
  - {{model_name: leak, litellm_params: {{model: hosted_vllm/up-leak, api_base: "http://127.0.0.1:{a}/v1", api_key: os.environ/UP_KEY}}}}
  - {{model_name: slow, litellm_params: {{model: hosted_vllm/up-slow, api_base: "http://127.0.0.1:{a}/v1"}}}}
  - {{model_name: stall, litellm_params: {{model: hosted_vllm/up-stall, api_base: "http://127.0.0.1:{a}/v1"}}}}
  - {{model_name: limited, litellm_params: {{model: hosted_vllm/up-gpt, api_base: "http://127.0.0.1:{a}/v1", rpm: 1}}}}
  - {{model_name: crash, litellm_params: {{model: hosted_vllm/up-crash, api_base: "http://127.0.0.1:{a}/v1"}}}}
  - {{model_name: slowjson, litellm_params: {{model: hosted_vllm/up-slowjson, api_base: "http://127.0.0.1:{a}/v1"}}}}
router_settings:
  num_retries: 0
  allowed_fails: 0
  cooldown_time: 60
  fallbacks: [{{primary: [gpt]}}]
  context_window_fallbacks: [{{small: [big]}}]
  model_group_alias:
    hidden-gpt: {{model: gpt, hidden: true}}
    smart: claude
general_settings:
  master_key: os.environ/MASTER
"#
    )
}

struct Setup {
    good: Mock,
    down: Mock,
    gw: Gateway,
}

async fn setup_with(options: ServerOptions, serve: ServeOptions) -> Setup {
    let good = Mock::start(good).await;
    let down = Mock::start(down).await;
    let gw = Gateway::start(
        load(&config_yaml(good.port, down.port), false),
        options,
        serve,
    )
    .await;
    Setup { good, down, gw }
}

async fn setup() -> Setup {
    setup_with(ServerOptions::default(), ServeOptions::default()).await
}

fn chat_body(model: &str) -> Value {
    json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
}

// ---- auth ----

#[tokio::test]
async fn auth_is_required_except_for_liveliness_and_readiness() {
    let s = setup().await;
    for path in [
        "/v1/models",
        "/models",
        "/model/info",
        "/health",
        "/metrics",
        "/v1/models/gpt",
    ] {
        let (status, _, body) = json(s.gw.get(path, false).await).await;
        assert_eq!(status, 401, "{path}");
        assert_eq!(body["error"]["type"], "authentication_error", "{path}");
    }
    let (status, _, _) = json(
        s.gw.post_with("/v1/chat/completions", &chat_body("gpt"), &[])
            .await,
    )
    .await;
    assert_eq!(status, 401);
    let (status, _, body) = json(
        s.gw.post_with(
            "/v1/messages",
            &chat_body("claude"),
            &[("x-api-key", "sk-wrong-key-0000000000000000000000")],
        )
        .await,
    )
    .await;
    assert_eq!(status, 401);
    assert_eq!(
        body,
        json!({"type": "error", "error": {"type": "authentication_error", "message": "missing or invalid API key"}})
    );
    assert!(
        s.good.seen().is_empty(),
        "an unauthenticated request reached the upstream"
    );

    for path in ["/health/liveliness", "/health/readiness"] {
        assert_eq!(s.gw.get(path, false).await.status(), 200, "{path}");
    }
}

#[tokio::test]
async fn every_key_header_is_accepted() {
    let s = setup().await;
    for (name, value) in [
        ("authorization", format!("Bearer {MASTER_KEY}")),
        ("x-api-key", MASTER_KEY.to_owned()),
        ("api-key", MASTER_KEY.to_owned()),
    ] {
        let resp =
            s.gw.client
                .get(s.gw.url("/v1/models"))
                .header(name, value)
                .send()
                .await
                .unwrap();
        assert_eq!(resp.status(), 200, "{name}");
    }
}

#[tokio::test]
async fn allow_no_auth_serves_without_a_key() {
    let good = Mock::start(good).await;
    let yaml = format!(
        "model_list:\n  - {{model_name: gpt, litellm_params: {{model: hosted_vllm/up-gpt, api_base: \"http://127.0.0.1:{}/v1\"}}}}\n",
        good.port
    );
    let gw = Gateway::start(
        load(&yaml, true),
        ServerOptions::default(),
        ServeOptions::default(),
    )
    .await;
    let (status, _, _) = json(
        gw.post_with("/v1/chat/completions", &chat_body("gpt"), &[])
            .await,
    )
    .await;
    assert_eq!(status, 200);
}

// ---- proxying ----

#[tokio::test]
async fn passthrough_strips_client_routing_fields_and_auth() {
    let s = setup().await;
    let body = json!({
        "model": "gpt",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": "http://evil.example/v1",
        "base_url": "http://evil.example/v1",
        "api_key": "sk-client-supplied",
        "extra_headers": {"X-Evil": "1"},
        "litellm_trace_id": "t",
    });
    for path in ["/v1/chat/completions", "/chat/completions"] {
        let (status, headers, resp) = json(s.gw.post(path, &body).await).await;
        assert_eq!(status, 200, "{path}");
        assert_eq!(resp, chat("Hi"));
        assert_eq!(headers["content-type"], "application/json");
        assert_eq!(headers["x-request-id"].len(), 36);
    }
    let seen = s.good.calls_for("up-gpt").pop().unwrap();
    assert_eq!(seen.path, "/v1/chat/completions");
    assert_eq!(
        seen.headers["authorization"],
        format!("Bearer {UPSTREAM_KEY}").as_str()
    );
    assert!(seen.headers.get("x-evil").is_none());
    assert_eq!(
        seen.body,
        json!({"model": "up-gpt", "messages": [{"role": "user", "content": "hi"}]})
    );
}

#[tokio::test]
async fn other_passthrough_endpoints() {
    let s = setup().await;
    for path in [
        "/v1/completions",
        "/completions",
        "/v1/embeddings",
        "/embeddings",
        "/v1/responses",
        "/responses",
    ] {
        let (status, _, _) = json(
            s.gw.post(path, &json!({"model": "gpt", "input": "x"}))
                .await,
        )
        .await;
        assert_eq!(status, 200, "{path}");
        let upstream_path = s.good.seen().pop().unwrap().path;
        assert_eq!(
            upstream_path,
            format!(
                "/v1/{}",
                path.trim_start_matches("/v1/").trim_start_matches('/')
            )
        );
    }
}

#[tokio::test]
async fn streaming_passthrough() {
    let s = setup().await;
    let mut body = chat_body("gpt");
    body["stream"] = true.into();
    let resp = s.gw.post("/v1/chat/completions", &body).await;
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    assert_eq!(resp.headers()["cache-control"], "no-cache");
    let text = text(resp).await;
    assert_eq!(text, OPENAI_SSE.concat());
}

#[tokio::test]
async fn chat_completions_to_anthropic() {
    let s = setup().await;
    let (status, _, body) = json(
        s.gw.post("/v1/chat/completions", &chat_body("claude"))
            .await,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello");
    let seen = s.good.calls_for("up-claude").pop().unwrap();
    assert_eq!(seen.path, "/v1/messages");
    assert_eq!(seen.headers["x-api-key"], ANTHROPIC_KEY);
    assert!(seen.headers.get("authorization").is_none());

    let mut streaming = chat_body("smart");
    streaming["stream"] = true.into();
    let text = text(s.gw.post("/v1/chat/completions", &streaming).await).await;
    assert!(text.contains("\"content\":\"Hello\""), "{text}");
    assert!(text.ends_with("data: [DONE]\n\n"));
}

#[tokio::test]
async fn messages_to_openai_backend_and_passthrough() {
    let s = setup().await;
    let body =
        json!({"model": "gpt", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]});
    let (status, _, resp) = json(s.gw.post("/v1/messages", &body).await).await;
    assert_eq!(status, 200);
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["content"], json!([{"type": "text", "text": "Hi"}]));
    assert_eq!(
        s.good.calls_for("up-gpt").pop().unwrap().path,
        "/v1/chat/completions"
    );

    let body = json!({"model": "claude", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]});
    let resp =
        s.gw.post_with(
            "/v1/messages",
            &body,
            &[
                ("x-api-key", MASTER_KEY),
                ("anthropic-version", "2023-06-01"),
                ("anthropic-beta", "prompt-caching-2024-07-31"),
            ],
        )
        .await;
    let (status, _, resp) = json(resp).await;
    assert_eq!(status, 200);
    assert_eq!(resp["content"][0]["text"], "Hello");
    let seen = s.good.calls_for("up-claude").pop().unwrap();
    assert_eq!(seen.headers["anthropic-beta"], "prompt-caching-2024-07-31");
    assert_eq!(
        seen.headers["x-api-key"], ANTHROPIC_KEY,
        "the client's key must not be forwarded"
    );
}

// ---- routing ----

#[tokio::test]
async fn falls_back_to_another_group() {
    let s = setup().await;
    let (status, _, body) = json(
        s.gw.post("/v1/chat/completions", &chat_body("primary"))
            .await,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["choices"][0]["message"]["content"], "Hi");
    assert_eq!(s.down.calls_for("up-primary").len(), 1);
    assert_eq!(s.good.calls_for("up-gpt").len(), 1);
}

#[tokio::test]
async fn context_window_errors_use_their_fallbacks() {
    let s = setup().await;
    let (status, _, body) =
        json(s.gw.post("/v1/chat/completions", &chat_body("small")).await).await;
    assert_eq!(status, 200);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "from the big model"
    );
}

#[tokio::test]
async fn failing_deployments_cool_down_across_requests() {
    let s = setup().await;
    let (status, _, body) =
        json(s.gw.post("/v1/chat/completions", &chat_body("flaky")).await).await;
    assert_eq!(status, 502, "{body}");
    let (status, _, _) = json(s.gw.post("/v1/chat/completions", &chat_body("flaky")).await).await;
    assert_eq!(status, 200);
    assert_eq!(
        s.down.calls_for("up-flaky").len(),
        1,
        "the cooled-down deployment was called again"
    );
    assert_eq!(s.good.calls_for("up-flaky-good").len(), 1);

    let metrics = text(s.gw.get("/metrics", true).await).await;
    assert!(
        metrics.contains(
            "ferrolite_deployment_cooling_down{model_group=\"flaky\",deployment=\"flaky#0\"} 1"
        ),
        "{metrics}"
    );
    assert!(
        metrics
            .contains("ferrolite_upstream_errors_total{model_group=\"flaky\",class=\"server\"} 1")
    );
}

#[tokio::test]
async fn no_available_deployment_is_429() {
    let s = setup().await;
    assert_eq!(
        s.gw.post("/v1/chat/completions", &chat_body("limited"))
            .await
            .status(),
        200
    );
    let (status, headers, body) = json(
        s.gw.post("/v1/chat/completions", &chat_body("limited"))
            .await,
    )
    .await;
    assert_eq!(status, 429);
    // The rpm window resets within the minute.
    let after: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((1..=60).contains(&after), "{after}");
    assert_eq!(body["error"]["type"], "rate_limit_error");
}

// ---- limits ----

#[tokio::test]
async fn oversized_bodies_are_413() {
    let s = setup_with(
        ServerOptions {
            max_request_bytes: 1024,
            ..ServerOptions::default()
        },
        ServeOptions::default(),
    )
    .await;
    let mut body = chat_body("gpt");
    body["messages"][0]["content"] = "x".repeat(2000).into();
    let (status, _, resp) = json(s.gw.post("/v1/chat/completions", &body).await).await;
    assert_eq!(status, 413);
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("1024 bytes")
    );
    let (status, _, resp) = json(s.gw.post("/v1/messages", &body).await).await;
    assert_eq!(status, 413);
    assert_eq!(resp["error"]["type"], "request_too_large");
    assert!(s.good.seen().is_empty());
}

#[tokio::test]
async fn concurrency_limit_refuses_excess_requests() {
    let s = setup_with(
        ServerOptions {
            max_concurrency: 1,
            ..ServerOptions::default()
        },
        ServeOptions::default(),
    )
    .await;
    let mut stalled = chat_body("stall");
    stalled["stream"] = true.into();
    let mut held = s.gw.post("/v1/chat/completions", &stalled).await;
    assert_eq!(held.status(), 200);
    assert!(held.chunk().await.unwrap().is_some());

    let (status, headers, _) =
        json(s.gw.post("/v1/chat/completions", &chat_body("gpt")).await).await;
    assert_eq!(status, 503);
    assert_eq!(headers["retry-after"], "1");

    drop(held);
    // The permit is released once the server notices the client has gone.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let status =
            s.gw.post("/v1/chat/completions", &chat_body("gpt"))
                .await
                .status();
        if status == 200 {
            break;
        }
        assert!(Instant::now() < deadline, "the permit was never released");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn slow_headers_are_cut_off() {
    let s = setup_with(
        ServerOptions::default(),
        ServeOptions {
            header_read_timeout: Duration::from_millis(500),
            ..ServeOptions::default()
        },
    )
    .await;
    let mut conn = tokio::net::TcpStream::connect(s.gw.addr).await.unwrap();
    conn.write_all(b"GET /health/liveliness HTTP/1.1\r\nHost: x\r\n")
        .await
        .unwrap();
    let started = Instant::now();
    let mut buf = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(5), conn.read_to_end(&mut buf)).await;
    assert!(read.is_ok(), "the server kept a half-sent request open");
    assert!(started.elapsed() < Duration::from_secs(3));
    assert!(!String::from_utf8_lossy(&buf).contains("I'm alive"));
}

// ---- leaks and errors ----

#[tokio::test]
async fn secrets_and_hosts_are_redacted_from_errors() {
    let s = setup().await;
    let (status, _, body) = json(s.gw.post("/v1/chat/completions", &chat_body("leak")).await).await;
    assert_eq!(status, 502);
    let message = body["error"]["message"].as_str().unwrap();
    assert!(!message.contains(UPSTREAM_KEY), "{message}");
    assert!(!message.contains("127.0.0.1"), "{message}");
    assert!(message.contains("[REDACTED]"));
}

#[tokio::test]
async fn model_listings_do_not_leak_configuration() {
    let s = setup().await;
    let (_, _, models) = json(s.gw.get("/v1/models", true).await).await;
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"gpt") && ids.contains(&"smart") && ids.contains(&"flaky"));
    assert!(!ids.contains(&"hidden-gpt"));
    assert_eq!(ids.iter().filter(|i| **i == "flaky").count(), 1);

    let (status, _, one) = json(s.gw.get("/v1/models/gpt", true).await).await;
    assert_eq!((status, one["id"].as_str()), (200, Some("gpt")));
    let (status, _, missing) = json(s.gw.get("/v1/models/nope", true).await).await;
    assert_eq!(
        (status, missing["error"]["code"].as_str()),
        (404, Some("model_not_found"))
    );

    let (_, _, info) = json(s.gw.get("/model/info", true).await).await;
    assert_eq!(
        info["data"][0]["litellm_params"],
        json!({"model": "hosted_vllm/up-gpt"})
    );
    assert_eq!(info["data"][0]["model_info"]["id"], "gpt#0");
    for body in [models.to_string(), info.to_string()] {
        for needle in [
            UPSTREAM_KEY,
            ANTHROPIC_KEY,
            MASTER_KEY,
            "127.0.0.1",
            "api_base",
            "api_key",
            "http://",
        ] {
            assert!(!body.contains(needle), "{needle} leaked: {body}");
        }
    }
}

#[tokio::test]
async fn request_errors_use_the_endpoint_shape() {
    let s = setup().await;
    let (status, _, body) = json(s.gw.post("/v1/chat/completions", &chat_body("nope")).await).await;
    assert_eq!(status, 404);
    assert_eq!(body["error"]["code"], "model_not_found");

    let (status, _, body) = json(s.gw.post("/v1/messages", &chat_body("nope")).await).await;
    assert_eq!(status, 404);
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "not_found_error");

    let resp =
        s.gw.client
            .post(s.gw.url("/v1/chat/completions"))
            .header("authorization", format!("Bearer {MASTER_KEY}"))
            .body("not json")
            .send()
            .await
            .unwrap();
    let (status, _, body) = json(resp).await;
    assert_eq!(status, 400);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("JSON object")
    );

    let (status, _, body) = json(
        s.gw.post("/v1/chat/completions", &json!({"messages": []}))
            .await,
    )
    .await;
    assert_eq!(status, 400);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("`model`")
    );

    let (status, _, _) = json(s.gw.get("/v1/chat/completions", true).await).await;
    assert_eq!(status, 405);
    let (status, _, _) = json(s.gw.get("/nope", false).await).await;
    assert_eq!(status, 404);
}

// ---- health, metrics ----

#[tokio::test]
async fn health_probes_backends_by_deployment_id() {
    let s = setup().await;
    let (status, _, body) = json(s.gw.get("/health", true).await).await;
    assert_eq!(status, 200);
    let unhealthy: Vec<&str> = body["unhealthy_endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["deployment"].as_str().unwrap())
        .collect();
    assert_eq!(unhealthy, ["flaky#0", "primary#0"]);
    assert_eq!(body["unhealthy_endpoints"][0]["error"], "server");
    assert_eq!(body["healthy_count"], 11);
    assert!(!body.to_string().contains("127.0.0.1"));
    let probe = s
        .good
        .seen()
        .into_iter()
        .find(|r| r.method == Method::GET && r.path == "/v1/models");
    assert!(probe.is_some());

    // A second call within the cache window probes nothing.
    let before = s.good.seen().len() + s.down.seen().len();
    s.gw.get("/health", true).await;
    assert_eq!(s.good.seen().len() + s.down.seen().len(), before);

    let (status, _, ready) = json(s.gw.get("/health/readiness", false).await).await;
    assert_eq!((status, ready["status"].as_str()), (200, Some("healthy")));
    let (_, _, alive) = json(s.gw.get("/health/liveliness", false).await).await;
    assert_eq!(alive, "I'm alive!");
}

#[tokio::test]
async fn metrics_count_requests_and_tokens() {
    let s = setup().await;
    s.gw.post("/v1/chat/completions", &chat_body("gpt")).await;
    let mut streaming = chat_body("gpt");
    streaming["stream"] = true.into();
    text(s.gw.post("/v1/chat/completions", &streaming).await).await;
    s.gw.post_with("/v1/chat/completions", &chat_body("gpt"), &[])
        .await;
    let resp = s.gw.get("/metrics", true).await;
    assert!(
        resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain; version=0.0.4")
    );
    let m = text(resp).await;
    for line in [
        "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"200\"} 2",
        "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"401\"} 1",
        "ferrolite_tokens_total{model_group=\"gpt\",kind=\"prompt\"} 8",
        "ferrolite_tokens_total{model_group=\"gpt\",kind=\"completion\"} 3",
        "ferrolite_auth_failures_total 1",
    ] {
        assert!(m.contains(line), "missing {line}\n{m}");
    }
}

// ---- TLS and shutdown ----

#[tokio::test]
async fn tls_round_trip_and_key_permissions() {
    let dir = TempDir::new();
    let key = dir.path().join("key.pem");
    std::fs::copy(fixture("test-key.pem"), &key).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
    let Err(err) = tls::acceptor(&fixture("test-cert.pem"), &key) else {
        panic!("a group-readable key was accepted");
    };
    assert!(err.to_string().contains("chmod 600"), "{err}");

    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let acceptor = tls::acceptor(&fixture("test-cert.pem"), &key).unwrap();
    let good = Mock::start(good).await;
    let down = Mock::start(down).await;
    let gw = Gateway::start_with_tls(
        load(&config_yaml(good.port, down.port), false),
        ServerOptions::default(),
        ServeOptions::default(),
        Some(acceptor),
    )
    .await;

    use tokio_rustls::rustls::pki_types::pem::PemObject;
    use tokio_rustls::rustls::pki_types::{CertificateDer, ServerName};
    use tokio_rustls::rustls::{ClientConfig, RootCertStore, crypto::aws_lc_rs};
    let mut roots = RootCertStore::empty();
    for cert in CertificateDer::pem_file_iter(fixture("test-cert.pem")).unwrap() {
        roots.add(cert.unwrap()).unwrap();
    }
    let config = ClientConfig::builder_with_provider(Arc::new(aws_lc_rs::default_provider()))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let tcp = tokio::net::TcpStream::connect(gw.addr).await.unwrap();
    let mut stream = connector
        .connect(ServerName::try_from("localhost").unwrap(), tcp)
        .await
        .unwrap();
    stream
        .write_all(
            b"GET /health/liveliness HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        )
        .await
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap_or_default();
    let response = String::from_utf8_lossy(&response);
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("I'm alive!"));

    // Plain HTTP to the TLS port gets nothing useful.
    let plain = reqwest::get(format!("http://{}/health/liveliness", gw.addr)).await;
    assert!(plain.is_err() || plain.unwrap().status() != 200);
}

#[tokio::test]
async fn shutdown_lets_open_streams_finish() {
    let mut s = setup().await;
    let mut body = chat_body("slow");
    body["stream"] = true.into();
    let mut resp = s.gw.post("/v1/chat/completions", &body).await;
    let mut got = resp.chunk().await.unwrap().unwrap().to_vec();
    s.gw.shutdown();
    while let Some(chunk) = resp.chunk().await.unwrap() {
        got.extend_from_slice(&chunk);
    }
    assert_eq!(String::from_utf8(got).unwrap(), OPENAI_SSE.concat());
    let task = s.gw.task.take().unwrap();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    // No new connections after shutdown.
    assert!(tokio::net::TcpStream::connect(s.gw.addr).await.is_err());
}

// ---- review fixes ----

/// Polls `/metrics` until it contains `line`.
async fn wait_for_metric(gw: &Gateway, line: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let m = text(gw.get("/metrics", true).await).await;
        if m.contains(line) {
            return m;
        }
        assert!(Instant::now() < deadline, "never saw {line}\n{m}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn failed_streams_are_counted_as_failures() {
    let s = setup().await;
    let mut body = chat_body("crash");
    body["stream"] = true.into();
    let resp = s.gw.post("/v1/chat/completions", &body).await;
    assert_eq!(resp.status(), 200);
    assert!(text(resp).await.contains("model crashed"));
    let m = wait_for_metric(
        &s.gw,
        "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"502\"} 1",
    )
    .await;
    assert!(
        m.contains(
            "ferrolite_upstream_errors_total{model_group=\"crash\",class=\"stream_failed\"} 1"
        ),
        "{m}"
    );
    assert!(!m.contains("code=\"200\""), "{m}");
}

#[tokio::test]
async fn requests_the_client_abandons_are_still_counted() {
    let s = setup().await;
    let impatient = reqwest::Client::builder()
        .timeout(Duration::from_millis(300))
        .build()
        .unwrap();
    let sent = impatient
        .post(s.gw.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .body(chat_body("slowjson").to_string())
        .send()
        .await;
    assert!(sent.is_err(), "the request should have timed out");
    wait_for_metric(
        &s.gw,
        "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"499\"} 1",
    )
    .await;
}

#[tokio::test]
async fn clients_cannot_attach_mcp_servers() {
    let s = setup().await;
    let body = json!({"model": "claude", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"type": "url", "url": "https://attacker.example/mcp", "name": "x"}]});
    let (status, _, resp) = json(s.gw.post("/v1/messages", &body).await).await;
    assert_eq!(status, 400);
    assert_eq!(resp["error"]["type"], "invalid_request_error");
    assert!(
        resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mcp_servers")
    );
    assert!(s.good.calls_for("up-claude").is_empty());
}
