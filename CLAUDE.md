# ferrolite

A security-focused Rust reimplementation of the **LiteLLM proxy's routing core**.
It is a drop-in, OpenAI- and Anthropic-compatible gateway that reads LiteLLM's `config.yaml`
format and runs on macOS (Apple Silicon) first.

The point of this project is a **small attack surface**. In 2026 LiteLLM had two
supply-chain compromises and several exploited CVEs (SQL injection in key validation,
an RCE chain, MCP auth bypass, SSRF). Almost all of them lived in features this project
deliberately excludes. When in doubt, leave a feature out.

## Scope (v1)

### In scope
- LiteLLM-compatible `config.yaml`: `model_list` (`model_name`, `litellm_params`, `model_info`),
  `router_settings`, `litellm_settings`, `general_settings.master_key`.
- Secret references: `os.environ/VAR`, `file/<path>` (refuse if group/world readable),
  and on macOS `keychain/<service>/<account>`, which runs `/usr/bin/security find-generic-password -w`
  by absolute path with no shell.
- Endpoints, each also without the `/v1` prefix where LiteLLM accepts that:
  - `POST /v1/chat/completions` (streaming and non-streaming)
  - `POST /v1/completions` and `POST /v1/embeddings` (OpenAI-compatible backends only)
  - `POST /v1/responses` (passthrough to OpenAI-compatible backends only)
  - `POST /v1/messages` (Anthropic Messages format in)
  - `GET /v1/models`, `GET /v1/models/{id}`, `GET /model/info` (never includes keys or `api_base`)
  - `GET /health/liveliness`, `GET /health/readiness` (unauthenticated);
    `GET /health` (authenticated, probes backends)
  - `GET /metrics` (authenticated, Prometheus text, hand-rolled with atomics)
- Providers:
  - OpenAI-compatible: `openai/`, `hosted_vllm/`, `vllm/`, `lm_studio/`, `custom_openai/`;
    `ollama/` and `ollama_chat/` map to Ollama's OpenAI-compatible `/v1` endpoint (document this).
    This covers TensorFold, vLLM, Ollama, LM Studio and OpenAI itself.
  - `anthropic/`: native Messages API.
- Translation in both directions:
  - OpenAI chat to Anthropic: system extraction, text and image parts (data URLs to base64
    source; http(s) URLs passed to Anthropic as URL source, **never fetched by the proxy**),
    tools and tool_choice, assistant tool_calls to `tool_use`, tool messages to `tool_result`,
    stop to stop_sequences, required max_tokens default, reasoning_effort to thinking budget,
    stop_reason and usage mapping (including cache tokens), streaming SSE translation with
    `stream_options.include_usage`.
  - Anthropic `/v1/messages` to an OpenAI-compatible backend: request, response and streaming
    translation, so Anthropic-format clients can use local models.
  - `/v1/messages` to an Anthropic backend: passthrough with model rewrite and server-side auth.
- Router, matching LiteLLM semantics where practical:
  - model groups, `model_group_alias`, trailing-`*` wildcard model names
  - strategies: `simple-shuffle` (weighted by `weight`, else rpm/tpm), `least-busy`,
    `latency-based-routing` (EWMA); anything else falls back to simple-shuffle with a startup warning
  - deployment `order` priority, per-deployment `rpm`/`tpm` limits
  - `num_retries` with exponential backoff on retryable errors (429, 5xx, timeouts, connection errors)
  - `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`
  - cooldowns: `allowed_fails` within a minute puts a deployment in cooldown for `cooldown_time`
  - `timeout` / `request_timeout`, `drop_params` (when false, unsupported params for a translated
    provider return 400 like LiteLLM)
  - extra keys in `litellm_params` (e.g. `temperature`, `max_tokens`) are applied as request
    defaults; `extra_headers` are sent upstream

### Out of scope (excluded by design; do not add)
Admin UI, database of any kind, virtual keys, users/teams/orgs, spend tracking DB,
MCP gateway, guardrails that run user-supplied code, callbacks or plugins loaded at runtime,
dynamic model add/delete over the API, pass-through endpoints to arbitrary URLs,
client-supplied credentials or base URLs, fetching remote images or files on a client's behalf.
If `config.yaml` sets any of these (e.g. `database_url`, `store_model_in_db`, `mcp_servers`,
`guardrails`, `callbacks`, `ui_*`), **fail at startup** with a clear message naming the key.
Unknown but harmless keys produce a warning, not an error.

## Security requirements (non-negotiable)

- `#![forbid(unsafe_code)]` in every crate in this repo.
- Bind `127.0.0.1:4000` by default; exposing it on other interfaces needs an explicit `--host`.
- `master_key` is required (must start with `sk-`, at least 32 characters). The only exception is
  `--allow-no-auth`, and that is refused unless the bind address is loopback.
  Compare keys in constant time (`subtle`). Accept `Authorization: Bearer`, `x-api-key`
  and `api-key` headers.
- Strip client-supplied routing and credential fields before routing: `api_base`, `base_url`,
  `api_key`, `api_version`, `custom_llm_provider`, `extra_headers`, `mock_response`, and any key
  starting with `litellm_`. Never forward the client's auth headers upstream. Forward only an
  allowlist of headers (`anthropic-beta`, `anthropic-version` for Anthropic passthrough).
- Upstream base URLs come only from config. Warn at startup for plain `http://` to a
  non-private, non-loopback host.
- Limits: request body size (default 32 MiB, configurable), global concurrency limit,
  connect/read timeouts, max upstream response size for non-streaming calls.
- Secrets live in a `Secret` newtype whose `Debug`/`Display` print `[REDACTED]`. A redactor scrubs
  every configured secret value from error messages before they reach a client or the log.
- Logs: structured JSON via `tracing`. Log request id, route, model group, deployment id
  (the `model_name` plus an index, never `api_base`), status, latency and token counts.
  **Never log prompts, completions or headers.**
- Errors returned to clients use OpenAI's error shape (or Anthropic's on `/v1/messages`),
  with upstream detail redacted and internal topology removed.
- TLS: rustls only (no OpenSSL / native-tls). Optional server TLS via `--tls-cert`/`--tls-key`.
- Warn at startup if `config.yaml` is group/world readable and contains literal secrets.

## Dependencies and supply chain

- Keep the dependency tree small. Preferred: `tokio`, `axum`, `reqwest`
  (`default-features = false`, `rustls`, `stream`, `json`), `serde`, `serde_json`,
  `serde-saphyr` (YAML; **not** `serde_yaml`, which is unmaintained), `tracing`,
  `tracing-subscriber`, `clap`, `thiserror`, `subtle`, `rand`, `futures-util`, `bytes`, `uuid`.
  Write the SSE parser and the Prometheus output by hand.
- Ask before adding any dependency not on this list, and say why.
- Commit `Cargo.lock`. Add `deny.toml` (advisories deny, licenses allowlist, crates.io as the
  only source, ban wildcards). `cargo deny check` and `cargo audit` must pass.
- GitHub Actions: `permissions: contents: read`, actions pinned by full commit SHA,
  no `pull_request_target`. CI runs fmt, clippy (`-D warnings`), tests, deny and audit
  on `macos-latest` and `ubuntu-latest`.

## Layout

```
src/
  main.rs          CLI (clap), startup checks, server bootstrap
  config.rs        config.yaml model, validation, forbidden-key checks
  secrets.rs       Secret type, secret-reference resolution, redactor
  router/          groups, aliases, wildcards, strategies, cooldowns, retries, fallbacks
  providers/       openai_compat.rs, anthropic.rs
  translate/       openai_to_anthropic.rs, anthropic_to_openai.rs, sse.rs
  server/          routes, auth, limits, errors, health, metrics
tests/             integration tests against mock upstreams (axum on ephemeral ports)
examples/          config.example.yaml, launchd plist
```

## Testing

- Unit tests for config parsing (including forbidden keys), secret resolution, the redactor,
  every translation direction including streaming, and the router (strategy choice,
  cooldown, retries, fallbacks, rpm limits).
- Integration tests with mock OpenAI and Anthropic upstreams covering: auth required,
  passthrough, streaming, both translation directions, fallbacks, context-window fallback,
  cooldown, body-size limit, client `api_base` being stripped, secrets redacted in errors,
  `/v1/models` not leaking keys or `api_base`.
- Before saying any phase is done: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, `cargo deny check`, `cargo audit`.

## Working style

- Build in phases, one commit per phase, and stop for review at the end of each:
  1. scaffold, config, secrets  2. router  3. providers and translation
  4. server, auth, hardening  5. docs, CI, launchd example
- Docs to produce: `README.md` (quickstart for macOS, config reference),
  `COMPATIBILITY.md` (LiteLLM feature matrix: supported / excluded by design / later),
  `SECURITY.md` (threat model and the hardening list above).
- Commit messages: imperative mood, explain why.
