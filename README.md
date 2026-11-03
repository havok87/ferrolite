# ferrolite

A small, security-focused LLM gateway that reads LiteLLM's `config.yaml`.
It reimplements the LiteLLM proxy's routing core in Rust: OpenAI- and
Anthropic-compatible endpoints, load balancing, retries, cooldowns and
fallbacks across OpenAI-compatible backends (OpenAI, vLLM, Ollama, LM Studio)
and Anthropic. It runs on macOS (Apple Silicon and Intel) and Linux.

The point is a small attack surface. The LiteLLM proxy also has an admin UI,
a database, virtual keys, callbacks, guardrails, an MCP gateway and
pass-through endpoints. Those features are where most of its security
incidents have come from, and ferrolite leaves them out. If your config
asks for one of them, ferrolite refuses to start and names the key. See
[COMPATIBILITY.md](COMPATIBILITY.md) for the feature matrix and
[SECURITY.md](SECURITY.md) for the threat model.

## Quickstart (macOS)

You need [rustup](https://rustup.rs) and a C compiler
(`xcode-select --install`).

```sh
git clone https://github.com/havok87/ferrolite.git
cd ferrolite
rustup toolchain install               # the version pinned in rust-toolchain.toml
cargo install --locked --path .        # installs ~/.cargo/bin/ferrolite
```

Create a master key, the key clients use to reach the gateway. It must
start with `sk-` and be at least 32 characters:

```sh
printf 'sk-%s\n' "$(openssl rand -hex 24)"
```

Store it and your upstream keys in the macOS keychain (each command prompts
for the value), so they never sit in a file:

```sh
security add-generic-password -s ferrolite -a master -w
security add-generic-password -s ferrolite -a openai -w
security add-generic-password -s ferrolite -a anthropic -w
```

Write a config. The format is LiteLLM's:

```sh
mkdir -p ~/.config/ferrolite
cat > ~/.config/ferrolite/config.yaml <<'EOF'
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: keychain/ferrolite/openai
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: keychain/ferrolite/anthropic
  - model_name: local
    litellm_params:
      model: ollama_chat/qwen3:8b       # Ollama on 127.0.0.1:11434

router_settings:
  num_retries: 2
  fallbacks:
    - gpt-4o: [claude-sonnet]

general_settings:
  master_key: keychain/ferrolite/master
EOF
chmod 600 ~/.config/ferrolite/config.yaml
```

Check it, then run it:

```sh
ferrolite --config ~/.config/ferrolite/config.yaml --check
ferrolite --config ~/.config/ferrolite/config.yaml
```

It listens on `127.0.0.1:4000` and logs JSON lines to stderr. Try it:

```sh
KEY=$(security find-generic-password -s ferrolite -a master -w)
curl -s localhost:4000/health/liveliness
curl -s localhost:4000/v1/models -H "Authorization: Bearer $KEY"
curl -s localhost:4000/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'content-type: application/json' \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

[examples/config.example.yaml](examples/config.example.yaml) shows more:
wildcards, aliases, vLLM and weights.

### Pointing clients at it

- **OpenAI SDKs and tools:** set `OPENAI_BASE_URL=http://127.0.0.1:4000/v1`
  and use the master key as `OPENAI_API_KEY`.
- **Anthropic-format clients:** set `ANTHROPIC_BASE_URL=http://127.0.0.1:4000`
  and send the master key as `x-api-key` or as a bearer token. For Claude Code
  that is `ANTHROPIC_AUTH_TOKEN`. `/v1/messages` works with Anthropic
  deployments (passthrough) and with OpenAI-compatible ones (translated), so
  Anthropic-format clients can use local models. The client must ask for a
  `model_name` (or alias) that the config defines. ferrolite does not
  implement `/v1/messages/count_tokens`.

## Running under launchd

[examples/dev.ferrolite.plist](examples/dev.ferrolite.plist) runs ferrolite as
a user agent. Because it runs in your login session, `keychain/` references
work. Replace `YOU` in it with your user name, then:

```sh
cp examples/dev.ferrolite.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/dev.ferrolite.plist
launchctl print gui/$(id -u)/dev.ferrolite | grep state     # running
launchctl kickstart -k gui/$(id -u)/dev.ferrolite          # restart after a config change
launchctl bootout gui/$(id -u)/dev.ferrolite               # stop and unload
```

Logs go to `~/Library/Logs/ferrolite.log`, which is readable only by you.
launchd restarts ferrolite if it exits with an error, at most once every
10 seconds.

## Command line

| Flag | Default | |
|---|---|---|
| `--config <path>` | `config.yaml` | LiteLLM-format config |
| `--host <ip>` | `127.0.0.1` | Bind address. Anything else exposes the gateway, and ferrolite warns |
| `--port <n>` | `4000` | |
| `--allow-no-auth` | off | Run without a master key. Refused unless `--host` is loopback |
| `--check` | | Load and validate the config, then exit |
| `--log-level` | `info` | `off`, `error`, `warn`, `info`, `debug`, `trace` |
| `--max-request-mb` | `max_request_size_mb`, else 32 | Largest request body |
| `--max-response-mb` | `max_response_size_mb`, else 64 | Largest non-streaming upstream response |
| `--max-concurrency` | `256` | Proxied requests at once, streams included. More get 503 |
| `--max-connections` | `1024` | Open client connections. More are closed on accept |
| `--connect-timeout` | `10` | Seconds to connect to an upstream |
| `--tls-cert`, `--tls-key` | | Serve HTTPS. The key file must be owner-only (`chmod 600`) |

## Configuration reference

ferrolite reads LiteLLM's `config.yaml`. Keys for features it excludes by
design fail startup, and so do keys it can't honor (for example
`litellm_credential_name`). Harmless keys it doesn't implement produce a
warning. Problems are reported together, by path (for example
`general_settings.database_url`), and never with their values.

### Secret references

Any of these work where a secret is expected:

| Form | Meaning |
|---|---|
| `os.environ/NAME` | Environment variable `NAME` |
| `file/<path>` | Contents of a file (one trailing newline removed). Refused if group or other has any access. Relative paths are resolved from the config file's directory |
| `keychain/<service>/<account>` | macOS keychain generic password, read with `/usr/bin/security find-generic-password -w` (no shell) |
| anything else | A literal value. ferrolite warns if the config file is group- or world-readable and contains literals |

References are accepted in `general_settings.master_key` and in each
deployment's `api_key`, `api_base` and `extra_headers` values.

### `model_list[]`

| Key | |
|---|---|
| `model_name` | The name clients request. A trailing `*` makes it a wildcard (`openai/*` matches `openai/gpt-4.1`) |
| `litellm_params.model` | `provider/upstream-model`. A provider prefix is required |
| `litellm_params.api_base` | Upstream base URL. It must be http(s) with no credentials, query string or fragment. ferrolite warns for plain `http://` to a public host |
| `litellm_params.api_key` | Upstream key (a secret reference or a literal) |
| `litellm_params.custom_llm_provider` | Alternative to the prefix in `model` |
| `litellm_params.rpm`, `tpm` | Per-minute limits for this deployment, also used as weights for simple-shuffle. `rpm` is enforced exactly. `tpm` counts tokens as responses finish, so requests already in flight can carry a minute past it |
| `litellm_params.weight` | Weight for simple-shuffle |
| `litellm_params.order` | Priority tier. Lower goes first, and deployments without an `order` come last |
| `litellm_params.timeout`, `stream_timeout` | Seconds per attempt. `stream_timeout` is used for streaming requests |
| `litellm_params.extra_headers` | Headers sent upstream. Values under credential-like names are treated as secrets |
| `litellm_params.drop_params`, `additional_drop_params` | Per-deployment versions of the setting below, and a list of params always removed |
| other `litellm_params` keys | Request defaults (for example `temperature`, `max_tokens`). The client's own value wins |
| `model_info` | Free-form metadata shown by `/model/info`. `max_output_tokens` / `max_tokens` set the Anthropic `max_tokens` default |

Providers:

| Prefix | API | Default `api_base` |
|---|---|---|
| `openai/` | OpenAI | `https://api.openai.com/v1` (needs `api_key`) |
| `anthropic/` | Anthropic Messages | `https://api.anthropic.com` (needs `api_key`) |
| `ollama/`, `ollama_chat/` | Ollama's OpenAI-compatible `/v1` API. `/v1` is appended to `api_base` if missing | `http://127.0.0.1:11434/v1` |
| `lm_studio/` | OpenAI-compatible | `http://127.0.0.1:1234/v1` |
| `hosted_vllm/`, `vllm/`, `custom_openai/` | OpenAI-compatible | none; `api_base` is required |

### `router_settings`

| Key | Default | |
|---|---|---|
| `routing_strategy` | `simple-shuffle` | Also `least-busy` and `latency-based-routing`. Anything else warns and uses simple-shuffle |
| `model_group_alias` | | `alias: group`, or `alias: {model: group, hidden: true}` to keep it out of `/v1/models` |
| `num_retries` | `0` | Extra attempts on 429, 5xx, timeouts and connection errors, with exponential backoff (honoring `Retry-After`, capped at 30 s) |
| `timeout` | `600` | Seconds per attempt when the deployment sets none |
| `allowed_fails` | `3` | Failures in a minute before a deployment cools down. Groups with one deployment never cool down |
| `cooldown_time` | `5` | Seconds a deployment sits out |
| `fallbacks` | | `[{group: [other, ...]}]`, tried after retries are exhausted or when no deployment is available |
| `context_window_fallbacks` | | Tried first for context-length errors |
| `content_policy_fallbacks` | | Tried first for content-policy errors |

Fallbacks don't chain (a fallback group's own fallbacks aren't used), at
most 5 are tried, and a bad request (400) never falls back. `num_retries`, `fallbacks`
and the specific fallback lists are also accepted under `litellm_settings`
(with `request_timeout` for `timeout`). If both sections set a key,
`router_settings` wins and ferrolite warns.

### `litellm_settings` and `general_settings`

| Key | |
|---|---|
| `litellm_settings.drop_params` | When a request is translated to another provider's format, unsupported params are dropped instead of returning a 400 that names them |
| `general_settings.master_key` | Required unless `--allow-no-auth`. Must start with `sk-`, at least 32 characters |
| `general_settings.max_request_size_mb` | Request body limit (default 32) |
| `general_settings.max_response_size_mb` | Non-streaming upstream response limit (default 64) |

## Endpoints

| Endpoint | Needs key | Also without `/v1` | Notes |
|---|---|---|---|
| `POST /v1/chat/completions` | yes | yes | Any backend |
| `POST /v1/completions`, `/v1/embeddings`, `/v1/responses` | yes | yes | OpenAI-compatible backends only |
| `POST /v1/messages` | yes | no | Any backend |
| `GET /v1/models`, `/v1/models/{id}` | yes | yes | |
| `GET /model/info`, `/v1/model/info` | yes | | |
| `GET /health` | yes | | Probes each deployment's `/models`. The result is cached for 10 s |
| `GET /health/liveliness`, `/health/readiness` | no | | Readiness is 503 when every deployment is cooling down |
| `GET /metrics` | yes | | Prometheus text |

The key can be sent as `Authorization: Bearer`, `x-api-key` or `api-key`.
Errors use OpenAI's shape, or Anthropic's on `/v1/messages`. `/v1/models`
and `/model/info` never include keys, `api_base` or headers. Every response
carries an `x-request-id`. When no deployment of a group is available, the
answer is 429 with a `Retry-After` set to when the first one frees up (its
cooldown ending, or the next minute for an rpm/tpm limit).

## Logs and metrics

ferrolite writes one JSON line per request, never with prompts,
completions or headers:

```json
{"timestamp":"…","level":"INFO","fields":{"message":"request","request_id":"2a7f059c-…","route":"/v1/chat/completions","model_group":"gpt-4o","deployment":"gpt-4o#0","status":200,"latency_ms":412,"prompt_tokens":9,"completion_tokens":12,"attempts":1},"target":"ferrolite::server::routes"}
```

Deployments are identified as `model_name#index`, never by URL. A request
whose client disconnects before the answer is logged with status 499. A
stream that carried an upstream error or broke off is logged as 502 and
counted as a `stream_failed` upstream error (its HTTP status was already
200). `/metrics` exports these series:
- `ferrolite_requests_total{route,code}`
- `ferrolite_request_duration_seconds{route}`
- `ferrolite_tokens_total{model_group,kind}`
- `ferrolite_upstream_errors_total{model_group,class}`
- `ferrolite_fallbacks_total{model_group}`
- `ferrolite_auth_failures_total`
- `ferrolite_in_flight_requests`
- `ferrolite_deployment_cooling_down{model_group,deployment}`

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo deny check
cargo audit
```

CI runs the same five checks on macOS and Linux. Unit tests sit next to the
code. `tests/server.rs` runs a real server in front of mock upstreams on
ephemeral ports.

## License

MIT or Apache-2.0, at your option.
