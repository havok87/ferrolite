# LiteLLM compatibility

ferrolite reads LiteLLM's `config.yaml` and serves the same core endpoints,
so many LiteLLM proxy setups move over unchanged. This page lists what's
supported, what's left out on purpose, and what isn't supported yet.

What ferrolite does when a config key falls in each group:

- **Supported:** the key behaves as in LiteLLM, apart from the differences
  listed at the end.
- **Excluded by design:** **startup fails**, naming the key. ferrolite
  never quietly runs without a feature you configured.
- **Not supported:** startup fails when ignoring the key would change
  behavior (for example `include`, `credential_list`, Azure keys). Otherwise
  it logs a warning (for example `set_verbose`, cost fields).
- **Unknown keys:** a warning.

## Supported

| Area | Feature |
|---|---|
| Config | `model_list` (`model_name`, `litellm_params`, `model_info`), `router_settings`, `litellm_settings`, `general_settings.master_key` |
| Secrets | `os.environ/VAR`, plus `file/<path>` and `keychain/<service>/<account>` (ferrolite additions) |
| Endpoints | `/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/models`, `/v1/models/{id}`, `/model/info`, `/health`, `/health/liveliness`, `/health/readiness`, `/metrics`; most also without `/v1` |
| Providers | `openai/`, `hosted_vllm/`, `vllm/`, `lm_studio/`, `custom_openai/`, `ollama/` and `ollama_chat/` (through Ollama's OpenAI-compatible `/v1`), `anthropic/` |
| Translation | OpenAI chat to Anthropic Messages and back, including streaming, tools, images, `reasoning_effort` and cache-token usage. Anthropic `/v1/messages` to OpenAI-compatible backends and back |
| Routing | Model groups, `model_group_alias` (including `hidden`), trailing-`*` wildcards, `simple-shuffle` (by `weight`, else `rpm`/`tpm`), `least-busy`, `latency-based-routing`, `order`, per-deployment `rpm`/`tpm` |
| Reliability | `num_retries` with backoff, `allowed_fails` and `cooldown_time`, `fallbacks`, `context_window_fallbacks`, `content_policy_fallbacks`, `timeout` / `request_timeout`, `stream_timeout` |
| Request handling | `drop_params` (global and per deployment), `additional_drop_params`, extra `litellm_params` as request defaults, `extra_headers` |
| Limits | `general_settings.max_request_size_mb`, `max_response_size_mb` |
| Auth | Master key via `Authorization: Bearer`, `x-api-key` or `api-key` |

## Excluded by design

These features hold most of the LiteLLM proxy's attack surface. Setting any
of them stops startup.

| Feature | Keys that stop startup |
|---|---|
| Database, spend logs | `general_settings.database_*`, `store_model_in_db`, `store_prompts_in_spend_logs`, `disable_spend_logs`; `router_settings.redis_*` |
| Admin UI | `general_settings.ui_*` |
| Virtual keys, users, teams, budgets, JWT auth | `litellm_settings.max_budget`, `budget_duration`, `max_end_user_budget`, `default_team_settings`, `default_internal_user_params`, `default_key_generate_params`, `upperbound_key_generate_params`, `key_generation_settings`; `general_settings.enable_jwt_auth`, `litellm_jwtauth`; `router_settings.provider_budget_config` |
| Callbacks and logging integrations | `litellm_settings.callbacks`, `*_callback`, `*_callbacks`; top-level `callback_settings` |
| Alerting | `general_settings.alerting`, `alerting_*`, `alert_types`, `alert_to_webhook_url` |
| Custom auth or key-generation code | `general_settings.custom_auth`, `custom_sso`, `custom_key_generate` |
| Guardrails | `guardrails` (top level and in `litellm_settings`) |
| MCP gateway | top-level `mcp_*` (for example `mcp_servers`) |
| Pass-through endpoints | `pass_through_endpoints` (top level and in `general_settings`) |
| Response caching | `litellm_settings.cache`, `cache_params`; `router_settings.cache_*` |
| Mock responses | `litellm_params.mock_response` |
| Client-supplied credentials | `litellm_params.configurable_clientside_auth_params` |

These are also never accepted from clients. Any of these fields in a
request body are removed before routing: `api_base`, `base_url`, `api_key`,
`api_version`, `custom_llm_provider`, `extra_headers`, `mock_response`, and
any `litellm_*` key. Client auth headers are never forwarded upstream.
Adding or deleting models over the API isn't available, and remote images
or files are never fetched on a client's behalf.

## Not supported yet

| Feature | What happens now |
|---|---|
| Other providers (Azure, Bedrock, Vertex, Gemini, watsonx and so on) | `model:` with their prefix fails startup; `aws_*`, `vertex_*`, `azure_*`, `watsonx_*` params fail startup |
| `include:` of other config files | Fails startup |
| `credential_list`, `litellm_credential_name` | Fails startup (set `api_key` on each deployment) |
| `environment_variables` | Warning. Export the variables before starting |
| Usage- and cost-based routing strategies | Warning, then simple-shuffle |
| `enable_pre_call_checks`, `retry_policy`, `model_group_retry_policy`, `allowed_fails_policy`, `routing_strategy_args`, `retry_after`, `client_ttl` | Warning, then ignored |
| Cost tracking fields (`input_cost_per_token` and so on), `tags`, `api_version`, `organization`, `max_retries` in `litellm_params` | Warning, then ignored |
| `/v1/messages/count_tokens`, batches, files, fine-tuning, images, audio, rerank, assistants | 404 |
| `response_format`, `logprobs`, `n > 1`, `seed`, penalties when translating chat to Anthropic | 400 naming the parameters, or dropped with `drop_params` |
| HTTP/2 to clients or upstreams | HTTP/1.1 only |
| Per-client rate limits | Use a reverse proxy |

## Deliberate differences from LiteLLM

- **Provider prefixes are required.** `model: gpt-4o` fails with a hint to
  write `openai/gpt-4o`. Providers are never guessed from a model name.
- **Secret references work only in fields that hold secrets or
  addresses:** `master_key`, `api_key`, `api_base` and `extra_headers`. A
  reference anywhere else in `litellm_params` is an error, so a secret
  can't end up in a request default.
- **`file/` refuses any group or other permission bits**, like ssh does
  with private keys.
- **The master key must start with `sk-` and be at least 32 characters.**
  It's required unless `--allow-no-auth` is set, and that flag only works
  on loopback.
- **Upstream 401/403 becomes 502.** The configured credentials are at
  fault, not the client's. An upstream 529 becomes 503, and an upstream 408
  becomes 504.
- **An unknown model is 404 `model_not_found`**, as in OpenAI's API.
- **Groups with a single deployment never cool down**, so fallbacks behave
  predictably.
- **Fallbacks don't chain.** A fallback group's own fallbacks aren't used,
  so one request's blast radius stays bounded.
- **`/health` probes `GET /models`** on each deployment rather than making
  completion calls, and caches the result for 10 seconds.
- **Upstream redirects are never followed.** A redirect is reported as a
  misconfigured deployment (502).
- **Request defaults never override the client.** A value the client
  sends wins over the deployment's `litellm_params`.
