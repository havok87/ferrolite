# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: **Security → Report
a vulnerability** on this repository. Don't open a public issue. Include a
config snippet (with secrets removed) and the request that triggers the
problem, if you can.

## Threat model

ferrolite sits between LLM clients and upstream providers and holds the
upstream credentials. It's designed to run on a developer machine or a
small server, bound to loopback by default.

**What it protects:**
- upstream API keys and any other configured secrets
- the master key
- prompts and completions passing through
- the internal network layout (upstream hosts and ports)

**Who it defends against:**

| Adversary | Examples of what they try |
|---|---|
| Unauthenticated network clients | Reach upstreams without the master key; exhaust connections or memory |
| Authenticated clients | Redirect requests to another host (SSRF); inject credentials or headers; read keys or `api_base` from responses or errors; create unbounded metric series |
| A malicious or compromised upstream | Huge or never-ending responses; redirects that would carry the key elsewhere; error messages that echo secrets |
| Other local users | Read the config, secret files, the TLS key or logs |
| The supply chain | A compromised dependency or CI action |

**Out of scope:**
- **A compromised host or user account.** Anyone who can read your process
  memory or your keychain session already has the keys.
- **The operator's own configuration.** For example, pointing `api_base` at
  a hostile server.
- **Content-level attacks on the models themselves.** For example, prompt
  injection.
- **Per-client rate limiting or quotas.** There are no per-client
  identities. Put a reverse proxy in front if you need this.

## Mitigations

| Threat | Mitigation | Where |
|---|---|---|
| Unauthenticated access | Master key required (`sk-`, at least 32 chars), compared in constant time (`subtle`); `--allow-no-auth` refused off loopback; only liveliness and readiness are open | `server/auth.rs`, `config.rs` |
| Network exposure | Binds `127.0.0.1:4000` by default; warns when bound elsewhere, and again when that is plain HTTP; optional rustls TLS whose key file must be owner-only | `main.rs`, `server/tls.rs` |
| SSRF and credential injection | Client `api_base`, `base_url`, `api_key`, `api_version`, `custom_llm_provider`, `extra_headers`, `mock_response` and `litellm_*` are stripped before routing; upstream URLs and headers come only from config; client auth headers are never forwarded (only `anthropic-version` and `anthropic-beta` on Anthropic passthrough); image URLs are passed on, never fetched; requests that attach MCP servers or tools, or reference `file_id`s or containers in the upstream account, are refused | `translate/mod.rs`, `providers/` |
| Credential leakage through redirects or proxies | Upstream redirects are not followed; `HTTP(S)_PROXY` is ignored | `providers/mod.rs` |
| Secrets in logs, errors and responses | `Secret` type whose `Debug`/`Display` print `[REDACTED]`; a redactor scrubs every configured secret and upstream host from error messages; config errors name paths, never values; YAML errors quote no source text; logs never contain prompts, completions or headers; `/v1/models` and `/model/info` never include keys, `api_base` or headers | `secrets.rs`, `config.rs`, `server/` |
| Secrets on disk | `file/` secrets and the TLS key refuse group/other permissions (checked on the open file handle); warning when a group/world-readable config contains literal secrets; keychain refs run `/usr/bin/security` by absolute path with no shell and a 15 s timeout; the launchd example sets umask 077 | `secrets.rs`, `examples/` |
| Excluded features smuggled in through config | Keys for databases, UI, virtual keys, callbacks, guardrails, MCP, pass-through, caching and mock responses fail startup | `config.rs` |
| Resource exhaustion by clients | Request body limit (32 MiB default) and 60 s body read timeout; 30 s header read timeout; TLS handshake timeout; open-connection cap; global concurrency limit with immediate 503; `/health` results cached for 10 s | `server/` |
| Resource exhaustion by upstreams | 10 s connect timeout; per-attempt timeouts; non-streaming response cap (64 MiB default); streams cut off when idle for the attempt timeout; 4 MiB SSE event cap; error bodies read to 64 KiB at most; retries bounded with capped backoff; fallbacks don't chain (at most 5) | `providers/`, `router/` |
| Hostile YAML | Parser budgets (depth, nodes, anchors, aliases), duplicate keys rejected, custom tags and `!include` refused, 4 MiB file cap | `config.rs` |
| Metrics cardinality | Every series is created at startup from fixed label sets and configured groups; client input can't add labels | `server/metrics.rs` |
| Memory-safety bugs | `#![forbid(unsafe_code)]` in every crate in this repo | |
| Supply chain | Small dependency tree; `Cargo.lock` committed; `cargo deny` (advisories, license allowlist, crates.io only, no wildcards) and `cargo audit` in CI; rustls only (no OpenSSL or native-tls); CI uses first-party actions pinned by commit SHA with a read-only token and no `pull_request_target`; Dependabot waits 7 days before proposing new releases | `deny.toml`, `.github/` |

## Operator checklist

- **Keep the gateway on loopback** where you can. To expose it, use
  `--tls-cert`/`--tls-key` or a TLS-terminating reverse proxy.
- **Use `keychain/` or `file/` references** instead of literal keys, and
  `chmod 600` the config.
- **Generate a fresh master key** (`printf 'sk-%s\n' "$(openssl rand -hex 24)"`)
  and rotate it if it leaks. The master key is the only client credential,
  so give it only to clients you trust with every configured upstream.
- **Prefer `https://` upstreams.** ferrolite warns about plain `http://` to
  hosts that aren't private.
- **Run `ferrolite --check`** after every config change; it reports every
  problem at once.
- **Watch `ferrolite_auth_failures_total`** and the `request` log lines. The
  logs hold no prompts, but they do hold model names and token counts.
