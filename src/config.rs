//! LiteLLM-compatible `config.yaml`: parsing, key policy, validation and
//! secret resolution.
//!
//! Loading has three stages:
//! 1. YAML is parsed into a `serde_json::Value` with parser budgets on and
//!    duplicate keys, custom tags and `!include` rejected.
//! 2. Each section is read with [`Section`], which pulls typed fields out by
//!    name and remembers which keys it used. Left-over keys are then checked
//!    against the key policy: excluded features fail startup, harmless
//!    unimplemented ones warn.
//! 3. The result is validated (providers, URLs, cross references) and secret
//!    references are resolved.
//!
//! Error and warning messages name config paths, never values: a value may be
//! a literal key.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value};
use url::{Host, Url};

use crate::secrets::{MIN_REDACTABLE_CHARS, Redactor, Resolver, Secret, SecretRef};

/// Largest accepted `config.yaml`.
pub const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MIN_MASTER_KEY_CHARS: usize = 32;
const DEFAULT_ALLOWED_FAILS: u32 = 3;
const DEFAULT_COOLDOWN: Duration = Duration::from_secs(5);

/// Settings that come from the command line but affect validation.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// The address the server will bind.
    pub bind: IpAddr,
    /// `--allow-no-auth`: run without a master key (loopback only).
    pub allow_no_auth: bool,
}

/// A loaded config plus everything startup needs to report about it.
#[derive(Debug)]
pub struct Loaded {
    pub config: Config,
    pub warnings: Vec<String>,
    /// Scrubs every secret the config resolved.
    pub redactor: Redactor,
}

#[derive(Debug)]
pub struct Config {
    /// `None` only when `--allow-no-auth` is in effect.
    pub master_key: Option<Secret>,
    pub deployments: Vec<Deployment>,
    pub router: RouterSettings,
    /// `litellm_settings.drop_params`.
    pub drop_params: bool,
    /// `general_settings.max_request_size_mb`, in bytes.
    pub max_request_bytes: Option<usize>,
    /// `general_settings.max_response_size_mb`, in bytes.
    pub max_response_bytes: Option<usize>,
}

impl Config {
    /// Distinct `model_name`s, including wildcard ones.
    pub fn model_groups(&self) -> BTreeSet<&str> {
        self.deployments
            .iter()
            .map(|d| d.model_name.as_str())
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAiCompat(OpenAiFlavor),
    Anthropic,
}

/// LiteLLM provider prefixes that map to an OpenAI-compatible API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiFlavor {
    OpenAi,
    HostedVllm,
    Vllm,
    LmStudio,
    CustomOpenAi,
    /// `ollama/` and `ollama_chat/` use Ollama's OpenAI-compatible `/v1` API.
    Ollama,
    OllamaChat,
}

const PROVIDERS: &[(&str, Provider)] = &[
    ("openai", Provider::OpenAiCompat(OpenAiFlavor::OpenAi)),
    (
        "hosted_vllm",
        Provider::OpenAiCompat(OpenAiFlavor::HostedVllm),
    ),
    ("vllm", Provider::OpenAiCompat(OpenAiFlavor::Vllm)),
    ("lm_studio", Provider::OpenAiCompat(OpenAiFlavor::LmStudio)),
    (
        "custom_openai",
        Provider::OpenAiCompat(OpenAiFlavor::CustomOpenAi),
    ),
    ("ollama", Provider::OpenAiCompat(OpenAiFlavor::Ollama)),
    (
        "ollama_chat",
        Provider::OpenAiCompat(OpenAiFlavor::OllamaChat),
    ),
    ("anthropic", Provider::Anthropic),
];

impl Provider {
    fn from_prefix(prefix: &str) -> Option<Self> {
        PROVIDERS
            .iter()
            .find(|(name, _)| *name == prefix)
            .map(|(_, p)| *p)
    }

    /// The LiteLLM prefix, e.g. `hosted_vllm`.
    pub fn prefix(self) -> &'static str {
        PROVIDERS
            .iter()
            .find(|(_, p)| *p == self)
            .map_or("unknown", |(name, _)| name)
    }

    fn default_api_base(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("https://api.anthropic.com"),
            Self::OpenAiCompat(OpenAiFlavor::OpenAi) => Some("https://api.openai.com/v1"),
            Self::OpenAiCompat(OpenAiFlavor::Ollama | OpenAiFlavor::OllamaChat) => {
                Some("http://127.0.0.1:11434/v1")
            }
            Self::OpenAiCompat(OpenAiFlavor::LmStudio) => Some("http://127.0.0.1:1234/v1"),
            Self::OpenAiCompat(
                OpenAiFlavor::HostedVllm | OpenAiFlavor::Vllm | OpenAiFlavor::CustomOpenAi,
            ) => None,
        }
    }

    fn is_ollama(self) -> bool {
        matches!(
            self,
            Self::OpenAiCompat(OpenAiFlavor::Ollama | OpenAiFlavor::OllamaChat)
        )
    }
}

/// A header value from `extra_headers`. Values that came from a secret
/// reference, or sit under a credential-looking header name, are secrets.
#[derive(Debug, Clone)]
pub enum HeaderValue {
    Plain(String),
    Secret(Secret),
}

impl HeaderValue {
    pub fn expose(&self) -> &str {
        match self {
            Self::Plain(s) => s,
            Self::Secret(s) => s.expose(),
        }
    }
}

/// One `model_list` entry.
#[derive(Debug, Clone)]
pub struct Deployment {
    /// `"{model_name}#{index within the group}"`. Safe to log.
    pub id: String,
    pub model_name: String,
    pub provider: Provider,
    /// The model name sent upstream (the part after the provider prefix).
    pub upstream_model: String,
    pub api_base: Url,
    pub api_key: Option<Secret>,
    pub rpm: Option<u64>,
    pub tpm: Option<u64>,
    pub weight: Option<f64>,
    pub order: Option<u32>,
    pub timeout: Option<Duration>,
    pub stream_timeout: Option<Duration>,
    pub extra_headers: Vec<(String, HeaderValue)>,
    /// Per-deployment `drop_params`; overrides `litellm_settings.drop_params`.
    pub drop_params: Option<bool>,
    /// Request parameters always removed before calling this deployment.
    pub additional_drop_params: Vec<String>,
    /// Extra `litellm_params` keys, applied as request defaults.
    pub defaults: Map<String, Value>,
    pub model_info: ModelInfo,
}

#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub mode: Option<String>,
    pub max_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    /// Every other `model_info` key, kept for `/model/info`.
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    #[default]
    SimpleShuffle,
    LeastBusy,
    LatencyBased,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAlias {
    pub target: String,
    pub hidden: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fallback {
    pub from: String,
    pub to: Vec<String>,
}

/// `router_settings`, merged with the router keys LiteLLM also accepts under
/// `litellm_settings` (`router_settings` wins).
#[derive(Debug, Clone)]
pub struct RouterSettings {
    pub routing_strategy: RoutingStrategy,
    pub model_group_alias: BTreeMap<String, GroupAlias>,
    pub num_retries: u32,
    pub timeout: Option<Duration>,
    pub allowed_fails: u32,
    pub cooldown_time: Duration,
    pub fallbacks: Vec<Fallback>,
    pub context_window_fallbacks: Vec<Fallback>,
    pub content_policy_fallbacks: Vec<Fallback>,
}

impl Default for RouterSettings {
    fn default() -> Self {
        Self {
            routing_strategy: RoutingStrategy::default(),
            model_group_alias: BTreeMap::new(),
            num_retries: 0,
            timeout: None,
            allowed_fails: DEFAULT_ALLOWED_FAILS,
            cooldown_time: DEFAULT_COOLDOWN,
            fallbacks: Vec::new(),
            context_window_fallbacks: Vec::new(),
            content_policy_fallbacks: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config file {}: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("config file {} is not a regular file", path.display())]
    NotRegular { path: PathBuf },
    #[error("config file {} is larger than {limit} bytes", path.display())]
    TooLarge { path: PathBuf, limit: u64 },
    #[error("config file {} is not valid UTF-8", path.display())]
    NotUtf8 { path: PathBuf },
    #[error("config file {} is not valid YAML: {message}", path.display())]
    Yaml { path: PathBuf, message: String },
    #[error("config file {} is invalid:{}", path.display(), bullets(problems))]
    Invalid {
        path: PathBuf,
        problems: Vec<String>,
        /// Warnings found before loading stopped, so they can still be logged.
        warnings: Vec<String>,
    },
}

fn bullets(items: &[String]) -> String {
    items.iter().map(|p| format!("\n  - {p}")).collect()
}

/// Reads and loads `path`. Relative `file/` secret paths resolve against the
/// directory holding the config.
pub fn load(
    path: &Path,
    options: &LoadOptions,
    resolver: &Resolver<'_>,
) -> Result<Loaded, ConfigError> {
    use std::os::unix::fs::MetadataExt;

    let io = |source| ConfigError::Io {
        path: path.to_owned(),
        source,
    };
    // stat before open, so a FIFO or device is never opened.
    let meta = std::fs::metadata(path).map_err(io)?;
    if !meta.is_file() {
        return Err(ConfigError::NotRegular {
            path: path.to_owned(),
        });
    }
    let file = File::open(path).map_err(io)?;
    let meta = file.metadata().map_err(io)?;
    let too_large = || ConfigError::TooLarge {
        path: path.to_owned(),
        limit: MAX_CONFIG_BYTES,
    };
    if meta.len() > MAX_CONFIG_BYTES {
        return Err(too_large());
    }
    let mut buf = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(io)?;
    if buf.len() as u64 > MAX_CONFIG_BYTES {
        return Err(too_large());
    }
    let text = String::from_utf8(buf).map_err(|_| ConfigError::NotUtf8 {
        path: path.to_owned(),
    })?;

    let base_dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let (mut loaded, has_literal_secret) = parse(&text, path, base_dir, options, resolver)?;
    if has_literal_secret && meta.mode() & 0o044 != 0 {
        loaded.warnings.push(format!(
            "config file {} is readable by group or other and contains literal secrets; \
             chmod 600 it, or use os.environ/, file/ or keychain/ references",
            path.display()
        ));
    }
    Ok(loaded)
}

/// Loads config text. `source` names the config in messages; relative
/// `file/` secret paths resolve against `base_dir`.
pub fn load_str(
    text: &str,
    source: &Path,
    base_dir: &Path,
    options: &LoadOptions,
    resolver: &Resolver<'_>,
) -> Result<Loaded, ConfigError> {
    parse(text, source, base_dir, options, resolver).map(|(loaded, _)| loaded)
}

fn parse(
    text: &str,
    source: &Path,
    base_dir: &Path,
    options: &LoadOptions,
    resolver: &Resolver<'_>,
) -> Result<(Loaded, bool), ConfigError> {
    if text.len() as u64 > MAX_CONFIG_BYTES {
        return Err(ConfigError::TooLarge {
            path: source.to_owned(),
            limit: MAX_CONFIG_BYTES,
        });
    }
    let root = parse_yaml(text).map_err(|message| ConfigError::Yaml {
        path: source.to_owned(),
        message,
    })?;

    let mut cx = Cx {
        base_dir,
        resolver,
        diag: Diag::default(),
        secrets: Vec::new(),
        has_literal_secret: false,
    };
    let config = build(&root, options, &mut cx);

    let Diag {
        errors,
        mut warnings,
    } = cx.diag;
    let config = match config {
        Some(config) if errors.is_empty() => config,
        _ => {
            return Err(ConfigError::Invalid {
                path: source.to_owned(),
                problems: errors,
                warnings,
            });
        }
    };
    for (path, secret) in &cx.secrets {
        if !Redactor::is_redactable(secret) {
            warnings.push(format!(
                "`{path}` is shorter than {MIN_REDACTABLE_CHARS} characters, \
                 so it cannot be redacted from logs and error messages"
            ));
        }
    }
    let redactor = Redactor::new(cx.secrets.iter().map(|(_, s)| s));
    Ok((
        Loaded {
            config,
            warnings,
            redactor,
        },
        cx.has_literal_secret,
    ))
}

fn parse_yaml(text: &str) -> Result<Value, String> {
    let options = serde_saphyr::options! {
        budget: serde_saphyr::budget! {
            max_reader_input_bytes: Some(MAX_CONFIG_BYTES as usize),
            max_documents: 1,
            max_depth: 64,
            max_events: 1_000_000,
            max_nodes: 250_000,
            max_anchors: 1_000,
            max_aliases: 1_000,
            max_total_scalar_bytes: MAX_CONFIG_BYTES as usize,
        },
        duplicate_keys: serde_saphyr::DuplicateKeyPolicy::Error,
        reject_unsupported_tags: true,
        // Snippets would echo source lines, which may hold literal keys.
        with_snippet: false,
        crop_radius: 0,
    };
    serde_saphyr::from_str_with_options::<Value>(text, options).map_err(|e| yaml_message(&e))
}

/// A YAML error as a location plus a short description that quotes no
/// source text.
fn yaml_message(err: &serde_saphyr::Error) -> String {
    let location = err
        .location()
        .filter(|l| l.line() > 0)
        .map(|l| format!("line {} column {}: ", l.line(), l.column()))
        .unwrap_or_default();
    format!("{location}{}", yaml_kind(err))
}

fn yaml_kind(err: &serde_saphyr::Error) -> &'static str {
    let text = err.to_string().to_ascii_lowercase();
    let kinds = [
        (
            "budget breached: documents",
            "expected exactly one YAML document",
        ),
        (
            "budget breached",
            "document exceeds parser limits (too large, too deep, or too many anchors or aliases)",
        ),
        ("duplicate mapping key", "duplicate mapping key"),
        (
            "unsupported tag",
            "unsupported YAML tag (custom tags and !include are not accepted)",
        ),
        (
            "multiple yaml documents",
            "expected exactly one YAML document",
        ),
        ("unknown anchor", "alias refers to an unknown anchor"),
        ("unclosed", "unclosed bracket or quote"),
    ];
    kinds
        .iter()
        .find(|(needle, _)| text.contains(needle))
        .map_or("syntax error", |(_, kind)| kind)
}

/// Errors and warnings collected while loading.
#[derive(Default)]
struct Diag {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Diag {
    fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

/// Load context: where to resolve secrets and what has been resolved.
struct Cx<'a> {
    base_dir: &'a Path,
    resolver: &'a Resolver<'a>,
    diag: Diag,
    /// Every resolved secret, with its config path.
    secrets: Vec<(String, Secret)>,
    has_literal_secret: bool,
}

impl Cx<'_> {
    /// Parses and resolves a secret string; records it for the redactor.
    fn secret(&mut self, path: &str, raw: &str) -> Option<Secret> {
        let secret = self.resolve(path, raw)?;
        self.secrets.push((path.to_owned(), secret.clone()));
        Some(secret)
    }

    /// Parses and resolves a string that may be a secret reference, without
    /// treating the result as a secret (used for `api_base`).
    fn resolve(&mut self, path: &str, raw: &str) -> Option<Secret> {
        let reference = match SecretRef::parse(raw) {
            Ok(r) => r,
            Err(e) => {
                self.diag.error(format!("`{path}`: {e}"));
                return None;
            }
        };
        match self.resolver.resolve(&reference, self.base_dir) {
            Ok(s) => Some(s),
            Err(e) => {
                self.diag.error(format!("`{path}`: {e}"));
                None
            }
        }
    }
}

fn child(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_owned()
    } else {
        format!("{path}.{key}")
    }
}

/// A YAML mapping being read field by field.
struct Section<'v> {
    path: String,
    map: &'v Map<String, Value>,
    used: BTreeSet<&'v str>,
}

impl<'v> Section<'v> {
    /// `None` (with an error) if `value` is not a mapping. YAML `null`, as
    /// left by an empty section, reads as an empty mapping.
    fn new(path: String, value: Option<&'v Value>, diag: &mut Diag) -> Option<Self> {
        static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
        let map = match value {
            None | Some(Value::Null) => EMPTY.get_or_init(Map::new),
            Some(Value::Object(map)) => map,
            Some(_) => {
                diag.error(format!("`{path}` must be a mapping"));
                return None;
            }
        };
        Some(Self {
            path,
            map,
            used: BTreeSet::new(),
        })
    }

    fn key_path(&self, key: &str) -> String {
        child(&self.path, key)
    }

    /// The value under `key`, marking it used. `null` reads as absent.
    fn get(&mut self, key: &str) -> Option<&'v Value> {
        let (k, v) = self.map.get_key_value(key)?;
        self.used.insert(k.as_str());
        (!v.is_null()).then_some(v)
    }

    fn typed<T>(
        &mut self,
        key: &str,
        expected: &str,
        diag: &mut Diag,
        convert: impl FnOnce(&'v Value) -> Option<T>,
    ) -> Option<T> {
        let value = self.get(key)?;
        let out = convert(value);
        if out.is_none() {
            diag.error(format!("`{}` must be {expected}", self.key_path(key)));
        }
        out
    }

    fn string(&mut self, key: &str, diag: &mut Diag) -> Option<&'v str> {
        self.typed(key, "a string", diag, Value::as_str)
    }

    fn required_string(&mut self, key: &str, diag: &mut Diag) -> Option<&'v str> {
        if self.get(key).is_none() {
            diag.error(format!("`{}` is required", self.key_path(key)));
            return None;
        }
        self.string(key, diag)
    }

    fn u64(&mut self, key: &str, diag: &mut Diag) -> Option<u64> {
        self.typed(key, "a non-negative integer", diag, Value::as_u64)
    }

    fn u32(&mut self, key: &str, diag: &mut Diag) -> Option<u32> {
        self.typed(key, "a non-negative integer below 2^32", diag, |v| {
            v.as_u64().and_then(|n| u32::try_from(n).ok())
        })
    }

    fn bool(&mut self, key: &str, diag: &mut Diag) -> Option<bool> {
        self.typed(key, "true or false", diag, Value::as_bool)
    }

    fn positive_f64(&mut self, key: &str, diag: &mut Diag) -> Option<f64> {
        self.typed(key, "a positive number", diag, |v| {
            v.as_f64().filter(|n| n.is_finite() && *n > 0.0)
        })
    }

    /// A duration in seconds (integer or decimal).
    fn seconds(&mut self, key: &str, allow_zero: bool, diag: &mut Diag) -> Option<Duration> {
        let expected = if allow_zero {
            "a non-negative number of seconds"
        } else {
            "a positive number of seconds"
        };
        self.typed(key, expected, diag, |v| {
            let n = v.as_f64()?;
            if n < 0.0 || (n == 0.0 && !allow_zero) {
                return None;
            }
            Duration::try_from_secs_f64(n).ok()
        })
    }

    /// Keys never read (in key order: `serde_json::Map` is sorted).
    fn leftovers(&self) -> impl Iterator<Item = (&'v String, &'v Value)> + '_ {
        self.map
            .iter()
            .filter(|(k, _)| !self.used.contains(k.as_str()))
    }

    /// Applies the key policy to every key that was not read.
    fn finish(self, kind: Kind, diag: &mut Diag) {
        for (key, _) in self.leftovers() {
            report_key(&self.key_path(key), classify(kind, key), diag);
        }
    }
}

/// Which section a key sits in, for the key policy.
#[derive(Clone, Copy)]
enum Kind {
    Root,
    General,
    Litellm,
    Router,
    ModelEntry,
    LitellmParams,
}

enum KeyClass {
    /// Excluded by design; startup fails.
    Forbidden(&'static str),
    /// Not supported, and ignoring it would change behaviour; startup fails.
    Unsupported(&'static str),
    /// Harmless and not implemented; warn.
    Ignored(&'static str),
    Unknown,
}

const DB: &str = "ferrolite uses no database";
const UI: &str = "there is no admin UI";
const CALLBACKS: &str = "callbacks and logging integrations are not loaded";
const GUARDRAILS: &str = "guardrails that run user-supplied code are not supported";
const MCP: &str = "there is no MCP gateway";
const PASSTHROUGH: &str = "pass-through endpoints to arbitrary URLs are not supported";
const KEYS: &str = "virtual keys, users, teams and budgets are not supported";
const CACHE: &str = "response caching needs an external store";
const CUSTOM_CODE: &str = "custom auth code is not loaded";
const ALERTING: &str = "alerting integrations are not loaded";
const CLIENT_CREDS: &str = "client-supplied credentials are not accepted";
const MOCK: &str = "mock responses are not supported";
const NOT_IMPLEMENTED: &str = "ferrolite does not implement it";

fn classify(kind: Kind, key: &str) -> KeyClass {
    use KeyClass::{Forbidden, Ignored, Unknown, Unsupported};

    let prefixed = |prefix: &str| key.starts_with(prefix);
    match kind {
        Kind::Root => match key {
            "guardrails" => Forbidden(GUARDRAILS),
            "callback_settings" => Forbidden(CALLBACKS),
            "pass_through_endpoints" => Forbidden(PASSTHROUGH),
            "include" => Unsupported("inline the included files into config.yaml"),
            "credential_list" => Unsupported("set api_key on each deployment instead"),
            "environment_variables" => Ignored(
                "ferrolite does not set environment variables from config; export them before starting",
            ),
            _ if prefixed("mcp_") => Forbidden(MCP),
            _ => Unknown,
        },
        Kind::General => match key {
            "store_model_in_db" | "store_prompts_in_spend_logs" | "disable_spend_logs" => {
                Forbidden(DB)
            }
            "custom_auth" | "custom_sso" | "custom_key_generate" => Forbidden(CUSTOM_CODE),
            "alerting" | "alert_types" | "alert_to_webhook_url" => Forbidden(ALERTING),
            "pass_through_endpoints" => Forbidden(PASSTHROUGH),
            "enable_jwt_auth" | "litellm_jwtauth" => Forbidden(KEYS),
            _ if prefixed("database_") => Forbidden(DB),
            _ if prefixed("ui_") => Forbidden(UI),
            _ if prefixed("alerting_") => Forbidden(ALERTING),
            _ => Unknown,
        },
        Kind::Litellm => match key {
            "callbacks" => Forbidden(CALLBACKS),
            "cache" | "cache_params" => Forbidden(CACHE),
            "guardrails" => Forbidden(GUARDRAILS),
            "max_budget"
            | "budget_duration"
            | "max_end_user_budget"
            | "default_team_settings"
            | "default_internal_user_params"
            | "default_key_generate_params"
            | "upperbound_key_generate_params"
            | "key_generation_settings" => Forbidden(KEYS),
            "set_verbose"
            | "json_logs"
            | "turn_off_message_logging"
            | "telemetry"
            | "redact_user_api_key_info"
            | "modify_params" => Ignored(NOT_IMPLEMENTED),
            _ if key.ends_with("_callback") || key.ends_with("_callbacks") => Forbidden(CALLBACKS),
            _ => Unknown,
        },
        Kind::Router => match key {
            "provider_budget_config" => Forbidden(KEYS),
            "enable_pre_call_checks"
            | "retry_after"
            | "client_ttl"
            | "routing_strategy_args"
            | "retry_policy"
            | "model_group_retry_policy"
            | "allowed_fails_policy" => Ignored(NOT_IMPLEMENTED),
            _ if prefixed("redis_") => Forbidden(DB),
            _ if prefixed("cache_") => Forbidden(CACHE),
            _ => Unknown,
        },
        Kind::ModelEntry => Unknown,
        Kind::LitellmParams => match key {
            "mock_response" => Forbidden(MOCK),
            "configurable_clientside_auth_params" => Forbidden(CLIENT_CREDS),
            "litellm_credential_name" => Unsupported("set api_key directly"),
            "base_url" => Unsupported("use api_base"),
            "api_version"
            | "organization"
            | "max_retries"
            | "tags"
            | "region_name"
            | "use_in_pass_through"
            | "merge_reasoning_content_in_choices"
            | "max_budget"
            | "budget_duration" => Ignored(NOT_IMPLEMENTED),
            _ if key.contains("_cost_per_") || prefixed("cost_per_") => {
                Ignored("cost tracking is not implemented")
            }
            _ if prefixed("litellm_") => Unsupported("litellm_* keys are internal to LiteLLM"),
            _ if ["aws_", "vertex_", "azure_", "watsonx_"]
                .iter()
                .any(|p| prefixed(p)) =>
            {
                Unsupported("this provider is not supported")
            }
            _ => Unknown,
        },
    }
}

fn report_key(path: &str, class: KeyClass, diag: &mut Diag) {
    match class {
        KeyClass::Forbidden(reason) => {
            diag.error(format!(
                "`{path}` is excluded by design ({reason}); remove it"
            ));
        }
        KeyClass::Unsupported(hint) => diag.error(format!("`{path}` is not supported: {hint}")),
        KeyClass::Ignored(why) => diag.warn(format!("`{path}` is ignored: {why}")),
        KeyClass::Unknown => diag.warn(format!("unknown key `{path}` is ignored")),
    }
}

/// Builds the config, recording problems in `cx.diag`. Returns `None` only
/// when no config could be assembled at all; callers must also check for
/// recorded errors.
fn build(root: &Value, options: &LoadOptions, cx: &mut Cx<'_>) -> Option<Config> {
    let Value::Object(_) = root else {
        cx.diag
            .error("the top level must be a mapping with a `model_list` key");
        return None;
    };
    let mut top = Section::new(String::new(), Some(root), &mut cx.diag)?;

    let model_list = top.get("model_list");
    let general = top.get("general_settings");
    let litellm = top.get("litellm_settings");
    let router = top.get("router_settings");
    top.finish(Kind::Root, &mut cx.diag);

    let general = parse_general(general, options, cx);
    let (litellm, drop_params) = parse_litellm_settings(litellm, cx);
    let router = parse_router_settings(router, litellm, cx);
    let deployments = parse_model_list(model_list, cx);
    check_references(&deployments, &router, &mut cx.diag);

    Some(Config {
        master_key: general.master_key,
        max_request_bytes: general.max_request_bytes,
        max_response_bytes: general.max_response_bytes,
        deployments,
        router,
        drop_params,
    })
}

#[derive(Default)]
struct General {
    master_key: Option<Secret>,
    max_request_bytes: Option<usize>,
    max_response_bytes: Option<usize>,
}

fn parse_general(value: Option<&Value>, options: &LoadOptions, cx: &mut Cx<'_>) -> General {
    let Some(mut sec) = Section::new("general_settings".into(), value, &mut cx.diag) else {
        return General::default();
    };
    let raw = sec.string("master_key", &mut cx.diag);
    let path = sec.key_path("master_key");
    let mut megabytes = |key: &str, d: &mut Diag| {
        sec.positive_f64(key, d)
            .map(|mb| (mb * 1024.0 * 1024.0) as usize)
    };
    let max_request_bytes = megabytes("max_request_size_mb", &mut cx.diag);
    let max_response_bytes = megabytes("max_response_size_mb", &mut cx.diag);
    sec.finish(Kind::General, &mut cx.diag);
    General {
        master_key: check_master_key(raw, &path, options, cx),
        max_request_bytes,
        max_response_bytes,
    }
}

fn check_master_key(
    raw: Option<&str>,
    path: &str,
    options: &LoadOptions,
    cx: &mut Cx<'_>,
) -> Option<Secret> {
    if options.allow_no_auth {
        if !options.bind.is_loopback() {
            cx.diag.error(format!(
                "--allow-no-auth is refused unless the bind address is loopback (it is {})",
                options.bind
            ));
        }
        if raw.is_some() {
            cx.diag.warn(
                "`general_settings.master_key` is set but --allow-no-auth disables authentication",
            );
        }
        return None;
    }

    let Some(raw) = raw else {
        cx.diag.error(
            "`general_settings.master_key` is required (or pass --allow-no-auth on a loopback address)",
        );
        return None;
    };
    if SecretRef::parse(raw).is_ok_and(|r| r.is_literal()) {
        cx.has_literal_secret = true;
    }
    let key = cx.secret(path, raw)?;
    let k = key.expose();
    if !k.starts_with("sk-") {
        cx.diag.error(format!("`{path}` must start with `sk-`"));
    }
    if k.chars().count() < MIN_MASTER_KEY_CHARS {
        cx.diag.error(format!(
            "`{path}` must be at least {MIN_MASTER_KEY_CHARS} characters long"
        ));
    }
    if !k.bytes().all(|b| b.is_ascii_graphic()) {
        cx.diag.error(format!(
            "`{path}` must contain only printable ASCII without spaces"
        ));
    }
    Some(key)
}

/// Router keys LiteLLM also accepts under `litellm_settings`.
#[derive(Default)]
struct RouterOverrides {
    num_retries: Option<u32>,
    timeout: Option<Duration>,
    fallbacks: Option<Vec<Fallback>>,
    context_window_fallbacks: Option<Vec<Fallback>>,
    content_policy_fallbacks: Option<Vec<Fallback>>,
}

fn parse_litellm_settings(value: Option<&Value>, cx: &mut Cx<'_>) -> (RouterOverrides, bool) {
    let Some(mut sec) = Section::new("litellm_settings".into(), value, &mut cx.diag) else {
        return (RouterOverrides::default(), false);
    };
    let d = &mut cx.diag;
    let drop_params = sec.bool("drop_params", d).unwrap_or(false);
    let overrides = RouterOverrides {
        num_retries: sec.u32("num_retries", d),
        timeout: sec.seconds("request_timeout", false, d),
        fallbacks: parse_fallbacks(&mut sec, "fallbacks", d),
        context_window_fallbacks: parse_fallbacks(&mut sec, "context_window_fallbacks", d),
        content_policy_fallbacks: parse_fallbacks(&mut sec, "content_policy_fallbacks", d),
    };
    sec.finish(Kind::Litellm, d);
    (overrides, drop_params)
}

fn parse_router_settings(
    value: Option<&Value>,
    litellm: RouterOverrides,
    cx: &mut Cx<'_>,
) -> RouterSettings {
    let mut out = RouterSettings::default();
    let Some(mut sec) = Section::new("router_settings".into(), value, &mut cx.diag) else {
        return out;
    };
    let d = &mut cx.diag;

    if let Some(name) = sec.string("routing_strategy", d) {
        out.routing_strategy = match name {
            "simple-shuffle" => RoutingStrategy::SimpleShuffle,
            "least-busy" => RoutingStrategy::LeastBusy,
            "latency-based-routing" => RoutingStrategy::LatencyBased,
            other => {
                d.warn(format!(
                    "`router_settings.routing_strategy` `{other}` is not supported; using simple-shuffle"
                ));
                RoutingStrategy::SimpleShuffle
            }
        };
    }
    if let Some(v) = sec.get("model_group_alias") {
        out.model_group_alias = parse_aliases(&sec.key_path("model_group_alias"), v, d);
    }

    fn merge<T>(
        router: Option<T>,
        litellm: Option<T>,
        router_key: &str,
        litellm_key: &str,
        d: &mut Diag,
    ) -> Option<T> {
        if router.is_some() && litellm.is_some() {
            d.warn(format!(
                "`litellm_settings.{litellm_key}` is overridden by `router_settings.{router_key}`"
            ));
        }
        router.or(litellm)
    }

    let num_retries = sec.u32("num_retries", d);
    out.num_retries = merge(
        num_retries,
        litellm.num_retries,
        "num_retries",
        "num_retries",
        d,
    )
    .unwrap_or(0);
    let timeout = sec.seconds("timeout", false, d);
    out.timeout = merge(timeout, litellm.timeout, "timeout", "request_timeout", d);
    if let Some(n) = sec.u32("allowed_fails", d) {
        out.allowed_fails = n;
    }
    if let Some(t) = sec.seconds("cooldown_time", true, d) {
        out.cooldown_time = t;
    }
    for (key, slot, from_litellm) in [
        ("fallbacks", &mut out.fallbacks, litellm.fallbacks),
        (
            "context_window_fallbacks",
            &mut out.context_window_fallbacks,
            litellm.context_window_fallbacks,
        ),
        (
            "content_policy_fallbacks",
            &mut out.content_policy_fallbacks,
            litellm.content_policy_fallbacks,
        ),
    ] {
        let from_router = parse_fallbacks(&mut sec, key, d);
        *slot = merge(from_router, from_litellm, key, key, d).unwrap_or_default();
    }
    sec.finish(Kind::Router, d);
    out
}

/// LiteLLM's `[{"group": ["fallback", ...]}, ...]` form. A bare string is
/// accepted for a single fallback.
fn parse_fallbacks(sec: &mut Section<'_>, key: &str, d: &mut Diag) -> Option<Vec<Fallback>> {
    let value = sec.get(key)?;
    let path = sec.key_path(key);
    let Some(items) = value.as_array() else {
        d.error(format!(
            "`{path}` must be a list like [{{\"group\": [\"fallback\"]}}]"
        ));
        return None;
    };
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let Some(map) = item.as_object() else {
            d.error(format!(
                "`{path}[{i}]` must be a mapping of group to fallbacks"
            ));
            continue;
        };
        for (from, to) in map {
            let to = match to {
                Value::String(s) => vec![s.clone()],
                Value::Array(list) if list.iter().all(Value::is_string) => list
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect(),
                _ => {
                    d.error(format!(
                        "`{path}[{i}].{from}` must be a list of model group names"
                    ));
                    continue;
                }
            };
            out.push(Fallback {
                from: from.clone(),
                to,
            });
        }
    }
    Some(out)
}

fn parse_aliases(path: &str, value: &Value, d: &mut Diag) -> BTreeMap<String, GroupAlias> {
    let mut out = BTreeMap::new();
    let Some(map) = value.as_object() else {
        d.error(format!(
            "`{path}` must be a mapping of alias to model group"
        ));
        return out;
    };
    for (alias, target) in map {
        let alias_path = child(path, alias);
        let parsed = match target {
            Value::String(s) => Some(GroupAlias {
                target: s.clone(),
                hidden: false,
            }),
            Value::Object(_) => match Section::new(alias_path.clone(), Some(target), d) {
                Some(mut sec) => {
                    let target = sec.required_string("model", d).map(str::to_owned);
                    let hidden = sec.bool("hidden", d).unwrap_or(false);
                    sec.finish(Kind::ModelEntry, d);
                    target.map(|target| GroupAlias { target, hidden })
                }
                None => None,
            },
            _ => {
                d.error(format!(
                    "`{alias_path}` must be a model group name or {{model, hidden}}"
                ));
                None
            }
        };
        if let Some(a) = parsed {
            out.insert(alias.clone(), a);
        }
    }
    out
}

fn parse_model_list(value: Option<&Value>, cx: &mut Cx<'_>) -> Vec<Deployment> {
    let items = match value {
        Some(Value::Array(items)) if !items.is_empty() => items,
        Some(Value::Array(_)) | None => {
            cx.diag
                .error("`model_list` must list at least one deployment");
            return Vec::new();
        }
        Some(_) => {
            cx.diag.error("`model_list` must be a list");
            return Vec::new();
        }
    };
    let mut per_group: BTreeMap<String, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if let Some(d) = parse_deployment(&format!("model_list[{i}]"), item, cx, &mut per_group) {
            out.push(d);
        }
    }
    out
}

fn parse_deployment(
    path: &str,
    value: &Value,
    cx: &mut Cx<'_>,
    per_group: &mut BTreeMap<String, usize>,
) -> Option<Deployment> {
    let mut entry = Section::new(path.to_owned(), Some(value), &mut cx.diag)?;
    let model_name = entry.required_string("model_name", &mut cx.diag);
    let params_value = entry.get("litellm_params");
    let info_value = entry.get("model_info");
    let params_path = entry.key_path("litellm_params");
    if params_value.is_none() {
        cx.diag.error(format!("`{params_path}` is required"));
    }
    let info_path = entry.key_path("model_info");
    entry.finish(Kind::ModelEntry, &mut cx.diag);

    let model_name = model_name?;
    let name_path = child(path, "model_name");
    if model_name.is_empty() {
        cx.diag.error(format!("`{name_path}` must not be empty"));
    }
    let wildcard = model_name.ends_with('*');
    if model_name.trim_end_matches('*').contains('*') || model_name.ends_with("**") {
        cx.diag.error(format!(
            "`{name_path}` may contain `*` only as its last character"
        ));
    }
    let model_info = parse_model_info(&info_path, info_value, &mut cx.diag);

    let mut params = Section::new(params_path, Some(params_value?), &mut cx.diag)?;
    let d = &mut cx.diag;
    let model = params.required_string("model", d);
    let custom_provider = params.string("custom_llm_provider", d);
    let api_base_raw = params.string("api_base", d);
    let api_key_raw = params.string("api_key", d);
    let rpm = params.u64("rpm", d);
    let tpm = params.u64("tpm", d);
    let weight = params.positive_f64("weight", d);
    let order = params.u32("order", d);
    let timeout = params.seconds("timeout", false, d);
    let stream_timeout = params.seconds("stream_timeout", false, d);
    let headers_value = params.get("extra_headers");
    let drop_params = params.bool("drop_params", d);
    let additional_drop_params = params
        .typed("additional_drop_params", "a list of strings", d, |v| {
            v.as_array()?
                .iter()
                .map(|x| x.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_default();
    let headers_path = params.key_path("extra_headers");
    let model_path = params.key_path("model");

    let (provider, upstream_model) = parse_provider(&model_path, model?, custom_provider, d)?;

    let api_base_path = params.key_path("api_base");
    let api_base = match api_base_raw {
        Some(raw) => {
            let resolved = cx.resolve(&api_base_path, raw)?;
            parse_api_base(&api_base_path, resolved.expose(), provider, &mut cx.diag)
        }
        None => match provider.default_api_base() {
            Some(default) => parse_api_base(&api_base_path, default, provider, &mut cx.diag),
            None => {
                cx.diag.error(format!(
                    "`{api_base_path}` is required for {}/ models",
                    provider.prefix()
                ));
                None
            }
        },
    };

    let api_key_path = params.key_path("api_key");
    let api_key = api_key_raw.and_then(|raw| {
        if SecretRef::parse(raw).is_ok_and(|r| r.is_literal()) {
            cx.has_literal_secret = true;
        }
        cx.secret(&api_key_path, raw)
    });
    let needs_key = match provider {
        Provider::Anthropic => true,
        Provider::OpenAiCompat(_) => api_base
            .as_ref()
            .and_then(Url::host_str)
            .is_some_and(|h| h.eq_ignore_ascii_case("api.openai.com")),
    };
    if needs_key && api_key_raw.is_none() {
        cx.diag
            .error(format!("`{api_key_path}` is required for this provider"));
    }

    let extra_headers = match headers_value {
        Some(v) => parse_extra_headers(&headers_path, v, cx),
        None => Vec::new(),
    };
    let defaults = request_defaults(&params, &mut cx.diag);

    let n = per_group.entry(model_name.to_owned()).or_insert(0);
    let id = format!("{model_name}#{n}");
    *n += 1;

    if upstream_model == "*" && !wildcard {
        cx.diag.error(format!(
            "`{model_path}` may be `{}/*` only when `{name_path}` ends with `*`",
            provider.prefix()
        ));
    }

    Some(Deployment {
        id,
        model_name: model_name.to_owned(),
        provider,
        upstream_model,
        api_base: api_base?,
        api_key,
        rpm,
        tpm,
        weight,
        order,
        timeout,
        stream_timeout,
        extra_headers,
        drop_params,
        additional_drop_params,
        defaults,
        model_info,
    })
}

fn parse_provider(
    path: &str,
    model: &str,
    custom_provider: Option<&str>,
    d: &mut Diag,
) -> Option<(Provider, String)> {
    let supported = || {
        PROVIDERS
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let (prefix, upstream) = match (custom_provider, model.split_once('/')) {
        (Some(p), Some((prefix, rest))) if prefix == p => (p, rest),
        (Some(p), _) => (p, model),
        (None, Some((prefix, rest))) => (prefix, rest),
        (None, None) => {
            d.error(format!(
                "`{path}` has no provider prefix; write it as e.g. `openai/{model}` \
                 (supported: {})",
                supported()
            ));
            return None;
        }
    };
    let Some(provider) = Provider::from_prefix(prefix) else {
        d.error(format!(
            "`{path}` uses provider `{prefix}`, which ferrolite does not support (supported: {})",
            supported()
        ));
        return None;
    };
    if upstream.is_empty() {
        d.error(format!("`{path}` has no model after `{prefix}/`"));
        return None;
    }
    Some((provider, upstream.to_owned()))
}

/// Validates an upstream base URL. Messages never include the URL itself:
/// it may have come from a secret reference, and it is internal topology.
fn parse_api_base(path: &str, raw: &str, provider: Provider, d: &mut Diag) -> Option<Url> {
    let Ok(mut url) = Url::parse(raw) else {
        d.error(format!("`{path}` is not a valid absolute URL"));
        return None;
    };
    let mut ok = true;
    if !matches!(url.scheme(), "http" | "https") {
        d.error(format!("`{path}` must use http:// or https://"));
        ok = false;
    }
    if !url.username().is_empty() || url.password().is_some() {
        d.error(format!(
            "`{path}` must not embed credentials; use api_key or extra_headers"
        ));
        ok = false;
    }
    if url.query().is_some() || url.fragment().is_some() {
        d.error(format!(
            "`{path}` must not contain a query string or fragment"
        ));
        ok = false;
    }
    let Some(host) = url.host() else {
        d.error(format!("`{path}` must include a host"));
        return None;
    };
    if !ok {
        return None;
    }
    if url.scheme() == "http" && !host_is_local(&host) {
        d.warn(format!(
            "`{path}` uses plain http:// to a host that is not loopback or private; \
             requests and the api_key travel unencrypted"
        ));
    }
    if provider.is_ollama() {
        let trimmed = url.path().trim_end_matches('/').to_owned();
        if !trimmed.ends_with("/v1") {
            url.set_path(&format!("{trimmed}/v1"));
        }
    }
    Some(url)
}

fn host_is_local(host: &Host<&str>) -> bool {
    match host {
        Host::Ipv4(ip) => ipv4_is_local(*ip),
        Host::Ipv6(ip) => ipv6_is_local(*ip),
        Host::Domain(name) => {
            let name = name.trim_end_matches('.').to_ascii_lowercase();
            // A single-label name (no dot) can only resolve locally.
            !name.contains('.')
                || [".localhost", ".local", ".internal", ".home.arpa", ".lan"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix))
        }
    }
}

fn ipv4_is_local(ip: Ipv4Addr) -> bool {
    let [a, b, ..] = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        // 100.64.0.0/10, carrier-grade NAT (used by Tailscale and similar).
        || (a == 100 && (64..=127).contains(&b))
}

fn ipv6_is_local(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return ipv4_is_local(v4);
    }
    let first = ip.segments()[0];
    ip.is_loopback()
        || ip.is_unspecified()
        || (first & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (first & 0xffc0) == 0xfe80 // fe80::/10 link local
}

/// Headers that would break or smuggle the upstream request if overridden.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "upgrade",
    "te",
    "trailer",
    "keep-alive",
    "proxy-connection",
];

fn parse_extra_headers(path: &str, value: &Value, cx: &mut Cx<'_>) -> Vec<(String, HeaderValue)> {
    let mut out = Vec::new();
    let Some(map) = value.as_object() else {
        cx.diag.error(format!(
            "`{path}` must be a mapping of header name to value"
        ));
        return out;
    };
    for (name, value) in map {
        let header_path = child(path, name);
        let lower = name.to_ascii_lowercase();
        let valid_name = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b));
        if !valid_name {
            cx.diag
                .error(format!("`{header_path}` is not a valid HTTP header name"));
            continue;
        }
        if RESERVED_HEADERS.contains(&lower.as_str()) {
            cx.diag.error(format!("`{header_path}` cannot be set"));
            continue;
        }
        let Some(raw) = value.as_str() else {
            cx.diag.error(format!("`{header_path}` must be a string"));
            continue;
        };
        let reference = match SecretRef::parse(raw) {
            Ok(r) => r,
            Err(e) => {
                cx.diag.error(format!("`{header_path}`: {e}"));
                continue;
            }
        };
        let sensitive = [
            "authorization",
            "key",
            "token",
            "secret",
            "cookie",
            "password",
        ]
        .iter()
        .any(|word| lower.contains(word));
        let header = if sensitive || !reference.is_literal() {
            if reference.is_literal() {
                cx.has_literal_secret = true;
            }
            cx.secret(&header_path, raw).map(HeaderValue::Secret)
        } else {
            Some(HeaderValue::Plain(raw.to_owned()))
        };
        let Some(header) = header else { continue };
        if header
            .expose()
            .bytes()
            .any(|b| b == b'\r' || b == b'\n' || b == 0)
        {
            cx.diag.error(format!(
                "`{header_path}` must not contain CR, LF or NUL characters"
            ));
            continue;
        }
        out.push((name.clone(), header));
    }
    out
}

/// Request fields a deployment default must never set.
const NOT_DEFAULTABLE: &[&str] = &["model", "messages", "stream", "prompt", "input"];

/// Turns the `litellm_params` keys not read so far into request defaults,
/// after the key policy has had its say.
fn request_defaults(params: &Section<'_>, d: &mut Diag) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in params.leftovers() {
        let path = params.key_path(key);
        match classify(Kind::LitellmParams, key) {
            KeyClass::Unknown => {}
            class => {
                report_key(&path, class, d);
                continue;
            }
        }
        if NOT_DEFAULTABLE.contains(&key.as_str()) {
            d.error(format!("`{path}` cannot be set as a request default"));
            continue;
        }
        if value
            .as_str()
            .is_some_and(|s| SecretRef::parse(s).map_or(true, |r| !r.is_literal()))
        {
            d.error(format!(
                "`{path}`: secret references are supported only in master_key, api_key, \
                 api_base and extra_headers"
            ));
            continue;
        }
        if value.is_null() {
            continue;
        }
        out.insert(key.clone(), value.clone());
    }
    out
}

fn parse_model_info(path: &str, value: Option<&Value>, d: &mut Diag) -> ModelInfo {
    let Some(mut sec) = Section::new(path.to_owned(), value, d) else {
        return ModelInfo::default();
    };
    let mut info = ModelInfo {
        mode: sec.string("mode", d).map(str::to_owned),
        max_tokens: sec.u64("max_tokens", d),
        max_input_tokens: sec.u64("max_input_tokens", d),
        max_output_tokens: sec.u64("max_output_tokens", d),
        extra: Map::new(),
    };
    for (k, v) in sec.leftovers() {
        info.extra.insert(k.clone(), v.clone());
    }
    info
}

/// Every alias and fallback must point at something that exists.
fn check_references(deployments: &[Deployment], router: &RouterSettings, d: &mut Diag) {
    let groups: BTreeSet<&str> = deployments.iter().map(|x| x.model_name.as_str()).collect();
    let matches_group = |name: &str| {
        groups.contains(name)
            || groups
                .iter()
                .filter_map(|g| g.strip_suffix('*'))
                .any(|prefix| name.starts_with(prefix))
    };

    for (alias, target) in &router.model_group_alias {
        let path = format!("router_settings.model_group_alias.{alias}");
        if groups.contains(alias.as_str()) {
            d.error(format!("`{path}` has the same name as a model group"));
        }
        if !matches_group(&target.target) {
            d.error(format!(
                "`{path}` points at `{}`, which is not a model group",
                target.target
            ));
        }
    }

    let known = |name: &str| matches_group(name) || router.model_group_alias.contains_key(name);
    for (key, list) in [
        ("fallbacks", &router.fallbacks),
        ("context_window_fallbacks", &router.context_window_fallbacks),
        ("content_policy_fallbacks", &router.content_policy_fallbacks),
    ] {
        for fb in list {
            for name in std::iter::once(&fb.from).chain(&fb.to) {
                if !known(name) {
                    d.error(format!(
                        "`{key}` names `{name}`, which is not a model group or alias"
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::{KeychainRunner, Resolver};
    use crate::test_support::TempDir;
    use std::collections::HashMap;
    use std::os::unix::fs::PermissionsExt;

    const OPENAI_KEY: &str = "sk-openai-test-key-000000";
    const MASTER_KEY: &str = "sk-master-0123456789abcdef0123456789";

    const MODELS: &str = "
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
";

    const AUTH: &str = "
general_settings:
  master_key: os.environ/MASTER_KEY
";

    struct NoKeychain;

    impl KeychainRunner for NoKeychain {
        fn find_generic_password(&self, _: &str, _: &str) -> Result<Secret, String> {
            Err("no such item in the keychain".to_owned())
        }
    }

    fn env() -> HashMap<String, String> {
        [
            ("OPENAI_API_KEY", OPENAI_KEY),
            ("MASTER_KEY", MASTER_KEY),
            ("ANTHROPIC_API_KEY", "sk-ant-test-key-000000"),
            ("SHORT", "abc"),
            ("VLLM_BASE", "http://10.0.0.7:8000/v1"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
    }

    fn local() -> LoadOptions {
        LoadOptions {
            bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            allow_no_auth: false,
        }
    }

    fn load_with(yaml: &str, options: LoadOptions) -> Result<Loaded, ConfigError> {
        let env = env();
        let resolver = Resolver::new(&env, &NoKeychain);
        load_str(
            yaml,
            Path::new("config.yaml"),
            Path::new("/"),
            &options,
            &resolver,
        )
    }

    fn ok(yaml: &str) -> Loaded {
        match load_with(yaml, local()) {
            Ok(l) => l,
            Err(e) => panic!("expected a valid config, got: {e}"),
        }
    }

    fn problems(yaml: &str) -> Vec<String> {
        match load_with(yaml, local()) {
            Err(ConfigError::Invalid { problems, .. }) => problems,
            Err(e) => panic!("expected validation problems, got: {e}"),
            Ok(_) => panic!("expected validation problems, config loaded"),
        }
    }

    fn yaml_error(yaml: &str) -> String {
        match load_with(yaml, local()) {
            Err(ConfigError::Yaml { message, .. }) => message,
            Err(e) => panic!("expected a YAML error, got: {e}"),
            Ok(_) => panic!("expected a YAML error, config loaded"),
        }
    }

    fn has(list: &[String], needle: &str) -> bool {
        list.iter().any(|m| m.contains(needle))
    }

    #[test]
    fn loads_minimal_config() {
        let l = ok(&format!("{MODELS}{AUTH}"));
        let c = &l.config;
        assert_eq!(c.master_key.as_ref().unwrap().expose(), MASTER_KEY);
        assert_eq!(c.deployments.len(), 1);
        let d = &c.deployments[0];
        assert_eq!(d.id, "gpt-4o#0");
        assert_eq!(d.provider, Provider::OpenAiCompat(OpenAiFlavor::OpenAi));
        assert_eq!(d.upstream_model, "gpt-4o");
        assert_eq!(d.api_base.as_str(), "https://api.openai.com/v1");
        assert_eq!(d.api_key.as_ref().unwrap().expose(), OPENAI_KEY);
        assert!(l.warnings.is_empty(), "{:?}", l.warnings);
        assert_eq!(
            l.redactor.redact(&format!("x{OPENAI_KEY}x")),
            "x[REDACTED]x"
        );
        assert_eq!(l.redactor.redact(MASTER_KEY), "[REDACTED]");
    }

    #[test]
    fn debug_output_contains_no_secrets() {
        let yaml = format!(
            "{MODELS}      extra_headers:\n        Authorization: Bearer sk-literal-header-secret\n{AUTH}"
        );
        let l = ok(&yaml);
        let dump = format!("{l:?}");
        for secret in [OPENAI_KEY, MASTER_KEY, "sk-literal-header-secret"] {
            assert!(!dump.contains(secret), "Debug leaked {secret}");
        }
    }

    #[test]
    fn rejects_forbidden_keys_and_reports_them_together() {
        let yaml = format!(
            "{MODELS}
general_settings:
  master_key: os.environ/MASTER_KEY
  database_url: postgres://u:p@db/litellm
  store_model_in_db: true
  ui_access_mode: admin_only
  custom_auth: my_module.auth
  alerting: [slack]
litellm_settings:
  callbacks: [langfuse]
  success_callback: [s3]
  cache: true
  guardrails: []
  max_budget: 100
router_settings:
  redis_host: localhost
mcp_servers: {{}}
guardrails: []
pass_through_endpoints: []
"
        );
        let p = problems(&yaml);
        for key in [
            "general_settings.database_url",
            "general_settings.store_model_in_db",
            "general_settings.ui_access_mode",
            "general_settings.custom_auth",
            "general_settings.alerting",
            "litellm_settings.callbacks",
            "litellm_settings.success_callback",
            "litellm_settings.cache",
            "litellm_settings.guardrails",
            "litellm_settings.max_budget",
            "router_settings.redis_host",
            "`mcp_servers`",
            "`guardrails`",
            "`pass_through_endpoints`",
        ] {
            assert!(has(&p, key), "missing {key} in {p:#?}");
        }
        assert!(p.iter().all(|m| !m.contains("postgres")), "{p:#?}");
    }

    #[test]
    fn rejects_forbidden_deployment_params() {
        let yaml = format!(
            "{MODELS}      mock_response: hi
      configurable_clientside_auth_params: [api_base]
      litellm_credential_name: openai-cred
      base_url: https://example.com
      aws_region_name: us-east-1
{AUTH}"
        );
        let p = problems(&yaml);
        for key in [
            "mock_response",
            "configurable_clientside_auth_params",
            "litellm_credential_name",
            "base_url",
            "aws_region_name",
        ] {
            assert!(has(&p, key), "missing {key} in {p:#?}");
        }
    }

    #[test]
    fn warns_on_unknown_and_ignored_keys() {
        let yaml = format!(
            "{MODELS}      input_cost_per_token: 0.000001
    some_entry_key: 1
{AUTH}  max_parallel_requests: 10
litellm_settings:
  set_verbose: true
router_settings:
  enable_pre_call_checks: true
environment_variables:
  FOO: bar
something_new: 1
"
        );
        let l = ok(&yaml);
        for key in [
            "input_cost_per_token",
            "model_list[0].some_entry_key",
            "general_settings.max_parallel_requests",
            "litellm_settings.set_verbose",
            "router_settings.enable_pre_call_checks",
            "environment_variables",
            "something_new",
        ] {
            assert!(has(&l.warnings, key), "missing {key} in {:#?}", l.warnings);
        }
        assert!(l.config.deployments[0].defaults.is_empty());
    }

    #[test]
    fn yaml_hardening() {
        let dup = format!("{MODELS}{AUTH}general_settings:\n  master_key: x\n");
        assert!(yaml_error(&dup).contains("duplicate"));

        let tagged = format!("{MODELS}{AUTH}extra: !include other.yaml\n");
        assert!(yaml_error(&tagged).contains("tag"));

        let bomb = "
a: &a [x, x, x, x, x, x, x, x, x, x]
b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]
c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]
d: &d [*c, *c, *c, *c, *c, *c, *c, *c, *c, *c]
e: &e [*d, *d, *d, *d, *d, *d, *d, *d, *d, *d]
f: &f [*e, *e, *e, *e, *e, *e, *e, *e, *e, *e]
g: &g [*f, *f, *f, *f, *f, *f, *f, *f, *f, *f]
h: [*g, *g, *g, *g, *g, *g, *g, *g, *g, *g]
";
        assert!(yaml_error(bomb).contains("limits"), "{}", yaml_error(bomb));

        let big = "a".repeat(MAX_CONFIG_BYTES as usize + 1);
        assert!(matches!(
            load_with(&big, local()),
            Err(ConfigError::TooLarge { .. })
        ));

        assert!(has(&problems("- just\n- a list\n"), "top level"));
        assert!(has(&problems(""), "top level"));
    }

    #[test]
    fn yaml_errors_do_not_echo_source() {
        let yaml = "general_settings:\n  master_key: sk-leaky-literal-0123456789abcdef\n  : [\n";
        let msg = yaml_error(yaml);
        assert!(!msg.contains("sk-leaky"), "{msg}");
        assert!(msg.contains("line 3"), "{msg}");
        assert!(yaml_error("a: [\n").contains("unclosed"));
        assert!(yaml_error("a: 1\n---\nb: 2\n").contains("one YAML document"));
        assert!(yaml_error("x: *nope\n").contains("unknown anchor"));
    }

    #[test]
    fn master_key_rules() {
        assert!(has(&problems(MODELS), "master_key` is required"));

        let short = format!("{MODELS}general_settings:\n  master_key: sk-short\n");
        let p = problems(&short);
        assert!(has(&p, "at least 32"), "{p:#?}");
        assert!(p.iter().all(|m| !m.contains("sk-short")));

        let prefix = format!(
            "{MODELS}general_settings:\n  master_key: pk-0123456789abcdef0123456789abcdef\n"
        );
        assert!(has(&problems(&prefix), "start with `sk-`"));

        let space = format!(
            "{MODELS}general_settings:\n  master_key: \"sk-0123456789 abcdef0123456789abcdef\"\n"
        );
        assert!(has(&problems(&space), "printable ASCII"));
    }

    #[test]
    fn allow_no_auth_requires_loopback() {
        for bind in ["127.0.0.1", "::1"] {
            let l = load_with(
                MODELS,
                LoadOptions {
                    bind: bind.parse().unwrap(),
                    allow_no_auth: true,
                },
            )
            .unwrap();
            assert!(l.config.master_key.is_none());
        }
        for bind in ["0.0.0.0", "192.168.1.10", "::"] {
            let err = load_with(
                MODELS,
                LoadOptions {
                    bind: bind.parse().unwrap(),
                    allow_no_auth: true,
                },
            )
            .unwrap_err();
            assert!(
                err.to_string().contains("--allow-no-auth is refused"),
                "{err}"
            );
        }
        let both = load_with(
            &format!("{MODELS}{AUTH}"),
            LoadOptions {
                bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
                allow_no_auth: true,
            },
        )
        .unwrap();
        assert!(both.config.master_key.is_none());
        assert!(has(&both.warnings, "disables authentication"));
    }

    #[test]
    fn maps_provider_prefixes() {
        let yaml = format!(
            "
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: llama
    litellm_params:
      model: hosted_vllm/meta-llama/Llama-3.1-8B
      api_base: os.environ/VLLM_BASE
  - model_name: local
    litellm_params:
      model: ollama_chat/qwen3
      api_base: http://localhost:11434
  - model_name: local
    litellm_params:
      model: ollama/qwen3
  - model_name: studio
    litellm_params:
      model: lm_studio/qwen3
  - model_name: custom
    litellm_params:
      model: my-model
      custom_llm_provider: custom_openai
      api_base: https://llm.example.com/v1
{AUTH}"
        );
        let l = ok(&yaml);
        let d = &l.config.deployments;
        assert_eq!(d[0].provider, Provider::Anthropic);
        assert_eq!(d[0].api_base.as_str(), "https://api.anthropic.com/");
        assert_eq!(
            d[1].provider,
            Provider::OpenAiCompat(OpenAiFlavor::HostedVllm)
        );
        assert_eq!(d[1].upstream_model, "meta-llama/Llama-3.1-8B");
        assert_eq!(d[1].api_base.as_str(), "http://10.0.0.7:8000/v1");
        assert_eq!(
            d[2].provider,
            Provider::OpenAiCompat(OpenAiFlavor::OllamaChat)
        );
        assert_eq!(d[2].api_base.as_str(), "http://localhost:11434/v1");
        assert_eq!((d[2].id.as_str(), d[3].id.as_str()), ("local#0", "local#1"));
        assert_eq!(d[3].api_base.as_str(), "http://127.0.0.1:11434/v1");
        assert_eq!(d[4].api_base.as_str(), "http://127.0.0.1:1234/v1");
        assert_eq!(
            d[5].provider,
            Provider::OpenAiCompat(OpenAiFlavor::CustomOpenAi)
        );
        assert_eq!(d[5].upstream_model, "my-model");
        assert!(d[1].api_key.is_none());
        // The api_base resolved from the environment is not a secret.
        assert_eq!(
            l.redactor.redact("http://10.0.0.7:8000/v1"),
            "http://10.0.0.7:8000/v1"
        );
    }

    #[test]
    fn rejects_bad_provider_and_model_names() {
        let yaml = format!(
            "
model_list:
  - model_name: bare
    litellm_params:
      model: gpt-4o
  - model_name: bedrock
    litellm_params:
      model: bedrock/claude
  - model_name: empty
    litellm_params:
      model: openai/
  - model_name: vllm
    litellm_params:
      model: vllm/llama
  - model_name: claude
    litellm_params:
      model: anthropic/claude
  - model_name: star
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY
  - model_name: a*b
    litellm_params:
      model: openai/x
      api_key: os.environ/OPENAI_API_KEY
  - model_name: noparams
{AUTH}"
        );
        let p = problems(&yaml);
        for needle in [
            "model_list[0].litellm_params.model` has no provider prefix",
            "provider `bedrock`",
            "model_list[2].litellm_params.model` has no model",
            "model_list[3].litellm_params.api_base` is required",
            "model_list[4].litellm_params.api_key` is required",
            "may be `openai/*` only",
            "model_list[6].model_name` may contain `*` only",
            "model_list[7].litellm_params` is required",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:#?}");
        }
    }

    #[test]
    fn wildcard_groups() {
        let yaml = format!(
            "
model_list:
  - model_name: openai/*
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY
router_settings:
  fallbacks: [{{\"openai/gpt-4o\": [\"openai/gpt-4o-mini\"]}}]
{AUTH}"
        );
        let l = ok(&yaml);
        assert_eq!(l.config.deployments[0].upstream_model, "*");
        assert_eq!(l.config.router.fallbacks.len(), 1);
    }

    #[test]
    fn api_base_rules() {
        let with_base = |base: &str| {
            format!(
                "
model_list:
  - model_name: m
    litellm_params:
      model: custom_openai/m
      api_base: \"{base}\"
{AUTH}"
            )
        };
        for (base, needle) in [
            ("not a url", "not a valid absolute URL"),
            ("ftp://example.com", "http:// or https://"),
            (
                "https://user:pass@example.com/v1",
                "must not embed credentials",
            ),
            ("https://example.com/v1?x=1", "query string"),
            ("https://example.com/v1#frag", "query string"),
        ] {
            let p = problems(&with_base(base));
            assert!(has(&p, needle), "{base}: {p:#?}");
            assert!(p.iter().all(|m| !m.contains("pass@")), "{p:#?}");
        }

        let public_http = [
            "http://example.com/v1",
            "http://8.8.8.8/v1",
            "http://[2001:4860::8888]/v1",
        ];
        for base in public_http {
            let l = ok(&with_base(base));
            assert!(has(&l.warnings, "plain http://"), "{base}");
        }
        let local_http = [
            "http://localhost:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://10.1.2.3/v1",
            "http://172.16.0.1/v1",
            "http://192.168.0.2/v1",
            "http://169.254.1.1/v1",
            "http://100.100.1.1/v1",
            "http://[::1]:8000/v1",
            "http://[fd00::1]/v1",
            "http://[fe80::1]/v1",
            "http://[::ffff:10.0.0.1]/v1",
            "http://gpu-box:8000/v1",
            "http://gpu.local/v1",
            "http://host.docker.internal/v1",
            "https://example.com/v1",
        ];
        for base in local_http {
            let l = ok(&with_base(base));
            assert!(
                !has(&l.warnings, "plain http://"),
                "{base}: {:?}",
                l.warnings
            );
        }
    }

    #[test]
    fn extra_headers() {
        let yaml = format!(
            "{MODELS}      extra_headers:
        X-Title: ferrolite-tests
        X-Api-Token: os.environ/OPENAI_API_KEY
        Authorization: Bearer sk-literal-header-value
{AUTH}"
        );
        let l = ok(&yaml);
        let h = &l.config.deployments[0].extra_headers;
        let header = |name: &str| &h.iter().find(|(n, _)| n == name).unwrap().1;
        assert!(matches!(header("X-Title"), HeaderValue::Plain(v) if v == "ferrolite-tests"));
        assert!(
            matches!(header("X-Api-Token"), HeaderValue::Secret(s) if s.expose() == OPENAI_KEY)
        );
        assert!(matches!(header("Authorization"), HeaderValue::Secret(_)));
        assert_eq!(
            l.redactor.redact("Bearer sk-literal-header-value"),
            "[REDACTED]"
        );

        let bad = format!(
            "{MODELS}      extra_headers:
        Host: evil.example
        \"Bad Name\": x
        X-Num: 5
        X-Split: \"a\\r\\nInjected: yes\"
{AUTH}"
        );
        let p = problems(&bad);
        for needle in [
            "extra_headers.Host` cannot be set",
            "not a valid HTTP header name",
            "X-Num` must be a string",
            "CR, LF or NUL",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:#?}");
        }
    }

    #[test]
    fn size_limits_from_general_settings() {
        let yaml = format!("{MODELS}{AUTH}  max_request_size_mb: 8\n  max_response_size_mb: 0.5\n");
        let l = ok(&yaml);
        assert_eq!(l.config.max_request_bytes, Some(8 * 1024 * 1024));
        assert_eq!(l.config.max_response_bytes, Some(512 * 1024));
        assert!(l.warnings.is_empty(), "{:?}", l.warnings);
        let bad = format!("{MODELS}{AUTH}  max_request_size_mb: 0\n");
        assert!(has(
            &problems(&bad),
            "max_request_size_mb` must be a positive number"
        ));
    }

    #[test]
    fn per_deployment_drop_params() {
        let yaml = format!(
            "{MODELS}      drop_params: true\n      additional_drop_params: [logprobs, seed]\n{AUTH}"
        );
        let l = ok(&yaml);
        let d = &l.config.deployments[0];
        assert_eq!(d.drop_params, Some(true));
        assert_eq!(d.additional_drop_params, ["logprobs", "seed"]);
        assert!(d.defaults.is_empty(), "{:?}", d.defaults);

        let bad = format!("{MODELS}      additional_drop_params: logprobs\n{AUTH}");
        assert!(has(&problems(&bad), "must be a list of strings"));
    }

    #[test]
    fn request_defaults_from_litellm_params() {
        let yaml = format!(
            "{MODELS}      temperature: 0.2
      max_tokens: 1024
      rpm: 60
      tpm: 100000
      weight: 2
      order: 1
      timeout: 30
      stream_timeout: 2.5
{AUTH}"
        );
        let l = ok(&yaml);
        let d = &l.config.deployments[0];
        assert_eq!(d.defaults.get("temperature"), Some(&serde_json::json!(0.2)));
        assert_eq!(d.defaults.get("max_tokens"), Some(&serde_json::json!(1024)));
        assert_eq!(d.defaults.len(), 2);
        assert_eq!(
            (d.rpm, d.tpm, d.weight, d.order),
            (Some(60), Some(100_000), Some(2.0), Some(1))
        );
        assert_eq!(d.timeout, Some(Duration::from_secs(30)));
        assert_eq!(d.stream_timeout, Some(Duration::from_millis(2500)));

        let bad = format!(
            "{MODELS}      stream: true
      messages: []
      user: os.environ/OPENAI_API_KEY
      litellm_trace_id: abc
      rpm: -1
      weight: 0
      timeout: fast
{AUTH}"
        );
        let p = problems(&bad);
        for needle in [
            "stream` cannot be set",
            "messages` cannot be set",
            "user`: secret references are supported only",
            "litellm_trace_id` is not supported",
            "rpm` must be a non-negative integer",
            "weight` must be a positive number",
            "timeout` must be a positive number of seconds",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:#?}");
        }
    }

    #[test]
    fn model_info_is_kept() {
        let yaml = format!(
            "{MODELS}    model_info:
      mode: chat
      max_input_tokens: 128000
      id: custom-id
{AUTH}"
        );
        let l = ok(&yaml);
        let info = &l.config.deployments[0].model_info;
        assert_eq!(info.mode.as_deref(), Some("chat"));
        assert_eq!(info.max_input_tokens, Some(128_000));
        assert_eq!(info.extra.get("id"), Some(&serde_json::json!("custom-id")));
    }

    #[test]
    fn router_settings_and_merge() {
        let yaml = format!(
            "
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude
    litellm_params:
      model: anthropic/claude
      api_key: os.environ/ANTHROPIC_API_KEY
litellm_settings:
  drop_params: True
  num_retries: 5
  request_timeout: 600
  context_window_fallbacks: [{{gpt-4o: [claude]}}]
router_settings:
  routing_strategy: latency-based-routing
  num_retries: 2
  allowed_fails: 1
  cooldown_time: 30
  model_group_alias:
    smart: claude
    hidden-smart: {{model: claude, hidden: true}}
  fallbacks:
    - gpt-4o: [claude, smart]
  content_policy_fallbacks: [{{claude: gpt-4o}}]
{AUTH}"
        );
        let l = ok(&yaml);
        let r = &l.config.router;
        assert!(l.config.drop_params);
        assert_eq!(r.routing_strategy, RoutingStrategy::LatencyBased);
        assert_eq!(r.num_retries, 2);
        assert_eq!(r.timeout, Some(Duration::from_secs(600)));
        assert_eq!(r.allowed_fails, 1);
        assert_eq!(r.cooldown_time, Duration::from_secs(30));
        assert_eq!(
            r.model_group_alias.get("hidden-smart"),
            Some(&GroupAlias {
                target: "claude".into(),
                hidden: true
            })
        );
        assert_eq!(
            r.fallbacks,
            vec![Fallback {
                from: "gpt-4o".into(),
                to: vec!["claude".into(), "smart".into()]
            }]
        );
        assert_eq!(r.context_window_fallbacks[0].to, vec!["claude".to_owned()]);
        assert_eq!(r.content_policy_fallbacks[0].to, vec!["gpt-4o".to_owned()]);
        assert!(has(
            &l.warnings,
            "`litellm_settings.num_retries` is overridden"
        ));
    }

    #[test]
    fn router_defaults_and_unknown_strategy() {
        let yaml =
            format!("{MODELS}{AUTH}router_settings:\n  routing_strategy: usage-based-routing-v2\n");
        let l = ok(&yaml);
        let r = &l.config.router;
        assert_eq!(r.routing_strategy, RoutingStrategy::SimpleShuffle);
        assert_eq!((r.num_retries, r.allowed_fails), (0, DEFAULT_ALLOWED_FAILS));
        assert_eq!(r.cooldown_time, DEFAULT_COOLDOWN);
        assert!(has(&l.warnings, "usage-based-routing-v2"));
    }

    #[test]
    fn references_must_exist() {
        let yaml = format!(
            "{MODELS}router_settings:
  model_group_alias:
    gpt-4o: gpt-4o
    fast: nowhere
  fallbacks: [{{gpt-4o: [missing]}}]
  context_window_fallbacks: [{{ghost: [gpt-4o]}}]
{AUTH}"
        );
        let p = problems(&yaml);
        for needle in [
            "model_group_alias.gpt-4o` has the same name",
            "points at `nowhere`",
            "`fallbacks` names `missing`",
            "`context_window_fallbacks` names `ghost`",
        ] {
            assert!(has(&p, needle), "missing {needle:?} in {p:#?}");
        }
    }

    #[test]
    fn secret_resolution_errors_are_reported() {
        let yaml = "
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/NOT_SET
general_settings:
  master_key: keychain/ferrolite/master
";
        let p = problems(yaml);
        assert!(
            has(&p, "api_key`: environment variable `NOT_SET`"),
            "{p:#?}"
        );
        assert!(has(&p, "master_key`: keychain item"), "{p:#?}");
    }

    #[test]
    fn warns_about_short_secrets() {
        let yaml = format!(
            "
model_list:
  - model_name: local
    litellm_params:
      model: hosted_vllm/llama
      api_base: http://127.0.0.1:8000/v1
      api_key: os.environ/SHORT
{AUTH}"
        );
        let l = ok(&yaml);
        assert!(
            has(&l.warnings, "api_key` is shorter than 8"),
            "{:?}",
            l.warnings
        );
    }

    fn write_config(dir: &TempDir, yaml: &str, mode: u32) -> PathBuf {
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, yaml).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    fn load_file(path: &Path) -> Result<Loaded, ConfigError> {
        let env = env();
        let resolver = Resolver::new(&env, &NoKeychain);
        load(path, &local(), &resolver)
    }

    #[test]
    fn warns_when_readable_config_holds_literal_secrets() {
        let dir = TempDir::new();
        let literal = format!("{MODELS}general_settings:\n  master_key: {MASTER_KEY}\n");

        let path = write_config(&dir, &literal, 0o644);
        let l = load_file(&path).unwrap();
        assert!(
            has(&l.warnings, "readable by group or other"),
            "{:?}",
            l.warnings
        );

        let path = write_config(&dir, &literal, 0o600);
        assert!(!has(
            &load_file(&path).unwrap().warnings,
            "readable by group"
        ));

        let path = write_config(&dir, &format!("{MODELS}{AUTH}"), 0o644);
        assert!(!has(
            &load_file(&path).unwrap().warnings,
            "readable by group"
        ));
    }

    #[test]
    fn file_secrets_resolve_relative_to_config() {
        let dir = TempDir::new();
        let key = dir.path().join("openai.key");
        std::fs::write(&key, "sk-from-a-file-000000\n").unwrap();
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        let yaml = format!(
            "
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: file/openai.key
{AUTH}"
        );
        let path = write_config(&dir, &yaml, 0o600);
        let l = load_file(&path).unwrap();
        let got = l.config.deployments[0].api_key.as_ref().unwrap().expose();
        assert_eq!(got, "sk-from-a-file-000000");
    }

    #[test]
    fn refuses_non_regular_and_oversize_config_files() {
        let dir = TempDir::new();
        assert!(matches!(
            load_file(dir.path()),
            Err(ConfigError::NotRegular { .. })
        ));
        let big = write_config(&dir, &"#".repeat(MAX_CONFIG_BYTES as usize + 1), 0o600);
        assert!(matches!(load_file(&big), Err(ConfigError::TooLarge { .. })));
        assert!(matches!(
            load_file(&dir.path().join("missing.yaml")),
            Err(ConfigError::Io { .. })
        ));
    }
}
