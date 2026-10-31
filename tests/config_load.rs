//! Loads the shipped example config through the public API.

#![forbid(unsafe_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

use ferrolite::config::{self, LoadOptions, OpenAiFlavor, Provider};
use ferrolite::secrets::{KeychainRunner, Resolver, Secret};

const MASTER_KEY: &str = "sk-integration-0123456789abcdef0123";
const OPENAI_KEY: &str = "sk-openai-integration-key";
const ANTHROPIC_KEY: &str = "sk-ant-integration-key";

struct NoKeychain;

impl KeychainRunner for NoKeychain {
    fn find_generic_password(&self, _: &str, _: &str) -> Result<Secret, String> {
        Err("no keychain in tests".to_owned())
    }
}

fn example_path() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/config.example.yaml"
    ))
}

fn env() -> HashMap<String, String> {
    [
        ("FERROLITE_MASTER_KEY", MASTER_KEY),
        ("OPENAI_API_KEY", OPENAI_KEY),
        ("ANTHROPIC_API_KEY", ANTHROPIC_KEY),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect()
}

fn options() -> LoadOptions {
    LoadOptions {
        bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
        allow_no_auth: false,
    }
}

#[test]
fn example_config_loads_cleanly() {
    let env = env();
    let resolver = Resolver::new(&env, &NoKeychain);
    let loaded = config::load(example_path(), &options(), &resolver).unwrap();
    assert!(loaded.warnings.is_empty(), "{:#?}", loaded.warnings);

    let c = &loaded.config;
    assert_eq!(c.master_key.as_ref().unwrap().expose(), MASTER_KEY);
    let ids: Vec<&str> = c.deployments.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "gpt-4o#0",
            "openai/*#0",
            "claude-sonnet#0",
            "local#0",
            "local#1"
        ]
    );
    assert_eq!(c.deployments[2].provider, Provider::Anthropic);
    assert_eq!(c.deployments[2].defaults["max_tokens"], 4096);
    assert_eq!(
        c.deployments[3].provider,
        Provider::OpenAiCompat(OpenAiFlavor::OllamaChat)
    );
    assert_eq!(
        c.deployments[3].api_base.as_str(),
        "http://127.0.0.1:11434/v1"
    );
    assert!(c.drop_params);
    assert_eq!(c.router.num_retries, 2);
    assert!(c.router.model_group_alias.contains_key("smart"));

    for secret in [MASTER_KEY, OPENAI_KEY, ANTHROPIC_KEY] {
        assert_eq!(loaded.redactor.redact(secret), "[REDACTED]");
        assert!(!format!("{loaded:?}").contains(secret));
    }
}

#[test]
fn example_config_needs_its_environment() {
    let empty = HashMap::new();
    let resolver = Resolver::new(&empty, &NoKeychain);
    let err = config::load(example_path(), &options(), &resolver).unwrap_err();
    let msg = err.to_string();
    for var in [
        "FERROLITE_MASTER_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ] {
        assert!(msg.contains(var), "{msg}");
    }
}
