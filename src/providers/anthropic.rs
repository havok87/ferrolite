//! Anthropic's native Messages API.

use reqwest::header::{HeaderMap, HeaderValue};
use url::Url;

use super::extra_headers;
use crate::config::Deployment;

/// Sent when the client does not choose an `anthropic-version`.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// `/v1/messages` under `api_base`, unless `api_base` already ends in it.
pub(super) fn messages_url(base: &Url) -> Url {
    let trimmed = base.path().trim_end_matches('/');
    let path = if trimmed.ends_with("/v1/messages") {
        trimmed.to_owned()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/messages")
    } else {
        format!("{trimmed}/v1/messages")
    };
    let mut url = base.clone();
    url.set_path(&path);
    url
}

/// Server-side auth plus, on passthrough, the client's `anthropic-version`
/// and `anthropic-beta`. Nothing else from the client is forwarded.
pub(super) fn headers(d: &Deployment, version: Option<&str>, beta: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Some(key) = &d.api_key
        && let Ok(mut v) = HeaderValue::from_str(key.expose())
    {
        v.set_sensitive(true);
        headers.insert("x-api-key", v);
    }
    let version = version
        .and_then(|v| HeaderValue::from_str(v).ok())
        .unwrap_or(HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION));
    headers.insert("anthropic-version", version);
    if let Some(beta) = beta.and_then(|b| HeaderValue::from_str(b).ok()) {
        headers.insert("anthropic-beta", beta);
    }
    extra_headers(d, &mut headers);
    headers
}
