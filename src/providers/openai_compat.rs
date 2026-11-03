//! OpenAI-compatible backends: OpenAI itself, vLLM, Ollama's `/v1`,
//! LM Studio and anything else that speaks the same API.

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use url::Url;

use super::{Endpoint, extra_headers};
use crate::config::Deployment;

/// `<api_base>/<endpoint>`, e.g. `http://127.0.0.1:8000/v1/chat/completions`.
pub(super) fn url(base: &Url, endpoint: Endpoint) -> Url {
    join(base, endpoint.path())
}

/// `<api_base>/models`, for health probes.
pub(super) fn models_url(base: &Url) -> Url {
    join(base, "models")
}

fn join(base: &Url, leaf: &str) -> Url {
    let mut url = base.clone();
    url.set_path(&format!("{}/{leaf}", base.path().trim_end_matches('/')));
    url
}

pub(super) fn headers(d: &Deployment) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Some(key) = &d.api_key
        && let Ok(mut v) = HeaderValue::from_str(&format!("Bearer {}", key.expose()))
    {
        v.set_sensitive(true);
        headers.insert(AUTHORIZATION, v);
    }
    extra_headers(d, &mut headers);
    headers
}
