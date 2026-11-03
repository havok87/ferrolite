//! OpenAI-compatible backends: OpenAI itself, vLLM, Ollama's `/v1`,
//! LM Studio and anything else that speaks the same API.

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use url::Url;

use super::{Endpoint, extra_headers};
use crate::config::Deployment;

/// `<api_base>/<endpoint>`, e.g. `http://127.0.0.1:8000/v1/chat/completions`.
pub(super) fn url(base: &Url, endpoint: Endpoint) -> Url {
    let mut url = base.clone();
    let path = format!("{}/{}", base.path().trim_end_matches('/'), endpoint.path());
    url.set_path(&path);
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
