//! Health endpoints. Liveliness and readiness need no key and touch no
//! upstream; `/health` probes every deployment, so its result is cached.

use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::Instant;

use super::AppState;
use super::metrics::class_label;

/// How long a `/health` result is reused, so repeated calls cannot turn
/// into a flood of upstream requests.
const CACHE_TTL: Duration = Duration::from_secs(10);
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const PROBE_CONCURRENCY: usize = 8;

#[derive(Default)]
pub(super) struct HealthCache {
    // A tokio mutex held across the probes, so concurrent callers share
    // one round of probing instead of starting their own.
    last: Mutex<Option<(Instant, Value)>>,
}

/// Probes every deployment (`GET .../models` with its configured key) and
/// reports by deployment id. Never includes `api_base` or keys.
pub(super) async fn backends(st: &AppState) -> Value {
    let mut last = st.health.last.lock().await;
    if let Some((at, value)) = &*last
        && at.elapsed() < CACHE_TTL
    {
        return value.clone();
    }
    let probes: Vec<_> = (0..st.config.deployments.len())
        .map(|i| probe(st, i))
        .collect();
    let mut results: Vec<(usize, Option<&'static str>)> = futures_util::stream::iter(probes)
        .buffer_unordered(PROBE_CONCURRENCY)
        .collect()
        .await;
    results.sort_by_key(|(i, _)| *i);

    let mut healthy = Vec::new();
    let mut unhealthy = Vec::new();
    for (i, error) in results {
        let d = &st.config.deployments[i];
        match error {
            None => healthy.push(json!({"model": d.model_name, "deployment": d.id})),
            Some(class) => {
                unhealthy.push(json!({"model": d.model_name, "deployment": d.id, "error": class}))
            }
        }
    }
    let value = json!({
        "healthy_count": healthy.len(),
        "unhealthy_count": unhealthy.len(),
        "healthy_endpoints": healthy,
        "unhealthy_endpoints": unhealthy,
    });
    *last = Some((Instant::now(), value.clone()));
    value
}

/// One deployment's probe result: `None` if healthy, else its error class.
async fn probe(st: &AppState, index: usize) -> (usize, Option<&'static str>) {
    let d = &st.config.deployments[index];
    let outcome = st.upstream.probe(d, PROBE_TIMEOUT).await;
    (index, outcome.err().map(|e| class_label(e.class)))
}

/// Ready while at least one deployment is not cooling down.
pub(super) fn readiness(st: &AppState) -> (u16, Value) {
    let snapshot = st.router.snapshot();
    let available = snapshot.iter().filter(|d| !d.cooling_down).count();
    let status = if available > 0 {
        "healthy"
    } else {
        "unhealthy"
    };
    let code = if available > 0 { 200 } else { 503 };
    (
        code,
        json!({"status": status, "deployments": snapshot.len(), "available": available}),
    )
}
