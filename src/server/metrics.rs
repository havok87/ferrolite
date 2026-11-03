//! Prometheus metrics, hand-rolled with atomics.
//!
//! Every series is created at startup from fixed label sets (routes,
//! status codes, error classes) and the configured model groups, so client
//! input can never add a series.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::router::{DeploymentStatus, ErrorClass};
use crate::translate::Usage;

/// The routes requests are counted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    ChatCompletions,
    Completions,
    Embeddings,
    Responses,
    Messages,
    Models,
    ModelInfo,
    Health,
    Readiness,
    Liveliness,
    Metrics,
    Other,
}

const ROUTES: [Route; 12] = [
    Route::ChatCompletions,
    Route::Completions,
    Route::Embeddings,
    Route::Responses,
    Route::Messages,
    Route::Models,
    Route::ModelInfo,
    Route::Health,
    Route::Readiness,
    Route::Liveliness,
    Route::Metrics,
    Route::Other,
];

impl Route {
    pub fn label(self) -> &'static str {
        match self {
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Embeddings => "/v1/embeddings",
            Self::Responses => "/v1/responses",
            Self::Messages => "/v1/messages",
            Self::Models => "/v1/models",
            Self::ModelInfo => "/model/info",
            Self::Health => "/health",
            Self::Readiness => "/health/readiness",
            Self::Liveliness => "/health/liveliness",
            Self::Metrics => "/metrics",
            Self::Other => "other",
        }
    }

    fn index(self) -> usize {
        ROUTES
            .iter()
            .position(|r| *r == self)
            .unwrap_or(ROUTES.len() - 1)
    }
}

/// Status codes ferrolite can answer with; anything else counts as "other".
const CODES: [u16; 13] = [
    200, 400, 401, 404, 405, 408, 413, 422, 429, 500, 502, 503, 504,
];
const BUCKETS: [f64; 12] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];
const ERROR_CLASSES: [&str; 11] = [
    "rate_limited",
    "server",
    "timeout",
    "connection",
    "context_window",
    "content_policy",
    "auth",
    "not_found",
    "bad_request",
    "no_deployment",
    "unknown_model",
];

/// The label for an error class (also used by `/health`).
pub fn class_label(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::RateLimited { .. } => "rate_limited",
        ErrorClass::Server => "server",
        ErrorClass::Timeout => "timeout",
        ErrorClass::Connection => "connection",
        ErrorClass::ContextWindowExceeded => "context_window",
        ErrorClass::ContentPolicy => "content_policy",
        ErrorClass::Auth => "auth",
        ErrorClass::NotFound => "not_found",
        ErrorClass::BadRequest => "bad_request",
    }
}

#[derive(Default)]
struct Histogram {
    buckets: [AtomicU64; BUCKETS.len()],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

#[derive(Default)]
struct GroupCounters {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    errors: [AtomicU64; ERROR_CLASSES.len()],
    fallbacks: AtomicU64,
}

pub struct Metrics {
    requests: Vec<AtomicU64>,
    durations: Vec<Histogram>,
    groups: BTreeMap<String, GroupCounters>,
    auth_failures: AtomicU64,
}

impl Metrics {
    pub fn new(groups: impl IntoIterator<Item = String>) -> Self {
        Self {
            requests: (0..ROUTES.len() * (CODES.len() + 1))
                .map(|_| AtomicU64::new(0))
                .collect(),
            durations: (0..ROUTES.len()).map(|_| Histogram::default()).collect(),
            groups: groups
                .into_iter()
                .map(|g| (g, GroupCounters::default()))
                .collect(),
            auth_failures: AtomicU64::new(0),
        }
    }

    pub fn request(&self, route: Route, status: u16, elapsed: Duration) {
        let code = CODES
            .iter()
            .position(|c| *c == status)
            .unwrap_or(CODES.len());
        self.requests[route.index() * (CODES.len() + 1) + code].fetch_add(1, Ordering::Relaxed);
        let h = &self.durations[route.index()];
        let secs = elapsed.as_secs_f64();
        for (bucket, le) in h.buckets.iter().zip(BUCKETS) {
            if secs <= le {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
        h.count.fetch_add(1, Ordering::Relaxed);
        h.sum_micros
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
    }

    pub fn tokens(&self, group: &str, usage: Usage) {
        if let Some(g) = self.groups.get(group) {
            g.prompt_tokens.fetch_add(usage.prompt, Ordering::Relaxed);
            g.completion_tokens
                .fetch_add(usage.completion, Ordering::Relaxed);
        }
    }

    /// `class` must be one of the fixed labels; unknown groups are ignored.
    pub fn upstream_error(&self, group: &str, class: &str) {
        let (Some(g), Some(i)) = (
            self.groups.get(group),
            ERROR_CLASSES.iter().position(|c| *c == class),
        ) else {
            return;
        };
        g.errors[i].fetch_add(1, Ordering::Relaxed);
    }

    pub fn fallback(&self, group: &str) {
        if let Some(g) = self.groups.get(group) {
            g.fallbacks.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn auth_failure(&self) {
        self.auth_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Prometheus text exposition format 0.0.4.
    pub fn render(&self, in_flight: usize, deployments: &[DeploymentStatus]) -> String {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut out = String::new();
        let header = |out: &mut String, name: &str, kind: &str, help: &str| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
        };

        header(
            &mut out,
            "ferrolite_requests_total",
            "counter",
            "Requests by route and status code.",
        );
        for (r, route) in ROUTES.iter().enumerate() {
            for c in 0..=CODES.len() {
                let n = load(&self.requests[r * (CODES.len() + 1) + c]);
                if n > 0 {
                    let code = CODES
                        .get(c)
                        .map_or_else(|| "other".to_owned(), u16::to_string);
                    let _ = writeln!(
                        out,
                        "ferrolite_requests_total{{route=\"{}\",code=\"{code}\"}} {n}",
                        route.label()
                    );
                }
            }
        }

        header(
            &mut out,
            "ferrolite_request_duration_seconds",
            "histogram",
            "Request latency by route.",
        );
        for (route, h) in ROUTES.iter().zip(&self.durations) {
            let count = load(&h.count);
            if count == 0 {
                continue;
            }
            let label = route.label();
            for (bucket, le) in h.buckets.iter().zip(BUCKETS) {
                let _ = writeln!(
                    out,
                    "ferrolite_request_duration_seconds_bucket{{route=\"{label}\",le=\"{le}\"}} {}",
                    load(bucket)
                );
            }
            let _ = writeln!(
                out,
                "ferrolite_request_duration_seconds_bucket{{route=\"{label}\",le=\"+Inf\"}} {count}\n\
                 ferrolite_request_duration_seconds_sum{{route=\"{label}\"}} {}\n\
                 ferrolite_request_duration_seconds_count{{route=\"{label}\"}} {count}",
                load(&h.sum_micros) as f64 / 1e6
            );
        }

        header(
            &mut out,
            "ferrolite_tokens_total",
            "counter",
            "Tokens by model group and kind.",
        );
        for (group, g) in &self.groups {
            let group = escape(group);
            for (kind, counter) in [
                ("prompt", &g.prompt_tokens),
                ("completion", &g.completion_tokens),
            ] {
                let _ = writeln!(
                    out,
                    "ferrolite_tokens_total{{model_group=\"{group}\",kind=\"{kind}\"}} {}",
                    load(counter)
                );
            }
        }

        header(
            &mut out,
            "ferrolite_upstream_errors_total",
            "counter",
            "Failed requests by model group and error class.",
        );
        for (group, g) in &self.groups {
            let group = escape(group);
            for (class, counter) in ERROR_CLASSES.iter().zip(&g.errors) {
                let n = load(counter);
                if n > 0 {
                    let _ = writeln!(
                        out,
                        "ferrolite_upstream_errors_total{{model_group=\"{group}\",class=\"{class}\"}} {n}"
                    );
                }
            }
        }

        header(
            &mut out,
            "ferrolite_fallbacks_total",
            "counter",
            "Requests answered by a fallback group, by requested group.",
        );
        for (group, g) in &self.groups {
            let _ = writeln!(
                out,
                "ferrolite_fallbacks_total{{model_group=\"{}\"}} {}",
                escape(group),
                load(&g.fallbacks)
            );
        }

        header(
            &mut out,
            "ferrolite_auth_failures_total",
            "counter",
            "Requests rejected for a missing or wrong key.",
        );
        let _ = writeln!(
            out,
            "ferrolite_auth_failures_total {}",
            load(&self.auth_failures)
        );

        header(
            &mut out,
            "ferrolite_in_flight_requests",
            "gauge",
            "Proxied requests in progress, streams included.",
        );
        let _ = writeln!(out, "ferrolite_in_flight_requests {in_flight}");

        header(
            &mut out,
            "ferrolite_deployment_cooling_down",
            "gauge",
            "1 while a deployment is in cooldown.",
        );
        for d in deployments {
            let _ = writeln!(
                out,
                "ferrolite_deployment_cooling_down{{model_group=\"{}\",deployment=\"{}\"}} {}",
                escape(&d.model_group),
                escape(&d.id),
                u8::from(d.cooling_down)
            );
        }
        out
    }
}

/// Escapes a label value (`\`, `"` and newlines).
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_fixed_series() {
        let m = Metrics::new(["gpt".to_owned(), "a\"b".to_owned()]);
        m.request(Route::ChatCompletions, 200, Duration::from_millis(300));
        m.request(Route::ChatCompletions, 200, Duration::from_secs(3));
        m.request(Route::ChatCompletions, 418, Duration::from_millis(1));
        m.tokens(
            "gpt",
            Usage {
                prompt: 10,
                completion: 5,
            },
        );
        m.tokens(
            "not-configured",
            Usage {
                prompt: 99,
                completion: 99,
            },
        );
        m.upstream_error("gpt", "server");
        m.upstream_error("gpt", "made-up");
        m.fallback("gpt");
        m.auth_failure();
        let text = m.render(
            3,
            &[DeploymentStatus {
                id: "gpt#0".into(),
                model_group: "gpt".into(),
                in_flight: 3,
                cooling_down: true,
            }],
        );
        for line in [
            "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"200\"} 2",
            "ferrolite_requests_total{route=\"/v1/chat/completions\",code=\"other\"} 1",
            "ferrolite_request_duration_seconds_bucket{route=\"/v1/chat/completions\",le=\"0.5\"} 2",
            "ferrolite_request_duration_seconds_bucket{route=\"/v1/chat/completions\",le=\"5\"} 3",
            "ferrolite_request_duration_seconds_count{route=\"/v1/chat/completions\"} 3",
            "ferrolite_tokens_total{model_group=\"gpt\",kind=\"prompt\"} 10",
            "ferrolite_tokens_total{model_group=\"a\\\"b\",kind=\"prompt\"} 0",
            "ferrolite_upstream_errors_total{model_group=\"gpt\",class=\"server\"} 1",
            "ferrolite_fallbacks_total{model_group=\"gpt\"} 1",
            "ferrolite_auth_failures_total 1",
            "ferrolite_in_flight_requests 3",
            "ferrolite_deployment_cooling_down{model_group=\"gpt\",deployment=\"gpt#0\"} 1",
        ] {
            assert!(text.contains(line), "missing {line}\n{text}");
        }
        assert!(!text.contains("not-configured") && !text.contains("made-up"));
        // Every non-comment line is `name{labels} value` or `name value`.
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let value = line.rsplit(' ').next().unwrap();
            assert!(value.parse::<f64>().is_ok(), "{line}");
        }
    }
}
