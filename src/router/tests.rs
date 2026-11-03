//! Router tests: fake upstream calls, a paused tokio clock and seeded randomness.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use super::strategy::{self, Candidate};
use super::*;
use crate::config::{self, LoadOptions, RoutingStrategy};
use crate::secrets::{KeychainRunner, Resolver, Secret};

struct NoKeychain;

impl KeychainRunner for NoKeychain {
    fn find_generic_password(&self, _: &str, _: &str) -> Result<Secret, String> {
        Err("no keychain in tests".to_owned())
    }
}

/// A config with one local deployment per `(model_name, extra params)`
/// entry. Extra params are `litellm_params` lines, unindented.
fn config(deployments: &[(&str, &str)], settings: &str) -> Config {
    let mut yaml = String::from("model_list:\n");
    for (name, extra) in deployments {
        let model = if name.ends_with('*') { "*" } else { "m" };
        yaml.push_str(&format!(
            "  - model_name: \"{name}\"\n    litellm_params:\n      model: hosted_vllm/{model}\n      api_base: http://127.0.0.1:8000/v1\n"
        ));
        for line in extra.lines().filter(|l| !l.trim().is_empty()) {
            yaml.push_str(&format!("      {}\n", line.trim()));
        }
    }
    yaml.push_str(settings);
    let env: HashMap<String, String> = HashMap::new();
    let resolver = Resolver::new(&env, &NoKeychain);
    let options = LoadOptions {
        bind: "127.0.0.1".parse().unwrap(),
        allow_no_auth: true,
    };
    match config::load_str(
        &yaml,
        Path::new("test.yaml"),
        Path::new("/"),
        &options,
        &resolver,
    ) {
        Ok(loaded) => loaded.config,
        Err(e) => panic!("{e}\n{yaml}"),
    }
}

fn router(deployments: &[(&str, &str)], settings: &str) -> Router {
    Router::with_seed(&config(deployments, settings), 7)
}

#[derive(Debug, Clone, PartialEq)]
struct Fail(ErrorClass);

impl Classify for Fail {
    fn classify(&self) -> ErrorClass {
        self.0
    }
}

impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Deployment ids in the order they were called.
#[derive(Clone, Default)]
struct Calls(Arc<Mutex<Vec<String>>>);

impl Calls {
    fn push(&self, id: &str) {
        self.0.lock().unwrap().push(id.to_owned());
    }

    fn get(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }

    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// A call closure that logs each attempt and answers with `outcome`.
fn call_with(
    calls: &Calls,
    outcome: impl Fn(&Attempt) -> Result<String, Fail>,
) -> impl FnMut(Attempt) -> std::future::Ready<Result<String, Fail>> {
    let calls = calls.clone();
    move |attempt| {
        calls.push(&attempt.deployment.id);
        std::future::ready(outcome(&attempt))
    }
}

fn ok(attempt: &Attempt) -> Result<String, Fail> {
    Ok(attempt.deployment.id.clone())
}

const SERVER: ErrorClass = ErrorClass::Server;

fn share(router: &Router, model: &str, picks: usize) -> HashMap<String, f64> {
    let resolved = router.resolve(model).unwrap();
    let mut counts: HashMap<String, f64> = HashMap::new();
    for _ in 0..picks {
        let i = router.pick(&resolved, &[]).unwrap();
        *counts.entry(router.deployments[i].id.clone()).or_default() += 1.0 / picks as f64;
    }
    counts
}

fn near(actual: Option<&f64>, expected: f64) -> bool {
    (actual.copied().unwrap_or(0.0) - expected).abs() < 0.03
}

// ---- resolution ----

#[tokio::test]
async fn resolves_exact_alias_and_wildcards() {
    let r = router(
        &[("gpt", ""), ("openai/*", ""), ("openai/o1-*", "")],
        "router_settings:\n  model_group_alias:\n    smart: gpt\n    any-o1: openai/o1-preview\n",
    );
    let exact = r.resolve("gpt").unwrap();
    assert_eq!((exact.group.as_str(), exact.capture), ("gpt", None));

    let alias = r.resolve("smart").unwrap();
    assert_eq!(
        (alias.requested.as_str(), alias.group.as_str()),
        ("smart", "gpt")
    );

    let wide = r.resolve("openai/gpt-4.1").unwrap();
    assert_eq!(wide.group, "openai/*");
    assert_eq!(wide.capture.as_deref(), Some("gpt-4.1"));

    let narrow = r.resolve("openai/o1-mini").unwrap();
    assert_eq!(narrow.group, "openai/o1-*");
    assert_eq!(narrow.capture.as_deref(), Some("mini"));

    let alias_to_wildcard = r.resolve("any-o1").unwrap();
    assert_eq!(alias_to_wildcard.group, "openai/o1-*");

    for missing in ["nope", "openai/", "openai/*", "openai/a*b"] {
        assert!(r.resolve(missing).is_none(), "{missing}");
    }
}

#[test]
fn fills_wildcard_into_upstream_model() {
    assert_eq!(resolve::upstream_model("*", Some("gpt-4.1")), "gpt-4.1");
    assert_eq!(resolve::upstream_model("o1-*", Some("mini")), "o1-mini");
    assert_eq!(resolve::upstream_model("fixed", Some("ignored")), "fixed");
    assert_eq!(resolve::upstream_model("*", None), "*");
}

#[tokio::test]
async fn execute_sends_the_wildcard_capture_upstream() {
    let r = router(&[("openai/*", "")], "");
    let routed = r
        .execute("openai/gpt-4.1", Hints::default(), |a: Attempt| {
            std::future::ready(Ok::<_, Fail>(a.upstream_model))
        })
        .await
        .unwrap();
    assert_eq!(routed.value, "gpt-4.1");
    assert_eq!(routed.model_group, "openai/*");
}

#[tokio::test]
async fn unknown_model_is_an_error() {
    let r = router(&[("gpt", "")], "");
    let calls = Calls::default();
    let err = r
        .execute("nope", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap_err();
    assert!(matches!(err, RouterError::UnknownModel { model } if model == "nope"));
    assert_eq!(calls.len(), 0);
}

// ---- strategies ----

#[tokio::test]
async fn simple_shuffle_weights() {
    let by_weight = router(&[("m", "weight: 3"), ("m", "weight: 1")], "");
    let s = share(&by_weight, "m", 20_000);
    assert!(
        near(s.get("m#0"), 0.75) && near(s.get("m#1"), 0.25),
        "{s:?}"
    );

    let missing_weight = router(&[("m", "weight: 2"), ("m", "")], "");
    let s = share(&missing_weight, "m", 2_000);
    assert!(near(s.get("m#0"), 1.0), "{s:?}");

    let by_rpm = router(&[("m", "rpm: 100"), ("m", "rpm: 300")], "");
    let s = share(&by_rpm, "m", 20_000);
    assert!(
        near(s.get("m#0"), 0.25) && near(s.get("m#1"), 0.75),
        "{s:?}"
    );

    let by_tpm = router(&[("m", "tpm: 1000"), ("m", "tpm: 1000")], "");
    let s = share(&by_tpm, "m", 20_000);
    assert!(near(s.get("m#0"), 0.5), "{s:?}");

    let uniform = router(&[("m", ""), ("m", ""), ("m", "")], "");
    let s = share(&uniform, "m", 30_000);
    for id in ["m#0", "m#1", "m#2"] {
        assert!(near(s.get(id), 1.0 / 3.0), "{s:?}");
    }
}

fn candidate(index: usize, in_flight: u64, latency: Option<f64>) -> Candidate {
    Candidate {
        index,
        weight: None,
        rpm: None,
        tpm: None,
        in_flight,
        latency,
    }
}

#[test]
fn least_busy_and_latency_choices() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let busy = [
        candidate(0, 3, None),
        candidate(1, 1, None),
        candidate(2, 2, None),
    ];
    for _ in 0..50 {
        assert_eq!(
            strategy::choose(RoutingStrategy::LeastBusy, &busy, &mut rng),
            Some(1)
        );
    }

    let unsampled_first = [
        candidate(0, 0, Some(0.5)),
        candidate(1, 0, None),
        candidate(2, 0, Some(0.1)),
    ];
    for _ in 0..50 {
        assert_eq!(
            strategy::choose(RoutingStrategy::LatencyBased, &unsampled_first, &mut rng),
            Some(1)
        );
    }
    let sampled = [candidate(0, 0, Some(0.5)), candidate(2, 0, Some(0.1))];
    assert_eq!(
        strategy::choose(RoutingStrategy::LatencyBased, &sampled, &mut rng),
        Some(2)
    );
    assert_eq!(
        strategy::choose(RoutingStrategy::SimpleShuffle, &[], &mut rng),
        None
    );
}

#[tokio::test]
async fn least_busy_avoids_held_leases() {
    let r = router(
        &[("m", ""), ("m", "")],
        "router_settings:\n  routing_strategy: least-busy\n",
    );
    let resolved = r.resolve("m").unwrap();
    let busy = r.index_of("m#0");
    let _lease = r.health.acquire(busy, Instant::now());
    for _ in 0..20 {
        assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#1")));
    }
}

#[tokio::test]
async fn latency_routing_prefers_the_fastest() {
    let r = router(
        &[("m", ""), ("m", "")],
        "router_settings:\n  routing_strategy: latency-based-routing\n",
    );
    let resolved = r.resolve("m").unwrap();
    r.health
        .record_latency(r.index_of("m#0"), Duration::from_secs(2));
    // m#1 has no samples yet, so it is explored first.
    assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#1")));
    r.health
        .record_latency(r.index_of("m#1"), Duration::from_millis(100));
    for _ in 0..20 {
        assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#1")));
    }
}

#[tokio::test]
async fn order_tiers_are_tried_lowest_first() {
    let r = router(&[("m", "order: 2"), ("m", "order: 1"), ("m", "")], "");
    let resolved = r.resolve("m").unwrap();
    for _ in 0..20 {
        assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#1")));
    }
    let now = Instant::now();
    r.health
        .record_failure(r.index_of("m#1"), now, 0, Duration::from_secs(30), true);
    assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#0")));
    r.health
        .record_failure(r.index_of("m#0"), now, 0, Duration::from_secs(30), true);
    assert_eq!(r.pick(&resolved, &[]), Some(r.index_of("m#2")));
}

// ---- rpm / tpm ----

#[tokio::test(start_paused = true)]
async fn rpm_limit_lifts_when_the_minute_rolls_over() {
    let r = router(&[("m", "rpm: 2"), ("m", "")], "");
    let calls = Calls::default();
    for _ in 0..3 {
        r.execute("m", Hints::default(), call_with(&calls, ok))
            .await
            .unwrap();
    }
    assert_eq!(calls.get(), ["m#0", "m#0", "m#1"]);
    tokio::time::advance(Duration::from_secs(60)).await;
    r.execute("m", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap();
    assert_eq!(calls.get()[3], "m#0");
}

#[tokio::test(start_paused = true)]
async fn tpm_limit_uses_recorded_tokens() {
    let r = router(&[("m", "tpm: 100"), ("m", "")], "");
    let calls = Calls::default();
    let spend = |a: &Attempt| {
        a.lease.record_tokens(150);
        ok(a)
    };
    for _ in 0..2 {
        r.execute("m", Hints::default(), call_with(&calls, spend))
            .await
            .unwrap();
    }
    assert_eq!(calls.get(), ["m#0", "m#1"]);
}

// ---- cooldown ----

#[tokio::test(start_paused = true)]
async fn cooldown_after_allowed_fails_and_recovery() {
    let r = router(
        &[("m", "order: 1"), ("m", "order: 2")],
        "router_settings:\n  allowed_fails: 1\n  cooldown_time: 30\n",
    );
    let calls = Calls::default();
    let first_fails = |a: &Attempt| {
        if a.deployment.id == "m#0" {
            Err(Fail(SERVER))
        } else {
            ok(a)
        }
    };
    for _ in 0..2 {
        let err = r
            .execute("m", Hints::default(), call_with(&calls, first_fails))
            .await
            .unwrap_err();
        assert!(matches!(err, RouterError::Upstream { .. }));
    }
    let m0 = r.index_of("m#0");
    assert!(r.health.is_cooling_down(m0, Instant::now()));
    let routed = r
        .execute("m", Hints::default(), call_with(&calls, first_fails))
        .await
        .unwrap();
    assert_eq!(routed.deployment_id, "m#1");

    tokio::time::advance(Duration::from_secs(31)).await;
    assert!(r.pick(&r.resolve("m").unwrap(), &[]) == Some(m0));
}

#[tokio::test(start_paused = true)]
async fn single_deployment_groups_never_cool_down() {
    let r = router(&[("solo", "")], "router_settings:\n  allowed_fails: 0\n");
    let calls = Calls::default();
    for _ in 0..5 {
        let err = r
            .execute(
                "solo",
                Hints::default(),
                call_with(&calls, |_| Err(Fail(SERVER))),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, RouterError::Upstream { .. }), "{err:?}");
    }
    assert_eq!(calls.len(), 5);
}

#[tokio::test(start_paused = true)]
async fn bad_requests_do_not_count_as_failures() {
    let r = router(
        &[("m", "order: 1"), ("m", "order: 2")],
        "router_settings:\n  allowed_fails: 0\n",
    );
    let calls = Calls::default();
    for _ in 0..3 {
        let _ = r
            .execute(
                "m",
                Hints::default(),
                call_with(&calls, |_| Err(Fail(ErrorClass::BadRequest))),
            )
            .await;
    }
    assert_eq!(calls.get(), ["m#0", "m#0", "m#0"]);
}

#[test]
fn failure_classes() {
    use ErrorClass::*;
    for c in [
        RateLimited { retry_after: None },
        Server,
        Timeout,
        Connection,
    ] {
        assert!(c.is_retryable() && c.counts_as_failure(), "{c:?}");
    }
    for c in [Auth, NotFound] {
        assert!(!c.is_retryable() && c.counts_as_failure(), "{c:?}");
    }
    for c in [ContextWindowExceeded, ContentPolicy, BadRequest] {
        assert!(!c.is_retryable() && !c.counts_as_failure(), "{c:?}");
    }
}

// ---- retries ----

/// A call that fails with `class` for the first `failures` attempts, logging
/// when each attempt started.
fn flaky(
    class: ErrorClass,
    failures: usize,
    starts: &Arc<Mutex<Vec<Instant>>>,
) -> impl FnMut(Attempt) -> std::future::Ready<Result<String, Fail>> {
    let starts = Arc::clone(starts);
    move |a| {
        let mut s = starts.lock().unwrap();
        s.push(Instant::now());
        std::future::ready(if s.len() <= failures {
            Err(Fail(class))
        } else {
            Ok(a.deployment.id.clone())
        })
    }
}

fn gaps(starts: &Arc<Mutex<Vec<Instant>>>) -> Vec<Duration> {
    let s = starts.lock().unwrap();
    s.windows(2).map(|w| w[1] - w[0]).collect()
}

#[tokio::test(start_paused = true)]
async fn retries_with_exponential_backoff() {
    let r = router(&[("solo", "")], "router_settings:\n  num_retries: 3\n");
    let starts = Arc::new(Mutex::new(Vec::new()));
    let routed = r
        .execute("solo", Hints::default(), flaky(SERVER, 3, &starts))
        .await
        .unwrap();
    assert_eq!(routed.attempts, 4);
    let g = gaps(&starts);
    let within = |d: Duration, lo: u64, hi: u64| {
        d >= Duration::from_millis(lo) && d <= Duration::from_millis(hi)
    };
    assert!(within(g[0], 250, 500), "{g:?}");
    assert!(within(g[1], 500, 1000), "{g:?}");
    assert!(within(g[2], 1000, 2000), "{g:?}");
}

#[tokio::test(start_paused = true)]
async fn gives_up_after_num_retries() {
    let r = router(&[("solo", "")], "router_settings:\n  num_retries: 2\n");
    let starts = Arc::new(Mutex::new(Vec::new()));
    let err = r
        .execute("solo", Hints::default(), flaky(SERVER, 10, &starts))
        .await
        .unwrap_err();
    assert!(matches!(err, RouterError::Upstream { .. }));
    assert_eq!(starts.lock().unwrap().len(), 3);
}

#[tokio::test(start_paused = true)]
async fn honours_and_caps_retry_after() {
    let r = router(&[("solo", "")], "router_settings:\n  num_retries: 1\n");
    for (after, expected) in [(2, 2), (120, 30)] {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let class = ErrorClass::RateLimited {
            retry_after: Some(Duration::from_secs(after)),
        };
        r.execute("solo", Hints::default(), flaky(class, 1, &starts))
            .await
            .unwrap();
        assert_eq!(gaps(&starts), [Duration::from_secs(expected)]);
    }
}

#[tokio::test(start_paused = true)]
async fn does_not_retry_non_retryable_errors() {
    let r = router(&[("solo", "")], "router_settings:\n  num_retries: 3\n");
    for class in [
        ErrorClass::Auth,
        ErrorClass::NotFound,
        ErrorClass::BadRequest,
        ErrorClass::ContextWindowExceeded,
        ErrorClass::ContentPolicy,
    ] {
        let starts = Arc::new(Mutex::new(Vec::new()));
        let err = r
            .execute("solo", Hints::default(), flaky(class, 10, &starts))
            .await
            .unwrap_err();
        assert_eq!(err.class(), Some(class));
        assert_eq!(starts.lock().unwrap().len(), 1, "{class:?}");
    }
}

#[tokio::test(start_paused = true)]
async fn retries_prefer_a_different_deployment() {
    for seed in 0..20 {
        let r = Router::with_seed(
            &config(
                &[("m", ""), ("m", ""), ("m", "")],
                "router_settings:\n  num_retries: 2\n  allowed_fails: 100\n",
            ),
            seed,
        );
        let calls = Calls::default();
        let fail_first_two = {
            let calls = calls.clone();
            move |a: &Attempt| {
                if calls.len() <= 2 {
                    Err(Fail(SERVER))
                } else {
                    ok(a)
                }
            }
        };
        r.execute("m", Hints::default(), call_with(&calls, fail_first_two))
            .await
            .unwrap();
        let mut seen = calls.get();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 3, "seed {seed}: {:?}", calls.get());
    }
}

// ---- timeouts ----

#[tokio::test(start_paused = true)]
async fn slow_calls_time_out_and_are_retried() {
    let r = router(
        &[("solo", "timeout: 1")],
        "router_settings:\n  num_retries: 1\n",
    );
    let started = Instant::now();
    let err = r
        .execute("solo", Hints::default(), |_a: Attempt| async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok::<String, Fail>("too late".into())
        })
        .await
        .unwrap_err();
    assert!(matches!(err, RouterError::Timeout { after, .. } if after == Duration::from_secs(1)));
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(2250) && elapsed <= Duration::from_millis(2500),
        "{elapsed:?}"
    );
    // The lease was dropped with the cancelled call.
    assert_eq!(r.health.in_flight(0), 0);
}

#[tokio::test]
async fn timeout_precedence() {
    let r = router(
        &[
            ("both", "timeout: 5\nstream_timeout: 1"),
            ("plain", "timeout: 5"),
            ("none", ""),
        ],
        "router_settings:\n  timeout: 42\n",
    );
    let timeout_of = |model: &str, stream: bool| {
        let d = &r.deployments[r.index_of(&format!("{model}#0"))];
        r.timeout_for(d, Hints { stream })
    };
    assert_eq!(timeout_of("both", true), Duration::from_secs(1));
    assert_eq!(timeout_of("both", false), Duration::from_secs(5));
    assert_eq!(timeout_of("plain", true), Duration::from_secs(5));
    assert_eq!(timeout_of("none", false), Duration::from_secs(42));

    let bare = router(&[("none", "")], "");
    assert_eq!(
        bare.timeout_for(&bare.deployments[0], Hints::default()),
        DEFAULT_TIMEOUT
    );
}

// ---- fallbacks ----

const FALLBACK_GROUPS: &[(&str, &str)] = &[
    ("primary", "rpm: 1"),
    ("backup", ""),
    ("cw", ""),
    ("cp", ""),
    ("other", ""),
    ("aliased", ""),
];

const FALLBACK_SETTINGS: &str = "router_settings:
  model_group_alias:
    smart: backup
  fallbacks:
    - primary: [backup]
    - backup: [other]
    - aliased: [smart]
  context_window_fallbacks:
    - primary: [cw]
  content_policy_fallbacks:
    - primary: [cp]
";

/// Fails every group listed in `failing` with `class`.
fn failing_groups(
    calls: &Calls,
    class: ErrorClass,
    failing: &'static [&'static str],
) -> impl FnMut(Attempt) -> std::future::Ready<Result<String, Fail>> {
    call_with(calls, move |a| {
        if failing.contains(&a.deployment.model_name.as_str()) {
            Err(Fail(class))
        } else {
            ok(a)
        }
    })
}

#[tokio::test(start_paused = true)]
async fn falls_back_by_error_class() {
    for (class, expected) in [
        (SERVER, "backup"),
        (ErrorClass::ContextWindowExceeded, "cw"),
        (ErrorClass::ContentPolicy, "cp"),
        (ErrorClass::Auth, "backup"),
    ] {
        let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
        let calls = Calls::default();
        let routed = r
            .execute(
                "primary",
                Hints::default(),
                failing_groups(&calls, class, &["primary"]),
            )
            .await
            .unwrap();
        assert_eq!(routed.model_group, expected, "{class:?}");
        assert_eq!(routed.fallback.as_deref(), Some(expected));
        assert_eq!(routed.attempts, 2);
        assert_eq!(
            calls.get(),
            ["primary#0".to_owned(), format!("{expected}#0")]
        );
    }
}

#[tokio::test(start_paused = true)]
async fn specific_fallbacks_then_generic() {
    let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
    let calls = Calls::default();
    let routed = r
        .execute(
            "primary",
            Hints::default(),
            failing_groups(
                &calls,
                ErrorClass::ContextWindowExceeded,
                &["primary", "cw"],
            ),
        )
        .await
        .unwrap();
    assert_eq!(routed.model_group, "backup");
    assert_eq!(calls.get(), ["primary#0", "cw#0", "backup#0"]);
}

#[tokio::test(start_paused = true)]
async fn bad_requests_never_fall_back() {
    let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
    let calls = Calls::default();
    let err = r
        .execute(
            "primary",
            Hints::default(),
            failing_groups(&calls, ErrorClass::BadRequest, &["primary"]),
        )
        .await
        .unwrap_err();
    assert_eq!(err.class(), Some(ErrorClass::BadRequest));
    assert_eq!(calls.get(), ["primary#0"]);
}

#[tokio::test(start_paused = true)]
async fn fallbacks_do_not_chain() {
    let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
    let calls = Calls::default();
    let err = r
        .execute(
            "primary",
            Hints::default(),
            failing_groups(&calls, SERVER, &["primary", "backup"]),
        )
        .await
        .unwrap_err();
    assert!(matches!(&err, RouterError::Upstream { deployment, .. } if deployment == "backup#0"));
    // backup's own fallback (`other`) is not used for a request to primary.
    assert_eq!(calls.get(), ["primary#0", "backup#0"]);
}

#[tokio::test(start_paused = true)]
async fn falls_back_through_an_alias() {
    let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
    let calls = Calls::default();
    let routed = r
        .execute(
            "aliased",
            Hints::default(),
            failing_groups(&calls, SERVER, &["aliased"]),
        )
        .await
        .unwrap();
    assert_eq!(routed.model_group, "backup");
    assert_eq!(routed.fallback.as_deref(), Some("smart"));
}

#[tokio::test(start_paused = true)]
async fn falls_back_when_nothing_is_available() {
    let r = router(FALLBACK_GROUPS, FALLBACK_SETTINGS);
    let calls = Calls::default();
    let first = r
        .execute("primary", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap();
    assert_eq!(first.model_group, "primary");
    // primary's rpm of 1 is used up for this minute.
    let second = r
        .execute("primary", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap();
    assert_eq!(second.model_group, "backup");
    assert_eq!(second.attempts, 1);
}

#[tokio::test(start_paused = true)]
async fn reports_no_deployment_without_fallbacks() {
    let r = router(&[("solo", "rpm: 1")], "");
    let calls = Calls::default();
    r.execute("solo", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap();
    let err = r
        .execute("solo", Hints::default(), call_with(&calls, ok))
        .await
        .unwrap_err();
    assert!(matches!(err, RouterError::NoDeployment { group } if group == "solo"));
}

// ---- leases ----

#[tokio::test]
async fn lease_counts_in_flight_until_dropped() {
    let r = router(&[("solo", "")], "");
    let seen = Arc::new(Mutex::new(None));
    let kept = r
        .execute("solo", Hints::default(), |a: Attempt| {
            *seen.lock().unwrap() = Some(r.health.in_flight(0));
            // Return the lease with the value, as a streaming response would.
            std::future::ready(Ok::<_, Fail>(a.lease))
        })
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), Some(1));
    assert_eq!(r.health.in_flight(0), 1);
    drop(kept);
    assert_eq!(r.health.in_flight(0), 0);
}

#[tokio::test(start_paused = true)]
async fn attempt_numbers_count_across_fallbacks() {
    let r = router(
        &[("primary", ""), ("backup", "")],
        "router_settings:\n  num_retries: 1\n  fallbacks: [{primary: [backup]}]\n",
    );
    let numbers = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&numbers);
    r.execute("primary", Hints::default(), move |a: Attempt| {
        log.lock().unwrap().push(a.number);
        std::future::ready(if a.deployment.model_name == "primary" {
            Err(Fail(SERVER))
        } else {
            Ok(())
        })
    })
    .await
    .unwrap();
    assert_eq!(*numbers.lock().unwrap(), [1, 2, 3]);
}
