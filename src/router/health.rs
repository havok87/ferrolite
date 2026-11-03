//! Per-deployment runtime state: requests in flight, latency, rpm/tpm usage
//! and cooldowns.
//!
//! Usage and failures are counted in fixed one-minute windows, as LiteLLM
//! does. Locks are held only for a few field updates and never across an
//! `.await`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use tokio::time::Instant;

/// Weight of each new latency sample in the moving average.
const LATENCY_WEIGHT: f64 = 0.2;
const WINDOW_SECS: u64 = 60;

pub(crate) struct Health {
    epoch: Instant,
    deployments: Vec<DeploymentHealth>,
}

#[derive(Default)]
struct DeploymentHealth {
    in_flight: AtomicU64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    window: u64,
    requests: u64,
    tokens: u64,
    fail_window: u64,
    fails: u32,
    cooldown_until: Option<Instant>,
    /// Moving average, in seconds.
    latency: Option<f64>,
}

impl State {
    fn available(&mut self, now: Instant, window: u64, limits: Limits) -> bool {
        if let Some(until) = self.cooldown_until {
            if now < until {
                return false;
            }
            self.cooldown_until = None;
            self.fails = 0;
        }
        self.roll(window);
        limits.rpm.is_none_or(|rpm| self.requests < rpm)
            && limits.tpm.is_none_or(|tpm| self.tokens < tpm)
    }

    fn roll(&mut self, window: u64) {
        if self.window != window {
            self.window = window;
            self.requests = 0;
            self.tokens = 0;
        }
    }
}

/// What `rpm` and `tpm` allow a deployment.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub rpm: Option<u64>,
    pub tpm: Option<u64>,
}

impl Health {
    pub(crate) fn new(deployments: usize) -> Arc<Self> {
        Arc::new(Self {
            epoch: Instant::now(),
            deployments: (0..deployments)
                .map(|_| DeploymentHealth::default())
                .collect(),
        })
    }

    fn window(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.epoch).as_secs() / WINDOW_SECS
    }

    fn state(&self, index: usize) -> MutexGuard<'_, State> {
        self.deployments[index]
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Not cooling down and under its rpm/tpm limits for this minute.
    pub(crate) fn is_available(&self, index: usize, now: Instant, limits: Limits) -> bool {
        let window = self.window(now);
        self.state(index).available(now, window, limits)
    }

    /// Takes a request slot if the deployment is available. The check and
    /// the rpm count happen under one lock, so concurrent requests cannot
    /// both take the last slot of a minute.
    pub(crate) fn try_acquire(
        self: &Arc<Self>,
        index: usize,
        now: Instant,
        limits: Limits,
    ) -> Option<Lease> {
        let window = self.window(now);
        {
            let mut st = self.state(index);
            if !st.available(now, window, limits) {
                return None;
            }
            st.requests += 1;
        }
        self.deployments[index]
            .in_flight
            .fetch_add(1, Ordering::Relaxed);
        Some(Lease {
            health: Arc::clone(self),
            index,
        })
    }

    /// How long until one of `deployments` can take a request again: the
    /// end of its cooldown, or the next minute if it is over rpm/tpm.
    pub(crate) fn available_in(
        &self,
        deployments: impl IntoIterator<Item = (usize, Limits)>,
        now: Instant,
    ) -> Duration {
        let window = self.window(now);
        let next_window = self.epoch + Duration::from_secs((window + 1) * WINDOW_SECS);
        deployments
            .into_iter()
            .map(|(index, limits)| {
                let st = self.state(index);
                if let Some(until) = st.cooldown_until.filter(|u| now < *u) {
                    return until - now;
                }
                let (requests, tokens) = if st.window == window {
                    (st.requests, st.tokens)
                } else {
                    (0, 0)
                };
                let over = limits.rpm.is_some_and(|rpm| requests >= rpm)
                    || limits.tpm.is_some_and(|tpm| tokens >= tpm);
                if over {
                    next_window.saturating_duration_since(now)
                } else {
                    Duration::ZERO
                }
            })
            .min()
            .unwrap_or(Duration::ZERO)
    }

    /// Counts a request against the deployment and returns the lease that
    /// keeps it in flight, without checking limits.
    #[cfg(test)]
    pub(crate) fn acquire(self: &Arc<Self>, index: usize, now: Instant) -> Lease {
        let window = self.window(now);
        {
            let mut st = self.state(index);
            st.roll(window);
            st.requests += 1;
        }
        self.deployments[index]
            .in_flight
            .fetch_add(1, Ordering::Relaxed);
        Lease {
            health: Arc::clone(self),
            index,
        }
    }

    pub(crate) fn record_latency(&self, index: usize, sample: Duration) {
        let sample = sample.as_secs_f64();
        let mut st = self.state(index);
        st.latency = Some(match st.latency {
            Some(avg) => avg * (1.0 - LATENCY_WEIGHT) + sample * LATENCY_WEIGHT,
            None => sample,
        });
    }

    /// Counts a failure. Once there are more than `allowed_fails` in a
    /// minute the deployment cools down, unless `may_cool_down` is false.
    /// Returns whether a cooldown started.
    pub(crate) fn record_failure(
        &self,
        index: usize,
        now: Instant,
        allowed_fails: u32,
        cooldown: Duration,
        may_cool_down: bool,
    ) -> bool {
        let window = self.window(now);
        let mut st = self.state(index);
        if st.fail_window != window {
            st.fail_window = window;
            st.fails = 0;
        }
        st.fails = st.fails.saturating_add(1);
        if may_cool_down && st.fails > allowed_fails && !cooldown.is_zero() {
            st.cooldown_until = Some(now + cooldown);
            st.fails = 0;
            return true;
        }
        false
    }

    pub(crate) fn in_flight(&self, index: usize) -> u64 {
        self.deployments[index].in_flight.load(Ordering::Relaxed)
    }

    pub(crate) fn latency(&self, index: usize) -> Option<f64> {
        self.state(index).latency
    }

    pub(crate) fn is_cooling_down(&self, index: usize, now: Instant) -> bool {
        self.state(index)
            .cooldown_until
            .is_some_and(|until| now < until)
    }
}

/// Holds a deployment "in flight" until dropped. A streaming response keeps
/// its lease for as long as the stream runs.
pub struct Lease {
    health: Arc<Health>,
    index: usize,
}

impl Lease {
    /// Counts tokens against the deployment's `tpm` for the current minute.
    pub fn record_tokens(&self, tokens: u64) {
        let window = self.health.window(Instant::now());
        let mut st = self.health.state(self.index);
        st.roll(window);
        st.tokens = st.tokens.saturating_add(tokens);
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.health.deployments[self.index]
            .in_flight
            .fetch_sub(1, Ordering::Relaxed);
    }
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lease").field("index", &self.index).finish()
    }
}
