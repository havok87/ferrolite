//! The router: resolves a requested model to a group, picks a deployment,
//! runs the upstream call with a timeout and retries, tracks deployment
//! health and applies fallbacks.
//!
//! The router makes no HTTP calls itself. [`Router::execute`] takes a
//! closure that performs one attempt against the deployment it is given,
//! so providers plug in without the router knowing about them.

mod health;
mod resolve;
mod retry;
mod strategy;
#[cfg(test)]
mod tests;

use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use rand::SeedableRng;
use rand::rngs::{SmallRng, SysRng};
use tokio::time::Instant;

use crate::config::{Config, Deployment, Fallback, RouterSettings};
use health::{Health, Limits};
use strategy::Candidate;

pub use health::Lease;
pub use resolve::Resolved;
pub use retry::{Classify, ErrorClass};

/// Per-attempt timeout when neither the deployment nor `router_settings` sets one.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
/// Most fallback groups tried for one request, as LiteLLM's `max_fallbacks`.
pub const MAX_FALLBACKS: usize = 5;

/// What the router needs to know about a request.
#[derive(Debug, Clone, Copy, Default)]
pub struct Hints {
    /// Use the deployment's `stream_timeout` if it has one.
    pub stream: bool,
}

/// One attempt against one deployment, handed to the call closure.
#[derive(Debug)]
pub struct Attempt {
    pub deployment: Arc<Deployment>,
    /// The model name to send upstream, with any wildcard filled in.
    pub upstream_model: String,
    /// The router enforces this; it is here so the call can pass it on.
    pub timeout: Duration,
    /// 1 for the first attempt of the request, counting across fallbacks.
    pub number: u32,
    /// Keep this alive for as long as the response is being read.
    pub lease: Lease,
}

/// A successful call and how it was routed.
#[derive(Debug)]
pub struct Routed<T> {
    pub value: T,
    pub deployment_id: String,
    pub model_group: String,
    /// Attempts made, counting across fallbacks.
    pub attempts: u32,
    /// The fallback name that succeeded, if the requested group did not.
    pub fallback: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError<E> {
    #[error("model `{model}` is not configured")]
    UnknownModel { model: String },
    #[error("no deployment of `{group}` is available (cooling down or rate limited)")]
    NoDeployment {
        group: String,
        /// When the first of the group's deployments frees up.
        retry_after: Duration,
    },
    #[error("deployment `{deployment}` timed out after {}s", after.as_secs_f64())]
    Timeout { deployment: String, after: Duration },
    #[error("deployment `{deployment}` failed: {error}")]
    Upstream { deployment: String, error: E },
}

impl<E: Classify> RouterError<E> {
    /// The class of the failure, if an upstream was reached.
    pub fn class(&self) -> Option<ErrorClass> {
        match self {
            Self::UnknownModel { .. } | Self::NoDeployment { .. } => None,
            Self::Timeout { .. } => Some(ErrorClass::Timeout),
            Self::Upstream { error, .. } => Some(error.classify()),
        }
    }
}

/// A deployment's current state, for health and metrics endpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentStatus {
    pub id: String,
    pub model_group: String,
    pub in_flight: u64,
    pub cooling_down: bool,
}

pub struct Router {
    deployments: Vec<Arc<Deployment>>,
    settings: RouterSettings,
    groups: resolve::Groups,
    health: Arc<Health>,
    rng: Mutex<SmallRng>,
}

impl Router {
    pub fn new(config: &Config) -> Self {
        // Load balancing needs no cryptographic randomness; if the OS source
        // fails, a clock-seeded generator still spreads load.
        let rng = SmallRng::try_from_rng(&mut SysRng).unwrap_or_else(|_| {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64);
            SmallRng::seed_from_u64(nanos ^ u64::from(std::process::id()))
        });
        Self::with_rng(config, rng)
    }

    /// A router with a fixed random seed, for reproducible tests.
    pub fn with_seed(config: &Config, seed: u64) -> Self {
        Self::with_rng(config, SmallRng::seed_from_u64(seed))
    }

    fn with_rng(config: &Config, rng: SmallRng) -> Self {
        Self {
            deployments: config.deployments.iter().cloned().map(Arc::new).collect(),
            settings: config.router.clone(),
            groups: resolve::Groups::new(config),
            health: Health::new(config.deployments.len()),
            rng: Mutex::new(rng),
        }
    }

    /// Every deployment's state right now, in config order.
    pub fn snapshot(&self) -> Vec<DeploymentStatus> {
        let now = Instant::now();
        self.deployments
            .iter()
            .enumerate()
            .map(|(i, d)| DeploymentStatus {
                id: d.id.clone(),
                model_group: d.model_name.clone(),
                in_flight: self.health.in_flight(i),
                cooling_down: self.health.is_cooling_down(i, now),
            })
            .collect()
    }

    pub fn resolve(&self, model: &str) -> Option<Resolved> {
        self.groups.resolve(model)
    }

    /// Runs `call` against a deployment of `model`'s group, retrying and
    /// falling back as configured.
    ///
    /// `call` performs one attempt. It should return as soon as the upstream
    /// has answered with a status (for a stream, before reading the body):
    /// the router never retries a call that returned `Ok`.
    pub async fn execute<T, E, F, Fut>(
        &self,
        model: &str,
        hints: Hints,
        mut call: F,
    ) -> Result<Routed<T>, RouterError<E>>
    where
        E: Classify,
        F: FnMut(Attempt) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let Some(resolved) = self.resolve(model) else {
            return Err(RouterError::UnknownModel {
                model: model.to_owned(),
            });
        };
        let mut attempts = 0;
        let mut last = match self
            .try_group(&resolved, hints, &mut call, &mut attempts)
            .await
        {
            Ok((value, index)) => {
                return Ok(self.routed(value, index, &resolved, attempts, None));
            }
            Err(e) => e,
        };

        for name in self.fallbacks_for(&resolved, &last) {
            let Some(target) = self.resolve(&name) else {
                continue;
            };
            if target.group == resolved.group {
                continue;
            }
            match self
                .try_group(&target, hints, &mut call, &mut attempts)
                .await
            {
                Ok((value, index)) => {
                    return Ok(self.routed(value, index, &target, attempts, Some(name)));
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn routed<T>(
        &self,
        value: T,
        index: usize,
        resolved: &Resolved,
        attempts: u32,
        fallback: Option<String>,
    ) -> Routed<T> {
        Routed {
            value,
            deployment_id: self.deployments[index].id.clone(),
            model_group: resolved.group.clone(),
            attempts,
            fallback,
        }
    }

    /// Tries one group: picks a deployment, calls it, and retries retryable
    /// failures up to `num_retries` times, preferring deployments that have
    /// not failed yet.
    async fn try_group<T, E, F, Fut>(
        &self,
        resolved: &Resolved,
        hints: Hints,
        call: &mut F,
        attempts: &mut u32,
    ) -> Result<(T, usize), RouterError<E>>
    where
        E: Classify,
        F: FnMut(Attempt) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let max_attempts = self.settings.num_retries.saturating_add(1);
        let may_cool_down = resolved.candidates.len() > 1;
        let mut failed: Vec<usize> = Vec::new();
        let mut last: Option<RouterError<E>> = None;

        for retry in 0..max_attempts {
            let started = Instant::now();
            let Some((index, lease)) = self.pick_and_acquire(resolved, &failed, started) else {
                return Err(last.unwrap_or_else(|| self.no_deployment(resolved, started)));
            };
            let deployment = Arc::clone(&self.deployments[index]);
            let timeout = self.timeout_for(&deployment, hints);
            *attempts += 1;
            let attempt = Attempt {
                upstream_model: resolve::upstream_model(
                    &deployment.upstream_model,
                    resolved.capture.as_deref(),
                ),
                deployment: Arc::clone(&deployment),
                timeout,
                number: *attempts,
                lease,
            };

            let (class, err) = match tokio::time::timeout(timeout, call(attempt)).await {
                Ok(Ok(value)) => {
                    self.health.record_latency(index, started.elapsed());
                    return Ok((value, index));
                }
                Ok(Err(error)) => (
                    error.classify(),
                    RouterError::Upstream {
                        deployment: deployment.id.clone(),
                        error,
                    },
                ),
                Err(_) => {
                    self.health.record_latency(index, timeout);
                    (
                        ErrorClass::Timeout,
                        RouterError::Timeout {
                            deployment: deployment.id.clone(),
                            after: timeout,
                        },
                    )
                }
            };

            if class.counts_as_failure() {
                self.health.record_failure(
                    index,
                    Instant::now(),
                    self.settings.allowed_fails,
                    self.settings.cooldown_time,
                    may_cool_down,
                );
            }
            if !class.is_retryable() || retry + 1 == max_attempts {
                return Err(err);
            }
            failed.push(index);
            last = Some(err);
            let wait = retry::delay(class, retry, &mut self.rng());
            tokio::time::sleep(wait).await;
        }
        Err(last.unwrap_or_else(|| self.no_deployment(resolved, Instant::now())))
    }

    fn no_deployment<E>(&self, resolved: &Resolved, now: Instant) -> RouterError<E> {
        let limits = resolved.candidates.iter().map(|&i| (i, self.limits(i)));
        RouterError::NoDeployment {
            group: resolved.group.clone(),
            retry_after: self.health.available_in(limits, now),
        }
    }

    fn limits(&self, index: usize) -> Limits {
        let d = &self.deployments[index];
        Limits {
            rpm: d.rpm,
            tpm: d.tpm,
        }
    }

    /// Picks a deployment and takes a request slot on it. If another request
    /// took the last slot between the pick and the reservation, picks again
    /// without that deployment.
    fn pick_and_acquire(
        &self,
        resolved: &Resolved,
        avoid: &[usize],
        now: Instant,
    ) -> Option<(usize, Lease)> {
        let mut lost: Vec<usize> = Vec::new();
        loop {
            let index = self.pick_excluding(resolved, avoid, &lost)?;
            match self.health.try_acquire(index, now, self.limits(index)) {
                Some(lease) => return Some((index, lease)),
                None => lost.push(index),
            }
        }
    }

    #[cfg(test)]
    fn pick(&self, resolved: &Resolved, avoid: &[usize]) -> Option<usize> {
        self.pick_excluding(resolved, avoid, &[])
    }

    /// Picks an eligible deployment of the group, never one in `excluded`:
    /// available, in the lowest `order` tier present, then chosen by the
    /// routing strategy. Deployments in `avoid` are used only if nothing
    /// else is eligible.
    fn pick_excluding(
        &self,
        resolved: &Resolved,
        avoid: &[usize],
        excluded: &[usize],
    ) -> Option<usize> {
        let now = Instant::now();
        let available: Vec<usize> = resolved
            .candidates
            .iter()
            .copied()
            .filter(|i| !excluded.contains(i))
            .filter(|&i| self.health.is_available(i, now, self.limits(i)))
            .collect();
        let fresh: Vec<usize> = available
            .iter()
            .copied()
            .filter(|i| !avoid.contains(i))
            .collect();
        let pool = if fresh.is_empty() { available } else { fresh };

        let tier = pool
            .iter()
            .map(|&i| order_key(&self.deployments[i]))
            .min()?;
        let candidates: Vec<Candidate> = pool
            .into_iter()
            .filter(|&i| order_key(&self.deployments[i]) == tier)
            .map(|i| {
                let d = &self.deployments[i];
                Candidate {
                    index: i,
                    weight: d.weight,
                    rpm: d.rpm,
                    tpm: d.tpm,
                    in_flight: self.health.in_flight(i),
                    latency: self.health.latency(i),
                }
            })
            .collect();
        strategy::choose(self.settings.routing_strategy, &candidates, &mut self.rng())
    }

    fn timeout_for(&self, deployment: &Deployment, hints: Hints) -> Duration {
        let own = if hints.stream {
            deployment.stream_timeout.or(deployment.timeout)
        } else {
            deployment.timeout
        };
        own.or(self.settings.timeout).unwrap_or(DEFAULT_TIMEOUT)
    }

    /// Fallback names for a failed group, in the order to try them.
    ///
    /// Context-window and content-policy errors try their own lists before
    /// the generic `fallbacks`. Bad requests never fall back: the same
    /// request would fail anywhere.
    fn fallbacks_for<E: Classify>(&self, resolved: &Resolved, err: &RouterError<E>) -> Vec<String> {
        let s = &self.settings;
        let lists: Vec<&[Fallback]> = match err.class() {
            Some(ErrorClass::BadRequest) => return Vec::new(),
            Some(ErrorClass::ContextWindowExceeded) => {
                vec![&s.context_window_fallbacks, &s.fallbacks]
            }
            Some(ErrorClass::ContentPolicy) => vec![&s.content_policy_fallbacks, &s.fallbacks],
            _ => vec![&s.fallbacks],
        };
        let mut out: Vec<String> = Vec::new();
        for list in lists {
            for name in fallback_targets(list, resolved) {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
        }
        out.truncate(MAX_FALLBACKS);
        out
    }

    fn rng(&self) -> MutexGuard<'_, SmallRng> {
        self.rng.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[cfg(test)]
    fn index_of(&self, id: &str) -> usize {
        self.deployments
            .iter()
            .position(|d| d.id == id)
            .unwrap_or_else(|| panic!("no deployment {id}"))
    }
}

/// Deployments with an `order` come first, lowest first.
fn order_key(d: &Deployment) -> (bool, u32) {
    (d.order.is_none(), d.order.unwrap_or(0))
}

/// The entry keyed by the requested name, else the one keyed by the group.
fn fallback_targets<'a>(list: &'a [Fallback], resolved: &Resolved) -> &'a [String] {
    list.iter()
        .find(|f| f.from == resolved.requested)
        .or_else(|| list.iter().find(|f| f.from == resolved.group))
        .map_or(&[], |f| f.to.as_slice())
}
