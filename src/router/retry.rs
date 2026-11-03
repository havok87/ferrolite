//! Error classes and retry backoff.

use std::time::Duration;

use rand::RngExt;
use rand::rngs::SmallRng;

const BACKOFF_BASE: Duration = Duration::from_millis(500);
const BACKOFF_CAP: Duration = Duration::from_secs(8);
const RETRY_AFTER_CAP: Duration = Duration::from_secs(30);

/// How an upstream call failed, as far as routing is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// 429. `retry_after` comes from the upstream's `Retry-After` header.
    RateLimited {
        retry_after: Option<Duration>,
    },
    /// 5xx.
    Server,
    Timeout,
    /// Could not connect, or the connection broke before a response.
    Connection,
    ContextWindowExceeded,
    ContentPolicy,
    /// 401 or 403: the deployment's credentials are wrong.
    Auth,
    /// 404: the deployment's model or URL is wrong.
    NotFound,
    /// Any other 4xx: the request itself is at fault.
    BadRequest,
}

impl ErrorClass {
    /// Worth another attempt, possibly on another deployment.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. } | Self::Server | Self::Timeout | Self::Connection
        )
    }

    /// Says something about the deployment's health, so it counts towards
    /// `allowed_fails`.
    pub fn counts_as_failure(self) -> bool {
        self.is_retryable() || matches!(self, Self::Auth | Self::NotFound)
    }
}

/// Implemented by the upstream error type so the router can classify it.
pub trait Classify {
    fn classify(&self) -> ErrorClass;
}

/// How long to wait before retry number `retry` (0-based). Honours a
/// capped `Retry-After`; otherwise exponential backoff with jitter.
pub(crate) fn delay(class: ErrorClass, retry: u32, rng: &mut SmallRng) -> Duration {
    if let ErrorClass::RateLimited {
        retry_after: Some(after),
    } = class
    {
        return after.min(RETRY_AFTER_CAP);
    }
    let base = BACKOFF_BASE
        .saturating_mul(2u32.saturating_pow(retry))
        .min(BACKOFF_CAP);
    // Between half and all of `base`, so clients retrying together spread out.
    base / 2 + base.mul_f64(rng.random::<f64>() / 2.0)
}
