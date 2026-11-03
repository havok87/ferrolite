//! Choosing one deployment from the eligible ones.

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::config::RoutingStrategy;

/// What a strategy needs to know about an eligible deployment.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub index: usize,
    pub weight: Option<f64>,
    pub rpm: Option<u64>,
    pub tpm: Option<u64>,
    pub in_flight: u64,
    /// Moving average in seconds; `None` until the first sample.
    pub latency: Option<f64>,
}

/// Returns the chosen candidate's deployment index.
pub(crate) fn choose(
    strategy: RoutingStrategy,
    candidates: &[Candidate],
    rng: &mut SmallRng,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let chosen = match strategy {
        RoutingStrategy::SimpleShuffle => simple_shuffle(candidates, rng),
        RoutingStrategy::LeastBusy => least_by(candidates, rng, |c| c.in_flight as f64),
        RoutingStrategy::LatencyBased => {
            let unsampled: Vec<&Candidate> =
                candidates.iter().filter(|c| c.latency.is_none()).collect();
            if unsampled.is_empty() {
                least_by(candidates, rng, |c| c.latency.unwrap_or(f64::MAX))
            } else {
                unsampled[rng.random_range(0..unsampled.len())]
            }
        }
    };
    Some(chosen.index)
}

/// Weighted by `weight` if any candidate sets it (missing counts as 0, as in
/// LiteLLM), else by `rpm`, else by `tpm`, else uniform.
fn simple_shuffle<'c>(candidates: &'c [Candidate], rng: &mut SmallRng) -> &'c Candidate {
    let weights: Vec<f64> = if candidates.iter().any(|c| c.weight.is_some()) {
        candidates.iter().map(|c| c.weight.unwrap_or(0.0)).collect()
    } else if candidates.iter().any(|c| c.rpm.is_some()) {
        candidates
            .iter()
            .map(|c| c.rpm.unwrap_or(0) as f64)
            .collect()
    } else if candidates.iter().any(|c| c.tpm.is_some()) {
        candidates
            .iter()
            .map(|c| c.tpm.unwrap_or(0) as f64)
            .collect()
    } else {
        vec![1.0; candidates.len()]
    };
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return &candidates[rng.random_range(0..candidates.len())];
    }
    let mut point = rng.random::<f64>() * total;
    for (candidate, weight) in candidates.iter().zip(&weights) {
        if point < *weight {
            return candidate;
        }
        point -= weight;
    }
    // Rounding can leave `point` just past the end; take the last weighted one.
    candidates
        .iter()
        .zip(&weights)
        .rev()
        .find(|(_, w)| **w > 0.0)
        .map_or(&candidates[0], |(c, _)| c)
}

/// The candidate with the smallest `key`; ties are broken at random.
fn least_by<'c>(
    candidates: &'c [Candidate],
    rng: &mut SmallRng,
    key: impl Fn(&Candidate) -> f64,
) -> &'c Candidate {
    let best = candidates.iter().map(&key).fold(f64::INFINITY, f64::min);
    let ties: Vec<&Candidate> = candidates.iter().filter(|c| key(c) <= best).collect();
    ties[rng.random_range(0..ties.len())]
}
