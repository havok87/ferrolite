//! Request body limits and the global concurrency limit.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyError {
    TooLarge,
    Timeout,
    Read,
}

/// Reads the whole request body, failing fast once it passes `max` bytes
/// or takes longer than `timeout` (a client trickling a body cannot hold a
/// connection and a permit indefinitely).
pub async fn read_body(body: Body, max: usize, timeout: Duration) -> Result<Bytes, BodyError> {
    let read = async {
        let mut stream = body.into_data_stream();
        let mut buf = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| BodyError::Read)?;
            if buf.len() + chunk.len() > max {
                return Err(BodyError::TooLarge);
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf.freeze())
    };
    tokio::time::timeout(timeout, read)
        .await
        .unwrap_or(Err(BodyError::Timeout))
}

/// At most `max` proxied requests at once, streams included. Excess
/// requests are refused at once rather than queued.
#[derive(Debug, Clone)]
pub struct Concurrency {
    permits: Arc<Semaphore>,
    max: usize,
}

impl Concurrency {
    pub fn new(max: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max)),
            max,
        }
    }

    pub fn try_acquire(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.permits).try_acquire_owned().ok()
    }

    pub fn in_flight(&self) -> usize {
        self.max.saturating_sub(self.permits.available_permits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn body_limits() {
        let ok = read_body(Body::from("hello"), 5, Duration::from_secs(1)).await;
        assert_eq!(ok.unwrap(), "hello");
        let big = read_body(Body::from("hello!"), 5, Duration::from_secs(1)).await;
        assert_eq!(big, Err(BodyError::TooLarge));
    }

    #[tokio::test(start_paused = true)]
    async fn slow_bodies_time_out() {
        let slow = futures_util::stream::once(async {
            tokio::time::sleep(Duration::from_secs(120)).await;
            Ok::<_, std::io::Error>(Bytes::from("late"))
        });
        let got = read_body(Body::from_stream(slow), 100, Duration::from_secs(60)).await;
        assert_eq!(got, Err(BodyError::Timeout));
    }

    #[test]
    fn concurrency_permits() {
        let c = Concurrency::new(2);
        let a = c.try_acquire().unwrap();
        let _b = c.try_acquire().unwrap();
        assert!(c.try_acquire().is_none());
        assert_eq!(c.in_flight(), 2);
        drop(a);
        assert!(c.try_acquire().is_some());
    }
}
