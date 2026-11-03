//! The HTTP server: state, routes and the accept loop.
//!
//! Connections are served with hyper directly rather than `axum::serve`
//! so that a header-read timeout applies (a client that never finishes its
//! headers cannot hold a connection), TLS handshakes run with their own
//! timeout off the accept path, the number of open connections is capped,
//! and shutdown drains connections gracefully.

mod auth;
mod errors;
mod health;
mod limits;
mod metrics;
mod routes;
pub mod tls;

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::service::TowerToHyperService;
use rand::rngs::{SmallRng, SysRng};
use rand::{RngExt, SeedableRng};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinSet;
use tokio_rustls::TlsAcceptor;

use crate::config::{Config, Loaded};
use crate::providers::{ClientBuildError, Upstream, UpstreamOptions};
use crate::router::Router;
use crate::secrets::{Redactor, Secret};
use health::HealthCache;
use limits::Concurrency;
use metrics::Metrics;

pub use errors::{ApiError, ErrorFormat, client_status};

/// Default request body limit (LiteLLM's `max_request_size_mb` overrides it).
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
pub const DEFAULT_MAX_CONCURRENCY: usize = 256;
pub const DEFAULT_MAX_CONNECTIONS: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct ServerOptions {
    pub max_request_bytes: usize,
    pub max_concurrency: usize,
    pub body_read_timeout: Duration,
    pub upstream: UpstreamOptions,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            body_read_timeout: Duration::from_secs(60),
            upstream: UpstreamOptions::default(),
        }
    }
}

/// Everything the handlers share.
pub struct AppState {
    config: Arc<Config>,
    master_key: Option<Secret>,
    router: Router,
    upstream: Upstream,
    scrub: Redactor,
    concurrency: Concurrency,
    metrics: Metrics,
    health: HealthCache,
    options: ServerOptions,
    ids: Mutex<SmallRng>,
    /// Unix time the server started, reported as models' `created`.
    started: u64,
}

impl AppState {
    pub fn new(loaded: Loaded, options: ServerOptions) -> Result<Arc<Self>, ClientBuildError> {
        let Loaded {
            config, redactor, ..
        } = loaded;
        let upstream = Upstream::new(&config, &redactor, options.upstream)?;
        let groups: Vec<String> = config
            .deployments
            .iter()
            .map(|d| d.model_name.clone())
            .collect();
        let ids = SmallRng::try_from_rng(&mut SysRng)
            .unwrap_or_else(|_| SmallRng::seed_from_u64(u64::from(std::process::id())));
        Ok(Arc::new(Self {
            master_key: config.master_key.clone(),
            router: Router::new(&config),
            upstream,
            scrub: redactor,
            concurrency: Concurrency::new(options.max_concurrency),
            metrics: Metrics::new(groups),
            health: HealthCache::default(),
            options,
            ids: Mutex::new(ids),
            started: crate::translate::unix_now(),
            config: Arc::new(config),
        }))
    }

    /// A random UUID v4 string.
    fn request_id(&self) -> String {
        let n: u128 = self
            .ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .random();
        let n = (n & !(0xf << 76) | (0x4 << 76)) & !(0x3 << 62) | (0x2 << 62);
        let h = format!("{n:032x}");
        format!(
            "{}-{}-{}-{}-{}",
            &h[..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..]
        )
    }
}

/// A JSON response that is never cached.
pub(crate) fn json_response(
    status: u16,
    body: impl Into<axum::body::Body>,
) -> axum::http::Response<axum::body::Body> {
    use axum::http::{HeaderValue, StatusCode, header};
    let mut resp = axum::http::Response::new(body.into());
    *resp.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let h = resp.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

/// The axum application for `state`.
pub fn app(state: Arc<AppState>) -> axum::Router {
    routes::router(state)
}

#[derive(Debug, Clone, Copy)]
pub struct ServeOptions {
    /// Time a client has to send a request's headers.
    pub header_read_timeout: Duration,
    pub tls_handshake_timeout: Duration,
    /// Connections beyond this are closed as soon as they are accepted.
    pub max_connections: usize,
    /// How long shutdown waits for open connections and streams.
    pub shutdown_grace: Duration,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            header_read_timeout: Duration::from_secs(30),
            tls_handshake_timeout: Duration::from_secs(10),
            max_connections: DEFAULT_MAX_CONNECTIONS,
            shutdown_grace: Duration::from_secs(30),
        }
    }
}

/// Serves `app` on `listener` until `shutdown` completes, then stops
/// accepting, lets open connections finish (up to `shutdown_grace`), and
/// returns.
pub async fn serve(
    listener: TcpListener,
    app: axum::Router,
    tls: Option<TlsAcceptor>,
    options: ServeOptions,
    shutdown: impl Future<Output = ()>,
) {
    let slots = Arc::new(Semaphore::new(options.max_connections));
    let (stop_tx, stop_rx) = watch::channel(false);
    let mut connections = JoinSet::new();
    let mut shutdown = std::pin::pin!(shutdown);

    loop {
        tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = match accepted {
                    Ok(c) => c,
                    Err(e) => {
                        // Usually out of file descriptors; back off briefly.
                        tracing::warn!(error = %e, "accept failed");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let Ok(slot) = Arc::clone(&slots).try_acquire_owned() else {
                    drop(stream);
                    continue;
                };
                let _ = stream.set_nodelay(true);
                let app = app.clone();
                let tls = tls.clone();
                let stop = stop_rx.clone();
                connections.spawn(async move {
                    let _slot = slot;
                    match tls {
                        None => serve_connection(stream, app, stop, options).await,
                        Some(acceptor) => {
                            let handshake = acceptor.accept(stream);
                            if let Ok(Ok(stream)) =
                                tokio::time::timeout(options.tls_handshake_timeout, handshake).await
                            {
                                serve_connection(stream, app, stop, options).await;
                            }
                        }
                    }
                });
            }
        }
        while connections.try_join_next().is_some() {}
    }

    let _ = stop_tx.send(true);
    let drained = tokio::time::timeout(options.shutdown_grace, async {
        while connections.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        tracing::warn!(
            open = connections.len(),
            "shutdown grace period over; closing remaining connections"
        );
        connections.shutdown().await;
    }
}

async fn serve_connection<I>(
    io: I,
    app: axum::Router,
    mut stop: watch::Receiver<bool>,
    options: ServeOptions,
) where
    I: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let service = TowerToHyperService::new(app);
    let mut builder = hyper::server::conn::http1::Builder::new();
    builder
        .timer(TokioTimer::new())
        .header_read_timeout(options.header_read_timeout);
    let conn = builder.serve_connection(TokioIo::new(io), service);
    let mut conn = std::pin::pin!(conn);
    tokio::select! {
        _ = conn.as_mut() => {}
        // The guard `wait_for` returns is dropped inside this block, so the
        // task stays Send.
        _ = async { stop.wait_for(|stopping| *stopping).await.is_ok() } => {
            // Finish the request in progress (including a stream), then close.
            conn.as_mut().graceful_shutdown();
            let _ = conn.await;
        }
    }
}
