//! Test harness: mock upstreams (axum on ephemeral ports) and a real
//! ferrolite server in front of them.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, Method, Response, Uri};
use bytes::Bytes;
use ferrolite::config::{self, LoadOptions, Loaded};
use ferrolite::secrets::{KeychainRunner, Resolver, Secret};
use ferrolite::server::{self, AppState, ServeOptions, ServerOptions};
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio_rustls::TlsAcceptor;

pub const MASTER_KEY: &str = "sk-integration-master-0123456789abcd";
pub const UPSTREAM_KEY: &str = "sk-upstream-secret-000001";
pub const ANTHROPIC_KEY: &str = "sk-ant-upstream-secret-01";

// ---- mock upstreams ----

#[derive(Debug, Clone)]
pub struct Seen {
    pub method: Method,
    pub path: String,
    pub headers: HeaderMap,
    pub body: Value,
}

pub enum Reply {
    Json(u16, Value),
    /// A JSON answer sent only after a delay.
    JsonAfter(Duration, u16, Value),
    /// SSE chunks, each sent after `delay`; `stall` keeps the stream open after the last.
    Sse {
        chunks: Vec<String>,
        delay: Duration,
        stall: bool,
    },
}

pub struct Mock {
    pub port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Mock {
    pub async fn start(reply: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Self {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let reply = Arc::new(reply);
        let log = Arc::clone(&seen);
        let app = axum::Router::new().fallback(
            move |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| {
                let log = Arc::clone(&log);
                let reply = Arc::clone(&reply);
                async move {
                    let req = Seen {
                        method,
                        path: uri.path().to_owned(),
                        headers,
                        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
                    };
                    let r = reply(&req);
                    log.lock().unwrap().push(req);
                    respond(r).await
                }
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { port, seen }
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    /// Requests whose body named `upstream_model`.
    pub fn calls_for(&self, upstream_model: &str) -> Vec<Seen> {
        self.seen()
            .into_iter()
            .filter(|s| s.body["model"] == upstream_model)
            .collect()
    }
}

async fn respond(reply: Reply) -> Response<Body> {
    match reply {
        Reply::JsonAfter(delay, status, v) => {
            tokio::time::sleep(delay).await;
            Box::pin(respond(Reply::Json(status, v))).await
        }
        Reply::Json(status, v) => Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        Reply::Sse {
            chunks,
            delay,
            stall,
        } => {
            let items = futures_util::stream::iter(chunks).then(move |c| async move {
                tokio::time::sleep(delay).await;
                Ok::<_, std::io::Error>(Bytes::from(c))
            });
            let body = if stall {
                Body::from_stream(items.chain(futures_util::stream::pending()))
            } else {
                Body::from_stream(items)
            };
            Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(body)
                .unwrap()
        }
    }
}

// ---- the gateway ----

struct NoKeychain;

impl KeychainRunner for NoKeychain {
    fn find_generic_password(&self, _: &str, _: &str) -> Result<Secret, String> {
        Err("no keychain in tests".to_owned())
    }
}

pub fn load(yaml: &str, allow_no_auth: bool) -> Loaded {
    let env: HashMap<String, String> = [
        ("MASTER", MASTER_KEY),
        ("UP_KEY", UPSTREAM_KEY),
        ("ANT_KEY", ANTHROPIC_KEY),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let resolver = Resolver::new(&env, &NoKeychain);
    let options = LoadOptions {
        bind: "127.0.0.1".parse().unwrap(),
        allow_no_auth,
    };
    match config::load_str(
        yaml,
        Path::new("test.yaml"),
        Path::new("/"),
        &options,
        &resolver,
    ) {
        Ok(l) => l,
        Err(e) => panic!("{e}\n{yaml}"),
    }
}

pub struct Gateway {
    pub addr: std::net::SocketAddr,
    pub client: reqwest::Client,
    shutdown: Option<oneshot::Sender<()>>,
    pub task: Option<JoinHandle<()>>,
}

impl Gateway {
    pub async fn start(loaded: Loaded, options: ServerOptions, serve: ServeOptions) -> Self {
        Self::start_with_tls(loaded, options, serve, None).await
    }

    pub async fn start_with_tls(
        loaded: Loaded,
        options: ServerOptions,
        serve: ServeOptions,
        tls: Option<TlsAcceptor>,
    ) -> Self {
        let state = AppState::new(loaded, options).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(server::serve(
            listener,
            server::app(state),
            tls,
            serve,
            async {
                let _ = rx.await;
            },
        ));
        Self {
            addr,
            client: reqwest::Client::new(),
            shutdown: Some(tx),
            task: Some(task),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    pub fn shutdown(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }

    /// POST JSON with the master key as a bearer token.
    pub async fn post(&self, path: &str, body: &Value) -> reqwest::Response {
        self.post_with(
            path,
            body,
            &[("authorization", &format!("Bearer {MASTER_KEY}"))],
        )
        .await
    }

    pub async fn post_with(
        &self,
        path: &str,
        body: &Value,
        headers: &[(&str, &str)],
    ) -> reqwest::Response {
        let mut req = self
            .client
            .post(self.url(path))
            .header("content-type", "application/json")
            .body(body.to_string());
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        req.send().await.unwrap()
    }

    pub async fn get(&self, path: &str, auth: bool) -> reqwest::Response {
        let mut req = self.client.get(self.url(path));
        if auth {
            req = req.header("authorization", format!("Bearer {MASTER_KEY}"));
        }
        req.send().await.unwrap()
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Status, headers and the body parsed as JSON.
pub async fn json(resp: reqwest::Response) -> (u16, reqwest::header::HeaderMap, Value) {
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap();
    let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, headers, value)
}

/// Reads a whole response body as text, chunk by chunk.
pub async fn text(mut resp: reqwest::Response) -> String {
    let mut out = Vec::new();
    while let Some(chunk) = resp.chunk().await.unwrap() {
        out.extend_from_slice(&chunk);
    }
    String::from_utf8(out).unwrap()
}

// ---- files ----

/// A private temporary directory removed on drop.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new() -> Self {
        use std::os::unix::fs::DirBuilderExt;
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrolite-it-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .unwrap();
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
