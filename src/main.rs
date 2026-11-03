#![forbid(unsafe_code)]

use std::net::IpAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use ferrolite::config::{self, ConfigError, LoadOptions, Loaded};
use ferrolite::providers::{DEFAULT_MAX_RESPONSE_BYTES, UpstreamOptions};
use ferrolite::secrets::Resolver;
use ferrolite::server::{
    self, AppState, DEFAULT_MAX_REQUEST_BYTES, ServeOptions, ServerOptions, tls,
};
use tracing::level_filters::LevelFilter;
use tracing::{error, info, warn};

/// A small, security-focused, LiteLLM-compatible LLM gateway.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Path to a LiteLLM-format config.yaml.
    #[arg(long, default_value = "config.yaml")]
    config: PathBuf,

    /// Address to bind. Anything but loopback exposes the gateway to the network.
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,

    #[arg(long, default_value_t = 4000)]
    port: u16,

    /// Run without a master key. Refused unless --host is a loopback address.
    #[arg(long)]
    allow_no_auth: bool,

    /// One of off, error, warn, info, debug, trace.
    #[arg(long, default_value = "info")]
    log_level: LevelFilter,

    /// Load and validate the config, then exit.
    #[arg(long)]
    check: bool,

    /// Largest request body, in MiB [default: general_settings.max_request_size_mb, else 32].
    #[arg(long, value_parser = positive_mib)]
    max_request_mb: Option<usize>,

    /// Largest non-streaming upstream response, in MiB [default:
    /// general_settings.max_response_size_mb, else 64].
    #[arg(long, value_parser = positive_mib)]
    max_response_mb: Option<usize>,

    /// Proxied requests in progress at once (streams included); more get 503.
    #[arg(long, default_value_t = server::DEFAULT_MAX_CONCURRENCY as u32, value_parser = clap::value_parser!(u32).range(1..))]
    max_concurrency: u32,

    /// Open client connections at once; more are closed on accept.
    #[arg(long, default_value_t = server::DEFAULT_MAX_CONNECTIONS as u32, value_parser = clap::value_parser!(u32).range(1..))]
    max_connections: u32,

    /// Seconds to wait when connecting to an upstream.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u64).range(1..=300))]
    connect_timeout: u64,

    /// PEM certificate chain; serves HTTPS together with --tls-key.
    #[arg(long, requires = "tls_key")]
    tls_cert: Option<PathBuf>,

    /// PEM private key; must not be readable by group or other.
    #[arg(long, requires = "tls_cert")]
    tls_key: Option<PathBuf>,
}

/// A positive number of MiB, as bytes.
fn positive_mib(s: &str) -> Result<usize, String> {
    match s.parse::<f64>() {
        Ok(mb) if mb.is_finite() && mb > 0.0 => Ok((mb * 1024.0 * 1024.0) as usize),
        _ => Err("expected a positive number of MiB".to_owned()),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .json()
        .with_max_level(cli.log_level)
        .with_writer(std::io::stderr)
        .init();

    let options = LoadOptions {
        bind: cli.host,
        allow_no_auth: cli.allow_no_auth,
    };
    let loaded = match config::load(&cli.config, &options, &Resolver::system()) {
        Ok(loaded) => loaded,
        Err(ConfigError::Invalid {
            path,
            problems,
            warnings,
        }) => {
            for w in &warnings {
                warn!(config = %path.display(), "{w}");
            }
            for p in &problems {
                error!(config = %path.display(), "{p}");
            }
            error!(
                config = %path.display(),
                problems = problems.len(),
                "config is invalid"
            );
            return ExitCode::FAILURE;
        }
        Err(e) => {
            error!("{e}");
            return ExitCode::FAILURE;
        }
    };

    for w in &loaded.warnings {
        warn!(config = %cli.config.display(), "{w}");
    }
    if !cli.host.is_loopback() {
        warn!(host = %cli.host, "binding a non-loopback address exposes the gateway to the network");
    }
    for d in &loaded.config.deployments {
        info!(
            model_group = %d.model_name,
            deployment = %d.id,
            provider = d.provider.prefix(),
            "deployment configured"
        );
    }
    info!(
        config = %cli.config.display(),
        deployments = loaded.config.deployments.len(),
        model_groups = loaded.config.model_groups().len(),
        "config is valid"
    );

    if cli.check {
        return ExitCode::SUCCESS;
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            error!(error = %e, "cannot start the async runtime");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(run(cli, loaded))
}

async fn run(cli: Cli, loaded: Loaded) -> ExitCode {
    let options = ServerOptions {
        max_request_bytes: cli
            .max_request_mb
            .or(loaded.config.max_request_bytes)
            .unwrap_or(DEFAULT_MAX_REQUEST_BYTES),
        max_concurrency: cli.max_concurrency as usize,
        upstream: UpstreamOptions {
            max_response_bytes: cli
                .max_response_mb
                .or(loaded.config.max_response_bytes)
                .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES),
            connect_timeout: Duration::from_secs(cli.connect_timeout),
        },
        ..ServerOptions::default()
    };
    let tls = match (&cli.tls_cert, &cli.tls_key) {
        (Some(cert), Some(key)) => match tls::acceptor(cert, key) {
            Ok(acceptor) => Some(acceptor),
            Err(e) => {
                error!("{e}");
                return ExitCode::FAILURE;
            }
        },
        _ => None,
    };
    if tls.is_none() && !cli.host.is_loopback() {
        warn!(
            host = %cli.host,
            "serving plain HTTP on a non-loopback address; keys and prompts travel unencrypted (use --tls-cert/--tls-key)"
        );
    }
    let state = match AppState::new(loaded, options) {
        Ok(state) => state,
        Err(e) => {
            error!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match tokio::net::TcpListener::bind((cli.host, cli.port)).await {
        Ok(l) => l,
        Err(e) => {
            error!(host = %cli.host, port = cli.port, error = %e, "cannot bind");
            return ExitCode::FAILURE;
        }
    };
    let address = listener
        .local_addr()
        .map_or_else(|_| format!("{}:{}", cli.host, cli.port), |a| a.to_string());
    info!(
        address,
        tls = tls.is_some(),
        max_request_bytes = options.max_request_bytes,
        max_concurrency = options.max_concurrency,
        "listening"
    );
    let serve_options = ServeOptions {
        max_connections: cli.max_connections as usize,
        ..ServeOptions::default()
    };
    server::serve(
        listener,
        server::app(state),
        tls,
        serve_options,
        shutdown_signal(),
    )
    .await;
    info!("stopped");
    ExitCode::SUCCESS
}

/// SIGINT or SIGTERM.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    match signal(SignalKind::terminate()) {
        Ok(mut term) => {
            tokio::select! {
                () = ctrl_c => {}
                _ = term.recv() => {}
            }
        }
        Err(_) => ctrl_c.await,
    }
    info!("shutting down; finishing open requests");
}
