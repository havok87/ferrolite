#![forbid(unsafe_code)]

use std::net::IpAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use ferrolite::config::{self, ConfigError, LoadOptions};
use ferrolite::secrets::Resolver;
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
    error!(
        host = %cli.host,
        port = cli.port,
        "the HTTP server is not implemented yet; use --check to validate a config"
    );
    ExitCode::from(2)
}
