//! Optional server TLS (`--tls-cert`, `--tls-key`): rustls with aws-lc-rs,
//! HTTP/1.1 only.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::crypto::aws_lc_rs;
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};

use crate::secrets::{SecretError, read_private_file};

const MAX_CERT_BYTES: u64 = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("cannot read TLS certificate {}: {source}", path.display())]
    Cert {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("TLS certificate {} contains no PEM certificates", path.display())]
    NoCert { path: PathBuf },
    #[error("TLS key: {0}")]
    Key(#[from] SecretError),
    #[error("TLS key {} is not a PEM private key", path.display())]
    BadKey { path: PathBuf },
    #[error("TLS configuration rejected: {0}")]
    Config(String),
}

/// Loads the certificate chain and the key. The key file is refused if
/// group or other can access it, like `file/` secrets.
pub fn acceptor(cert_path: &Path, key_path: &Path) -> Result<TlsAcceptor, TlsError> {
    let cert_err = |source| TlsError::Cert {
        path: cert_path.to_owned(),
        source,
    };
    let size = std::fs::metadata(cert_path).map_err(cert_err)?.len();
    if size > MAX_CERT_BYTES {
        return Err(cert_err(std::io::Error::other("file is too large")));
    }
    let pem = std::fs::read(cert_path).map_err(cert_err)?;
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(&pem)
        .filter_map(Result::ok)
        .collect();
    if certs.is_empty() {
        return Err(TlsError::NoCert {
            path: cert_path.to_owned(),
        });
    }
    let key_pem = read_private_file(key_path)?;
    let key = PrivateKeyDer::from_pem_slice(key_pem.expose().as_bytes()).map_err(|_| {
        TlsError::BadKey {
            path: key_path.to_owned(),
        }
    })?;

    let mut config = ServerConfig::builder_with_provider(Arc::new(aws_lc_rs::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| TlsError::Config(e.to_string()))?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| TlsError::Config(e.to_string()))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(TlsAcceptor::from(Arc::new(config)))
}
