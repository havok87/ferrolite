//! Secret values, secret references in config, and the redactor.
//!
//! Config strings may be literal secrets or references:
//! `os.environ/VAR`, `file/<path>` and (macOS) `keychain/<service>/<account>`.
//! Resolved values live in [`Secret`], whose `Debug` and `Display` never show
//! the value. Every resolved secret is registered with a [`Redactor`] that
//! scrubs it from text before the text reaches a client or the log.

use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Placeholder printed instead of a secret value.
pub const REDACTED: &str = "[REDACTED]";

/// Secrets shorter than this (in characters) are not added to the redactor:
/// scrubbing them would mangle ordinary text. Config loading warns about them.
pub const MIN_REDACTABLE_CHARS: usize = 8;

const MAX_SECRET_FILE_BYTES: u64 = 64 * 1024;
const SECURITY_BIN: &str = "/usr/bin/security";
const KEYCHAIN_TIMEOUT: Duration = Duration::from_secs(15);

/// A secret value. `Debug` and `Display` print `[REDACTED]`; use
/// [`Secret::expose`] at the one place the value is actually needed.
#[derive(Clone)]
pub struct Secret(Box<str>);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into().into_boxed_str())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Where a secret comes from, parsed from a config string.
#[derive(Debug, Clone)]
pub enum SecretRef {
    Literal(Secret),
    Env(String),
    File(PathBuf),
    Keychain { service: String, account: String },
}

impl SecretRef {
    pub fn parse(raw: &str) -> Result<Self, SecretError> {
        if let Some(name) = raw.strip_prefix("os.environ/") {
            let valid =
                !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
            if !valid {
                return Err(SecretError::InvalidRef {
                    reference: format!("os.environ/{}", printable(name)),
                    reason: "variable names may contain only ASCII letters, digits and '_'",
                });
            }
            return Ok(Self::Env(name.to_owned()));
        }
        if let Some(path) = raw.strip_prefix("file/") {
            if path.is_empty() || path.chars().any(char::is_control) {
                return Err(SecretError::InvalidRef {
                    reference: format!("file/{}", printable(path)),
                    reason: "the path must be non-empty and contain no control characters",
                });
            }
            return Ok(Self::File(PathBuf::from(path)));
        }
        if let Some(rest) = raw.strip_prefix("keychain/") {
            let parts = rest.split_once('/');
            let valid = parts.is_some_and(|(service, account)| {
                !service.is_empty() && !account.is_empty() && !rest.chars().any(char::is_control)
            });
            return match parts {
                Some((service, account)) if valid => Ok(Self::Keychain {
                    service: service.to_owned(),
                    account: account.to_owned(),
                }),
                _ => Err(SecretError::InvalidRef {
                    reference: format!("keychain/{}", printable(rest)),
                    reason: "expected keychain/<service>/<account> with no control characters",
                }),
            };
        }
        Ok(Self::Literal(Secret::new(raw)))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(_))
    }
}

/// Renders a reference fragment for an error message, escaping control characters.
fn printable(s: &str) -> String {
    s.escape_debug().to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("invalid secret reference `{reference}`: {reason}")]
    InvalidRef {
        reference: String,
        reason: &'static str,
    },
    #[error("environment variable `{var}` (from os.environ/{var}) is not set or not valid UTF-8")]
    EnvMissing { var: String },
    #[error("environment variable `{var}` (from os.environ/{var}) is empty")]
    EnvEmpty { var: String },
    #[error("cannot read secret file {}: {source}", path.display())]
    FileIo { path: PathBuf, source: io::Error },
    #[error("secret file {} is not a regular file", path.display())]
    FileNotRegular { path: PathBuf },
    #[error(
        "secret file {} has mode {mode:03o}; it must not be accessible by group or other (chmod 600)",
        path.display()
    )]
    FilePermissions { path: PathBuf, mode: u32 },
    #[error("secret file {} changed while it was being opened", path.display())]
    FileChanged { path: PathBuf },
    #[error("secret file {} is larger than {limit} bytes", path.display())]
    FileTooLarge { path: PathBuf, limit: u64 },
    #[error("secret file {} is not valid UTF-8", path.display())]
    FileNotUtf8 { path: PathBuf },
    #[error("secret file {} is empty", path.display())]
    FileEmpty { path: PathBuf },
    #[error("keychain item service=`{service}` account=`{account}`: {reason}")]
    Keychain {
        service: String,
        account: String,
        reason: String,
    },
}

/// Source of environment variables. Injectable so tests never need
/// `std::env::set_var`, which is `unsafe` in Rust 2024.
pub trait EnvSource {
    fn var(&self, name: &str) -> Option<String>;
}

/// The process environment.
pub struct SystemEnv;

impl EnvSource for SystemEnv {
    fn var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

impl EnvSource for std::collections::HashMap<String, String> {
    fn var(&self, name: &str) -> Option<String> {
        self.get(name).cloned()
    }
}

/// Looks up a generic password in the macOS keychain. Errors are short
/// reasons that never include the secret.
pub trait KeychainRunner {
    fn find_generic_password(&self, service: &str, account: &str) -> Result<Secret, String>;
}

/// Runs `/usr/bin/security find-generic-password -w` by absolute path, with no shell.
pub struct SecurityCli;

impl KeychainRunner for SecurityCli {
    fn find_generic_password(&self, service: &str, account: &str) -> Result<Secret, String> {
        if !cfg!(target_os = "macos") {
            return Err("keychain references are supported only on macOS".to_owned());
        }
        let mut cmd = Command::new(SECURITY_BIN);
        cmd.args(keychain_args(service, account));
        let stdout = run_with_timeout(cmd, KEYCHAIN_TIMEOUT, MAX_SECRET_FILE_BYTES)?;
        let text = String::from_utf8(stdout)
            .map_err(|_| "`security` printed a value that is not valid UTF-8".to_owned())?;
        let value = strip_one_newline(&text);
        if value.is_empty() {
            return Err("the stored password is empty".to_owned());
        }
        Ok(Secret::new(value))
    }
}

fn keychain_args<'a>(service: &'a str, account: &'a str) -> [&'a str; 6] {
    ["find-generic-password", "-s", service, "-a", account, "-w"]
}

/// Runs `cmd` with stdin and stderr closed, returning at most `max_bytes` of
/// stdout. Kills the child if it outlives `timeout` (for example while a
/// keychain unlock prompt is waiting on a user who is not there).
fn run_with_timeout(
    mut cmd: Command,
    timeout: Duration,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot run `security`: {e}"))?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("cannot capture `security` output".to_owned());
    };
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        stdout
            .take(max_bytes + 1)
            .read_to_end(&mut buf)
            .map(|_| buf)
    });

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return Err(format!("timed out after {}s", timeout.as_secs_f64()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("cannot wait for `security`: {e}"));
            }
        }
    };
    let output = match reader.join() {
        Ok(Ok(buf)) => buf,
        _ => return Err("cannot read `security` output".to_owned()),
    };
    if !status.success() {
        // `security` exits 44 when the item does not exist.
        return Err(match status.code() {
            Some(44) => "no such item in the keychain".to_owned(),
            Some(code) => format!("`security` exited with status {code}"),
            None => "`security` was terminated by a signal".to_owned(),
        });
    }
    if output.len() as u64 > max_bytes {
        return Err(format!("value is larger than {max_bytes} bytes"));
    }
    Ok(output)
}

/// Resolves [`SecretRef`]s against an environment and the keychain.
#[derive(Clone, Copy)]
pub struct Resolver<'a> {
    env: &'a dyn EnvSource,
    keychain: &'a dyn KeychainRunner,
}

impl<'a> Resolver<'a> {
    pub fn new(env: &'a dyn EnvSource, keychain: &'a dyn KeychainRunner) -> Self {
        Self { env, keychain }
    }

    /// The process environment and `/usr/bin/security`.
    pub fn system() -> Resolver<'static> {
        Resolver {
            env: &SystemEnv,
            keychain: &SecurityCli,
        }
    }

    /// Resolves `reference`. Relative `file/` paths are taken from `base_dir`,
    /// normally the directory holding `config.yaml`.
    pub fn resolve(&self, reference: &SecretRef, base_dir: &Path) -> Result<Secret, SecretError> {
        match reference {
            SecretRef::Literal(secret) => Ok(secret.clone()),
            SecretRef::Env(var) => match self.env.var(var) {
                None => Err(SecretError::EnvMissing { var: var.clone() }),
                Some(v) if v.is_empty() => Err(SecretError::EnvEmpty { var: var.clone() }),
                Some(v) => Ok(Secret::new(v)),
            },
            SecretRef::File(path) => read_secret_file(&base_dir.join(path)),
            SecretRef::Keychain { service, account } => self
                .keychain
                .find_generic_password(service, account)
                .map_err(|reason| SecretError::Keychain {
                    service: service.clone(),
                    account: account.clone(),
                    reason,
                }),
        }
    }
}

/// Reads a secret file, refusing anything group/other can access.
///
/// The path is checked with `stat` before opening, so a FIFO or device is never
/// opened, and the open handle is checked again with `fstat`, so the file
/// that is read is the file that was checked.
fn read_secret_file(path: &Path) -> Result<Secret, SecretError> {
    use std::os::unix::fs::MetadataExt;

    let io_err = |source| SecretError::FileIo {
        path: path.to_owned(),
        source,
    };
    let check = |meta: &std::fs::Metadata| {
        if !meta.is_file() {
            return Err(SecretError::FileNotRegular {
                path: path.to_owned(),
            });
        }
        let mode = meta.mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(SecretError::FilePermissions {
                path: path.to_owned(),
                mode,
            });
        }
        if meta.len() > MAX_SECRET_FILE_BYTES {
            return Err(SecretError::FileTooLarge {
                path: path.to_owned(),
                limit: MAX_SECRET_FILE_BYTES,
            });
        }
        Ok(())
    };

    let before = std::fs::metadata(path).map_err(io_err)?;
    check(&before)?;
    let file = File::open(path).map_err(io_err)?;
    let after = file.metadata().map_err(io_err)?;
    if (before.dev(), before.ino()) != (after.dev(), after.ino()) {
        return Err(SecretError::FileChanged {
            path: path.to_owned(),
        });
    }
    check(&after)?;

    let mut buf = Vec::new();
    file.take(MAX_SECRET_FILE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(io_err)?;
    if buf.len() as u64 > MAX_SECRET_FILE_BYTES {
        return Err(SecretError::FileTooLarge {
            path: path.to_owned(),
            limit: MAX_SECRET_FILE_BYTES,
        });
    }
    let text = String::from_utf8(buf).map_err(|_| SecretError::FileNotUtf8 {
        path: path.to_owned(),
    })?;
    let value = strip_one_newline(&text);
    if value.is_empty() {
        return Err(SecretError::FileEmpty {
            path: path.to_owned(),
        });
    }
    Ok(Secret::new(value))
}

fn strip_one_newline(s: &str) -> &str {
    s.strip_suffix("\r\n")
        .or_else(|| s.strip_suffix('\n'))
        .unwrap_or(s)
}

/// Scrubs configured secret values from text.
#[derive(Clone, Default)]
pub struct Redactor {
    patterns: Arc<[Box<str>]>,
}

impl Redactor {
    /// Builds a redactor from `secrets`, skipping any shorter than
    /// [`MIN_REDACTABLE_CHARS`] (see [`Redactor::is_redactable`]).
    pub fn new<'s>(secrets: impl IntoIterator<Item = &'s Secret>) -> Self {
        let mut patterns: Vec<Box<str>> = secrets
            .into_iter()
            .filter(|s| Self::is_redactable(s))
            .map(|s| Box::from(s.expose()))
            .collect();
        patterns.sort_unstable();
        patterns.dedup();
        Self {
            patterns: patterns.into(),
        }
    }

    pub fn is_redactable(secret: &Secret) -> bool {
        secret.expose().chars().count() >= MIN_REDACTABLE_CHARS
    }

    /// Replaces every occurrence of every secret with `[REDACTED]`.
    ///
    /// Matches are collected for all secrets first and merged, so secrets that
    /// overlap in the text are removed whole rather than leaving fragments.
    pub fn redact<'t>(&self, text: &'t str) -> Cow<'t, str> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for pattern in self.patterns.iter() {
            let mut from = 0;
            while let Some(pos) = text[from..].find(&**pattern) {
                let start = from + pos;
                spans.push((start, start + pattern.len()));
                // Step one character so overlapping occurrences are found too.
                from = start + text[start..].chars().next().map_or(1, char::len_utf8);
            }
        }
        if spans.is_empty() {
            return Cow::Borrowed(text);
        }
        spans.sort_unstable();
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0;
        let mut iter = spans.into_iter().peekable();
        while let Some((start, mut end)) = iter.next() {
            while let Some(&(next_start, next_end)) = iter.peek() {
                if next_start > end {
                    break;
                }
                end = end.max(next_end);
                iter.next();
            }
            out.push_str(&text[cursor..start]);
            out.push_str(REDACTED);
            cursor = end;
        }
        out.push_str(&text[cursor..]);
        Cow::Owned(out)
    }
}

impl fmt::Debug for Redactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Redactor")
            .field("patterns", &self.patterns.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use std::collections::HashMap;
    use std::os::unix::fs::PermissionsExt;

    struct FakeKeychain;

    impl KeychainRunner for FakeKeychain {
        fn find_generic_password(&self, service: &str, account: &str) -> Result<Secret, String> {
            match (service, account) {
                ("ferrolite", "openai") => Ok(Secret::new("sk-from-keychain-0123")),
                _ => Err("no such item in the keychain".to_owned()),
            }
        }
    }

    fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    fn write_secret(dir: &TempDir, name: &str, contents: &[u8], mode: u32) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        path
    }

    #[test]
    fn secret_never_prints_its_value() {
        let s = Secret::new("sk-super-secret-value");
        assert_eq!(format!("{s}"), REDACTED);
        assert_eq!(format!("{s:?}"), REDACTED);
        assert_eq!(format!("{:?}", Some(s.clone())), "Some([REDACTED])");
        assert_eq!(s.expose(), "sk-super-secret-value");
    }

    #[test]
    fn parses_references() {
        assert!(
            matches!(SecretRef::parse("os.environ/OPENAI_API_KEY").unwrap(), SecretRef::Env(v) if v == "OPENAI_API_KEY")
        );
        assert!(
            matches!(SecretRef::parse("file/keys/openai").unwrap(), SecretRef::File(p) if p == Path::new("keys/openai"))
        );
        assert!(matches!(
            SecretRef::parse("keychain/ferrolite/team/openai").unwrap(),
            SecretRef::Keychain { service, account } if service == "ferrolite" && account == "team/openai"
        ));
        let literal = SecretRef::parse("sk-literal").unwrap();
        assert!(literal.is_literal());
        assert!(!SecretRef::parse("os.environ/X").unwrap().is_literal());
    }

    #[test]
    fn rejects_malformed_references() {
        for bad in [
            "os.environ/",
            "os.environ/HAS-DASH",
            "os.environ/A B",
            "file/",
            "file/a\nb",
            "keychain/",
            "keychain/service",
            "keychain/service/",
            "keychain//account",
            "keychain/svc/acc\0",
        ] {
            let err = SecretRef::parse(bad).unwrap_err();
            assert!(
                matches!(err, SecretError::InvalidRef { .. }),
                "{bad:?}: {err}"
            );
        }
    }

    #[test]
    fn resolves_environment_variables() {
        let env = env(&[("KEY", "sk-env-value"), ("EMPTY", "")]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let base = Path::new("/");
        let got = resolver
            .resolve(&SecretRef::Env("KEY".into()), base)
            .unwrap();
        assert_eq!(got.expose(), "sk-env-value");

        let err = resolver
            .resolve(&SecretRef::Env("MISSING".into()), base)
            .unwrap_err();
        assert!(matches!(err, SecretError::EnvMissing { .. }));
        assert!(err.to_string().contains("MISSING"));

        let err = resolver
            .resolve(&SecretRef::Env("EMPTY".into()), base)
            .unwrap_err();
        assert!(matches!(err, SecretError::EnvEmpty { .. }));
    }

    #[test]
    fn resolves_keychain_through_runner() {
        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let ok = SecretRef::parse("keychain/ferrolite/openai").unwrap();
        assert_eq!(
            resolver.resolve(&ok, Path::new("/")).unwrap().expose(),
            "sk-from-keychain-0123"
        );
        let missing = SecretRef::parse("keychain/ferrolite/nope").unwrap();
        let err = resolver.resolve(&missing, Path::new("/")).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("service=`ferrolite`") && msg.contains("account=`nope`"),
            "{msg}"
        );
    }

    #[test]
    fn keychain_command_is_absolute_and_shell_free() {
        assert!(Path::new(SECURITY_BIN).is_absolute());
        assert_eq!(
            keychain_args("svc; rm -rf /", "$(acct)"),
            [
                "find-generic-password",
                "-s",
                "svc; rm -rf /",
                "-a",
                "$(acct)",
                "-w"
            ]
        );
    }

    #[test]
    fn run_with_timeout_captures_stdout() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hello");
        let out = run_with_timeout(cmd, Duration::from_secs(5), 1024).unwrap();
        assert_eq!(out, b"hello\n");
    }

    #[test]
    fn run_with_timeout_kills_slow_children() {
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("10");
        let started = Instant::now();
        let err = run_with_timeout(cmd, Duration::from_millis(200), 1024).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn run_with_timeout_reports_failure_status() {
        let err = run_with_timeout(Command::new("/usr/bin/false"), Duration::from_secs(5), 1024)
            .unwrap_err();
        assert!(err.contains("status"), "{err}");
    }

    #[test]
    fn reads_private_secret_file_and_trims_one_newline() {
        let dir = TempDir::new();
        write_secret(&dir, "key", b"sk-file-secret\n", 0o600);
        write_secret(&dir, "crlf", b"sk-file-secret\r\n", 0o400);
        write_secret(&dir, "two", b"sk-file-secret\n\n", 0o600);
        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let resolve = |name: &str| {
            resolver
                .resolve(&SecretRef::File(name.into()), dir.path())
                .map(|s| s.expose().to_owned())
        };
        assert_eq!(resolve("key").unwrap(), "sk-file-secret");
        assert_eq!(resolve("crlf").unwrap(), "sk-file-secret");
        assert_eq!(resolve("two").unwrap(), "sk-file-secret\n");
    }

    #[test]
    fn refuses_secret_files_others_can_access() {
        let dir = TempDir::new();
        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        for mode in [0o640, 0o604, 0o644, 0o620, 0o602, 0o610] {
            let name = format!("key{mode:o}");
            write_secret(&dir, &name, b"sk-file-secret", mode);
            let err = resolver
                .resolve(&SecretRef::File(name.clone().into()), dir.path())
                .unwrap_err();
            assert!(
                matches!(err, SecretError::FilePermissions { .. }),
                "{mode:o}: {err}"
            );
            assert!(!err.to_string().contains("sk-file-secret"));
        }
    }

    #[test]
    fn refuses_symlink_to_readable_file() {
        let dir = TempDir::new();
        let target = write_secret(&dir, "target", b"sk-file-secret", 0o644);
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let err = resolver
            .resolve(&SecretRef::File("link".into()), dir.path())
            .unwrap_err();
        assert!(matches!(err, SecretError::FilePermissions { .. }), "{err}");
    }

    #[test]
    fn refuses_directories_empty_oversize_and_missing_files() {
        let dir = TempDir::new();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o700)).unwrap();
        write_secret(&dir, "empty", b"\n", 0o600);
        let big = vec![b'a'; (MAX_SECRET_FILE_BYTES + 1) as usize];
        write_secret(&dir, "big", &big, 0o600);
        write_secret(&dir, "latin1", b"\xff\xfe", 0o600);

        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let resolve = |name: &str| resolver.resolve(&SecretRef::File(name.into()), dir.path());
        assert!(matches!(
            resolve("sub"),
            Err(SecretError::FileNotRegular { .. })
        ));
        assert!(matches!(
            resolve("empty"),
            Err(SecretError::FileEmpty { .. })
        ));
        assert!(matches!(
            resolve("big"),
            Err(SecretError::FileTooLarge { .. })
        ));
        assert!(matches!(
            resolve("latin1"),
            Err(SecretError::FileNotUtf8 { .. })
        ));
        assert!(matches!(
            resolve("missing"),
            Err(SecretError::FileIo { .. })
        ));
    }

    #[test]
    fn absolute_file_paths_ignore_base_dir() {
        let dir = TempDir::new();
        let path = write_secret(&dir, "key", b"sk-file-secret", 0o600);
        let env = env(&[]);
        let resolver = Resolver::new(&env, &FakeKeychain);
        let got = resolver
            .resolve(&SecretRef::File(path), Path::new("/nonexistent"))
            .unwrap();
        assert_eq!(got.expose(), "sk-file-secret");
    }

    #[test]
    fn redacts_every_occurrence() {
        let a = Secret::new("sk-aaaaaaaaaaaa");
        let b = Secret::new("tok-bbbbbbbbbb");
        let r = Redactor::new([&a, &b]);
        assert_eq!(
            r.redact("x sk-aaaaaaaaaaaa y tok-bbbbbbbbbb z sk-aaaaaaaaaaaa"),
            "x [REDACTED] y [REDACTED] z [REDACTED]"
        );
    }

    #[test]
    fn redacts_overlapping_secrets_whole() {
        let a = Secret::new("abcdefgh12");
        let b = Secret::new("12345678xx");
        let r = Redactor::new([&a, &b]);
        assert_eq!(r.redact("<abcdefgh12345678xx>"), "<[REDACTED]>");

        let nested_outer = Secret::new("prefix-sk-0123456789");
        let nested_inner = Secret::new("sk-0123456789");
        let r = Redactor::new([&nested_inner, &nested_outer]);
        assert_eq!(r.redact("k=prefix-sk-0123456789;"), "k=[REDACTED];");
    }

    #[test]
    fn leaves_text_without_secrets_untouched() {
        let s = Secret::new("sk-aaaaaaaaaaaa");
        let r = Redactor::new([&s]);
        assert!(matches!(r.redact("nothing to see"), Cow::Borrowed(_)));
        assert!(matches!(Redactor::default().redact("x"), Cow::Borrowed(_)));
    }

    #[test]
    fn skips_short_secrets() {
        let short = Secret::new("abc");
        assert!(!Redactor::is_redactable(&short));
        let r = Redactor::new([&short]);
        assert_eq!(r.redact("abcabc"), "abcabc");
        assert_eq!(format!("{r:?}"), "Redactor { patterns: 0 }");
    }

    #[test]
    fn redacts_multibyte_text() {
        let s = Secret::new("clé-secrète-ü");
        let r = Redactor::new([&s]);
        assert_eq!(r.redact("ü→clé-secrète-ü←ü"), "ü→[REDACTED]←ü");
    }
}
