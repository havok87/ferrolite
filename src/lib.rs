//! ferrolite: a small, security-focused reimplementation of the LiteLLM
//! proxy's routing core.

#![forbid(unsafe_code)]

#[cfg(not(unix))]
compile_error!("ferrolite supports macOS and Linux only");

pub mod config;
pub mod router;
pub mod secrets;

#[cfg(test)]
pub(crate) mod test_support;
