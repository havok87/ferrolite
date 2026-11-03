//! Master-key authentication.

use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;
use subtle::ConstantTimeEq;

use crate::secrets::Secret;

/// The key a client presented: `Authorization: Bearer`, then `x-api-key`,
/// then `api-key`. An `Authorization` header with another scheme is skipped.
pub fn presented_key(headers: &HeaderMap) -> Option<&str> {
    let bearer = headers
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            let (scheme, token) = v.trim().split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
        });
    bearer
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .or_else(|| headers.get("api-key").and_then(|v| v.to_str().ok()))
        .filter(|k| !k.is_empty())
}

/// Whether the request may proceed. With no master key (`--allow-no-auth`)
/// everything is allowed. The comparison runs in constant time; a key of
/// the wrong length is rejected without comparing, which reveals only its
/// length.
pub fn authorized(master_key: Option<&Secret>, headers: &HeaderMap) -> bool {
    let Some(master) = master_key else {
        return true;
    };
    presented_key(headers)
        .is_some_and(|k| bool::from(k.as_bytes().ct_eq(master.expose().as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const KEY: &str = "sk-master-0123456789abcdef0123456789";

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn accepts_each_header() {
        let master = Secret::new(KEY);
        for h in [
            headers(&[("authorization", &format!("Bearer {KEY}"))]),
            headers(&[("authorization", &format!("bearer  {KEY} "))]),
            headers(&[("x-api-key", KEY)]),
            headers(&[("api-key", KEY)]),
            headers(&[("authorization", "Basic Zm9vOmJhcg=="), ("x-api-key", KEY)]),
        ] {
            assert!(authorized(Some(&master), &h), "{h:?}");
        }
    }

    #[test]
    fn rejects_wrong_or_missing_keys() {
        let master = Secret::new(KEY);
        for h in [
            headers(&[]),
            headers(&[("authorization", KEY)]),
            headers(&[("authorization", "Bearer sk-wrong")]),
            headers(&[("authorization", &format!("Bearer {KEY}x"))]),
            headers(&[("x-api-key", "")]),
            headers(&[("authorization", "Bearer sk-wrong"), ("x-api-key", KEY)]),
        ] {
            assert!(!authorized(Some(&master), &h), "{h:?}");
        }
    }

    #[test]
    fn no_master_key_allows_everything() {
        assert!(authorized(None, &headers(&[])));
    }
}
