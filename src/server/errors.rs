//! Client-facing errors: status mapping and the OpenAI and Anthropic
//! error shapes.

use axum::body::Body;
use axum::http::{HeaderValue, Response, header};
use serde_json::{Value, json};

use crate::providers::ProviderError;
use crate::router::{ErrorClass, RouterError};

/// Which error shape the client expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorFormat {
    OpenAi,
    /// `/v1/messages`.
    Anthropic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub code: Option<&'static str>,
    pub retry_after: Option<u64>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: None,
            retry_after: None,
        }
    }

    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub fn with_retry_after(mut self, seconds: u64) -> Self {
        self.retry_after = Some(seconds.max(1));
        self
    }

    /// Maps a routing failure to what the client sees.
    pub fn from_router(err: &RouterError<ProviderError>) -> Self {
        match err {
            RouterError::UnknownModel { model } => Self::new(
                404,
                format!("model `{}` is not configured", crate::translate::truncate_chars(model, 200)),
            )
            .with_code("model_not_found"),
            RouterError::NoDeployment { group, retry_after } => Self::new(
                429,
                format!("no deployment of `{group}` is available right now (cooling down or rate limited)"),
            )
            .with_retry_after(retry_after.as_secs_f64().ceil() as u64),
            RouterError::Timeout { .. } => Self::new(504, "the upstream did not respond in time"),
            RouterError::Upstream { error, .. } => {
                let mut e = Self::new(client_status(error.status), error.message.clone());
                if let ErrorClass::RateLimited {
                    retry_after: Some(after),
                } = error.class
                {
                    e = e.with_retry_after(after.as_secs_f64().ceil() as u64);
                }
                if error.class == ErrorClass::ContextWindowExceeded {
                    e = e.with_code("context_length_exceeded");
                }
                e
            }
        }
    }

    fn openai_type(&self) -> &'static str {
        match self.status {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "invalid_request_error",
            429 => "rate_limit_error",
            400..=499 => "invalid_request_error",
            503 => "service_unavailable",
            504 => "timeout",
            _ => "api_error",
        }
    }

    fn anthropic_type(&self) -> &'static str {
        match self.status {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            400..=499 => "invalid_request_error",
            503 | 529 => "overloaded_error",
            _ => "api_error",
        }
    }

    pub fn body(&self, format: ErrorFormat) -> Value {
        match format {
            ErrorFormat::OpenAi => json!({"error": {
                "message": self.message,
                "type": self.openai_type(),
                "param": null,
                "code": self.code,
            }}),
            ErrorFormat::Anthropic => json!({"type": "error", "error": {
                "type": self.anthropic_type(),
                "message": self.message,
            }}),
        }
    }

    pub fn response(&self, format: ErrorFormat) -> Response<Body> {
        let mut resp = super::json_response(self.status, self.body(format).to_string());
        if let Some(after) = self.retry_after {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(after));
        }
        resp
    }
}

/// The status a client sees for an upstream status.
///
/// Client errors that describe the request pass through. The upstream
/// rejecting the configured credentials is a gateway problem, not the
/// client's, so it becomes 502, as do server errors, except "unavailable"
/// and "timeout", which keep their meaning (529 "overloaded" becomes 503,
/// and an upstream 408 becomes 504).
pub fn client_status(upstream: u16) -> u16 {
    match upstream {
        400 | 404 | 413 | 422 | 429 | 503 | 504 => upstream,
        401 | 403 => 502,
        // The upstream timing out reading our request is not the client's 408.
        408 => 504,
        529 => 503,
        402..=499 => 400,
        _ => 502,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn upstream(class: ErrorClass, status: u16) -> RouterError<ProviderError> {
        RouterError::Upstream {
            deployment: "m#0".into(),
            error: ProviderError {
                class,
                status,
                message: "upstream said no".into(),
            },
        }
    }

    #[test]
    fn status_mapping() {
        for (up, client) in [
            (400, 400),
            (401, 502),
            (403, 502),
            (404, 404),
            (408, 504),
            (409, 400),
            (413, 413),
            (422, 422),
            (429, 429),
            (500, 502),
            (502, 502),
            (503, 503),
            (504, 504),
            (529, 503),
        ] {
            assert_eq!(client_status(up), client, "{up}");
        }
    }

    #[test]
    fn router_errors() {
        let e = ApiError::from_router(&RouterError::UnknownModel {
            model: "nope".into(),
        });
        assert_eq!((e.status, e.code), (404, Some("model_not_found")));

        let e = ApiError::from_router(&RouterError::NoDeployment {
            group: "m".into(),
            retry_after: Duration::from_millis(29_200),
        });
        assert_eq!((e.status, e.retry_after), (429, Some(30)));
        let e = ApiError::from_router(&RouterError::NoDeployment {
            group: "m".into(),
            retry_after: Duration::ZERO,
        });
        assert_eq!(e.retry_after, Some(1));

        let e = ApiError::from_router(&upstream(
            ErrorClass::RateLimited {
                retry_after: Some(Duration::from_millis(1500)),
            },
            429,
        ));
        assert_eq!((e.status, e.retry_after), (429, Some(2)));

        let e = ApiError::from_router(&upstream(ErrorClass::ContextWindowExceeded, 400));
        assert_eq!(e.code, Some("context_length_exceeded"));
        assert_eq!(e.message, "upstream said no");
    }

    #[test]
    fn error_shapes() {
        let e = ApiError::new(429, "slow down").with_retry_after(3);
        assert_eq!(
            e.body(ErrorFormat::OpenAi),
            json!({"error": {"message": "slow down", "type": "rate_limit_error", "param": null, "code": null}})
        );
        assert_eq!(
            e.body(ErrorFormat::Anthropic),
            json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
        );
        let resp = e.response(ErrorFormat::OpenAi);
        assert_eq!(resp.status(), 429);
        assert_eq!(resp.headers()["retry-after"], "3");
        assert_eq!(resp.headers()["cache-control"], "no-store");
    }
}
