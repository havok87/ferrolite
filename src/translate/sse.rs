//! A small incremental parser and writer for server-sent events
//! (<https://html.spec.whatwg.org/multipage/server-sent-events.html>).
//!
//! Only `event` and `data` matter here; `id` and `retry` are ignored, as
//! upstream LLM APIs do not use them.

/// Largest event (or unterminated line) accepted from an upstream.
pub const MAX_EVENT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("upstream sent a server-sent event larger than {limit} bytes")]
pub struct SseTooLarge {
    pub limit: usize,
}

/// Feed bytes in as they arrive; complete events come out.
#[derive(Debug)]
pub struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: String,
    has_data: bool,
    limit: usize,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::with_limit(MAX_EVENT_BYTES)
    }
}

impl SseParser {
    pub fn with_limit(limit: usize) -> Self {
        Self {
            buf: Vec::new(),
            event: None,
            data: String::new(),
            has_data: false,
            limit,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, SseTooLarge> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        let mut start = 0;
        loop {
            let rest = &self.buf[start..];
            let Some(pos) = rest.iter().position(|&b| b == b'\n' || b == b'\r') else {
                break;
            };
            let terminator_len = if rest[pos] == b'\r' {
                match rest.get(pos + 1) {
                    Some(b'\n') => 2,
                    Some(_) => 1,
                    // A trailing CR may be the first half of CRLF; wait for more.
                    None => break,
                }
            } else {
                1
            };
            let line = String::from_utf8_lossy(&rest[..pos]).into_owned();
            start += pos + terminator_len;
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
            if self.data.len() > self.limit {
                return Err(SseTooLarge { limit: self.limit });
            }
        }
        self.buf.drain(..start);
        if self.buf.len() > self.limit {
            return Err(SseTooLarge { limit: self.limit });
        }
        Ok(events)
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            let event = self.event.take();
            if !std::mem::take(&mut self.has_data) {
                self.data.clear();
                return None;
            }
            let mut data = std::mem::take(&mut self.data);
            if data.ends_with('\n') {
                data.pop();
            }
            return Some(SseEvent { event, data });
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "data" => {
                self.data.push_str(value);
                self.data.push('\n');
                self.has_data = true;
            }
            "event" => self.event = Some(value.to_owned()),
            _ => {}
        }
        None
    }
}

/// Serializes one event. Multi-line data becomes several `data:` lines.
pub fn write_event(out: &mut Vec<u8>, event: Option<&str>, data: &str) {
    if let Some(name) = event {
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }
    for line in data.split('\n') {
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(line.as_bytes());
        out.push(b'\n');
    }
    out.push(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event: Option<&str>, data: &str) -> SseEvent {
        SseEvent {
            event: event.map(str::to_owned),
            data: data.to_owned(),
        }
    }

    #[test]
    fn parses_events_and_ignores_comments() {
        let mut p = SseParser::default();
        let got = p
            .push(b": keep-alive\nevent: message_start\ndata: {\"a\":1}\n\ndata: x\nid: 7\nretry: 10\n\n")
            .unwrap();
        assert_eq!(got, [ev(Some("message_start"), "{\"a\":1}"), ev(None, "x")]);
    }

    #[test]
    fn handles_every_line_ending() {
        for input in [
            &b"data: a\n\ndata: b\n\n"[..],
            b"data: a\r\n\r\ndata: b\r\n\r\n",
            // A trailing CR could start a CRLF, so it waits for the next byte.
            b"data: a\r\rdata: b\r\r:",
        ] {
            let mut p = SseParser::default();
            assert_eq!(p.push(input).unwrap(), [ev(None, "a"), ev(None, "b")]);
        }
    }

    #[test]
    fn joins_multi_line_data_and_keeps_inner_spaces() {
        let mut p = SseParser::default();
        let got = p.push(b"data: one\ndata:  two\ndata\n\n").unwrap();
        assert_eq!(got, [ev(None, "one\n two\n")]);
    }

    #[test]
    fn blank_events_and_event_without_data_are_dropped() {
        let mut p = SseParser::default();
        assert!(p.push(b"\n\nevent: ping\n\n").unwrap().is_empty());
        // The event name does not leak into the next event.
        assert_eq!(p.push(b"data: x\n\n").unwrap(), [ev(None, "x")]);
    }

    #[test]
    fn events_split_at_every_byte() {
        let input = b"event: a\r\ndata: hello\r\ndata: world\r\n\r\ndata: [DONE]\n\n";
        let expected = [ev(Some("a"), "hello\nworld"), ev(None, "[DONE]")];
        for split in 0..=input.len() {
            let mut p = SseParser::default();
            let mut got = p.push(&input[..split]).unwrap();
            got.extend(p.push(&input[split..]).unwrap());
            assert_eq!(got, expected, "split at {split}");
        }
        let mut p = SseParser::default();
        let mut got = Vec::new();
        for b in input {
            got.extend(p.push(&[*b]).unwrap());
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn rejects_oversized_events_and_lines() {
        let mut p = SseParser::with_limit(16);
        assert_eq!(
            p.push(b"data: 0123456789abcdef0123\n"),
            Err(SseTooLarge { limit: 16 })
        );
        let mut p = SseParser::with_limit(16);
        assert_eq!(
            p.push(b"data: 0123456789abcdef0123"),
            Err(SseTooLarge { limit: 16 })
        );
        let mut p = SseParser::with_limit(16);
        assert!(p.push(b"data: 0123456\n").is_ok());
        assert!(p.push(b"data: 01234567\n").is_err());
    }

    #[test]
    fn writes_events() {
        let mut out = Vec::new();
        write_event(
            &mut out,
            Some("message_stop"),
            "{\"type\":\"message_stop\"}",
        );
        write_event(&mut out, None, "a\nb");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\ndata: a\ndata: b\n\n"
        );
    }

    #[test]
    fn round_trips() {
        let mut out = Vec::new();
        write_event(&mut out, Some("x"), "line1\nline2");
        let mut p = SseParser::default();
        assert_eq!(p.push(&out).unwrap(), [ev(Some("x"), "line1\nline2")]);
    }
}
