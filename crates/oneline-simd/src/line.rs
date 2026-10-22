use oneline_core::{FlatNode, NodeKind};

/// Parses a single line byte slice into a FlatNode.
#[inline]
pub fn parse_line(line_bytes: &[u8], line_start: u32) -> FlatNode {
    let len = line_bytes.len();
    if len == 0 {
        return FlatNode::new(0, NodeKind::Blank, line_start, 0);
    }

    // Exclude trailing '\r' and '\n' from effective length.
    let mut eff_len = len;
    while eff_len > 0 && (line_bytes[eff_len - 1] == b'\r' || line_bytes[eff_len - 1] == b'\n') {
        eff_len -= 1;
    }

    if eff_len == 0 {
        return FlatNode::new(0, NodeKind::Blank, line_start, 0);
    }

    // Count consecutive prefix '|' characters (nesting depth).
    let mut depth = 0;
    while depth < eff_len && line_bytes[depth] == b'|' {
        depth += 1;
    }

    // Skip whitespace immediately following the prefix.
    let mut content_start = depth;
    while content_start < eff_len
        && (line_bytes[content_start] == b' ' || line_bytes[content_start] == b'\t')
    {
        content_start += 1;
    }

    let span_start = line_start + content_start as u32;
    let span_len = (eff_len - content_start) as u32;

    if span_len == 0 {
        return FlatNode::new(depth as u32, NodeKind::Blank, span_start, 0);
    }

    let rest = &line_bytes[content_start..eff_len];
    let kind = classify_line(rest);

    FlatNode::new(depth as u32, kind, span_start, span_len)
}

/// Classifies syntax kind from leading line tokens (tomet / oneline compatible).
#[inline]
fn classify_line(bytes: &[u8]) -> NodeKind {
    if bytes.is_empty() {
        return NodeKind::Blank;
    }

    match bytes[0] {
        b'=' => {
            // Heading: =[ Header ] or ==[ Subheader ]
            NodeKind::Heading
        }
        b'>' => {
            // Callout or Header: >[ Header ] or >> Content
            NodeKind::Callout
        }
        b'@' => {
            // Element: @card(...) or @memo[ ... ]
            NodeKind::Element
        }
        b'-' | b'*' => {
            // List: - Item or * Item
            if bytes.len() > 1 && (bytes[1] == b' ' || bytes[1] == b'\t') {
                NodeKind::List
            } else {
                NodeKind::Text
            }
        }
        b'`' => {
            // Fenced code block: ```
            if bytes.starts_with(b"```") {
                NodeKind::Raw
            } else {
                NodeKind::Text
            }
        }
        _ => {
            // Numbered list item check (e.g. "1. ")
            if bytes[0].is_ascii_digit() {
                let mut idx = 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                if idx < bytes.len()
                    && bytes[idx] == b'.'
                    && idx + 1 < bytes.len()
                    && bytes[idx + 1] == b' '
                {
                    return NodeKind::List;
                }
            }
            NodeKind::Text
        }
    }
}
