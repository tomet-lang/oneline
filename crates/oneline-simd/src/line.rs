use oneline_core::{FlatNode, NodeKind};

/// Parses a single line byte slice into a FlatNode.
///
/// A line is `|`* (depth), blanks, then content whose first byte decides the kind.
/// Trailing `\r` / `\n` are not part of the content.
#[inline]
pub fn parse_line(line: &[u8], line_start: u32) -> FlatNode {
    // Exclude trailing '\r' and '\n' (a run of them: "a\r\r\n" ends in three).
    let mut eff = line.len();
    while eff > 0 && matches!(line[eff - 1], b'\r' | b'\n') {
        eff -= 1;
    }
    if eff == 0 {
        return FlatNode::new(0, NodeKind::Blank, line_start, 0);
    }
    let line = &line[..eff];

    // Depth: `|` prefixes, which blanks may separate (`| | x` is as deep as `|| x`). The blank run
    // found after the last `|` is the one that precedes the content, so it is counted once.
    let blank = |b: &u8| *b == b' ' || *b == b'\t';
    let mut depth = line.iter().take_while(|&&b| b == b'|').count();
    let mut pos = depth;
    let mut blanks = line[pos..].iter().take_while(|b| blank(b)).count();
    while depth > 0 && line.get(pos + blanks) == Some(&b'|') {
        pos += blanks;
        let run = line[pos..].iter().take_while(|&&b| b == b'|').count();
        depth += run;
        pos += run;
        blanks = line[pos..].iter().take_while(|b| blank(b)).count();
    }
    let content = &line[pos + blanks..];

    let span_start = line_start + (pos + blanks) as u32;
    if content.is_empty() {
        return FlatNode::new(depth as u32, NodeKind::Blank, span_start, 0);
    }
    FlatNode::new(
        depth as u32,
        classify_line(content),
        span_start,
        content.len() as u32,
    )
}

/// What a line's first byte says about it. `0` = plain text.
const FIRST_BYTE: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b'=' as usize] = 1;
    t[b':' as usize] = 2;
    t[b'@' as usize] = 3;
    t[b'-' as usize] = 4;
    t[b'/' as usize] = 5;
    t
};

/// Classifies the line by its first byte (tomet syntax).
#[inline]
fn classify_line(bytes: &[u8]) -> NodeKind {
    // One table load and (for the common plain-text line) one predictable branch.
    match FIRST_BYTE[bytes[0] as usize] {
        0 => NodeKind::Text,
        // Section heading: `=[ title ]` or `= title`
        1 => NodeKind::Heading,
        // Connect: `:rule(..)`, `:{..}`, attached to the element before it
        2 => NodeKind::Connect,
        // Element: `@card(..)`
        3 => NodeKind::Element,
        // Comment: `// ...` (the whole line)
        5 => match bytes.get(1) {
            Some(b'/') => NodeKind::Comment,
            _ => NodeKind::Text,
        },
        // List item: `- item`, or `-. item` (ordered)
        _ => match bytes.get(1) {
            Some(b' ' | b'\t') => NodeKind::List,
            Some(b'.') if matches!(bytes.get(2), Some(b' ' | b'\t')) => NodeKind::OrderedList,
            _ => NodeKind::Text,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oneline_core::datagen::Rng;

    /// The previous byte-loop implementation, kept as the oracle.
    fn reference(line_bytes: &[u8], line_start: u32) -> FlatNode {
        let len = line_bytes.len();
        if len == 0 {
            return FlatNode::new(0, NodeKind::Blank, line_start, 0);
        }
        let mut eff_len = len;
        while eff_len > 0 && (line_bytes[eff_len - 1] == b'\r' || line_bytes[eff_len - 1] == b'\n')
        {
            eff_len -= 1;
        }
        if eff_len == 0 {
            return FlatNode::new(0, NodeKind::Blank, line_start, 0);
        }
        // Written the other way round from `parse_line`: scan byte by byte, remembering where
        // the last `|` was; a blank run only belongs to the prefix if another `|` follows it.
        let mut depth = 0;
        let mut prefix_end = 0;
        let mut i = 0;
        while i < eff_len {
            match line_bytes[i] {
                b'|' => {
                    depth += 1;
                    i += 1;
                    prefix_end = i;
                }
                b' ' | b'\t' if depth > 0 => i += 1,
                _ => break,
            }
        }
        let mut content_start = prefix_end;
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
        let kind = match rest[0] {
            b'=' => NodeKind::Heading,
            b':' => NodeKind::Connect,
            b'@' => NodeKind::Element,
            b'-' if rest.len() > 1 && (rest[1] == b' ' || rest[1] == b'\t') => NodeKind::List,
            b'-' if rest.len() > 2 && rest[1] == b'.' && (rest[2] == b' ' || rest[2] == b'\t') => {
                NodeKind::OrderedList
            }
            b'/' if rest.len() > 1 && rest[1] == b'/' => NodeKind::Comment,
            _ => NodeKind::Text,
        };
        FlatNode::new(depth as u32, kind, span_start, span_len)
    }

    #[test]
    fn matches_reference_on_random_lines() {
        // A small alphabet makes every structural case (prefixes, blanks, markers, digits,
        // CR/LF runs, backtick runs) occur constantly, including at awkward lengths.
        let alphabet: &[u8] = b"|| \t=:@->*`0123456789.\r\nxy";
        let mut rng = Rng::new(4242);
        for round in 0..300_000 {
            let len = rng.below(14) as usize;
            let line: Vec<u8> = (0..len)
                .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
                .collect();
            let start = rng.below(1000) as u32;
            assert_eq!(
                parse_line(&line, start),
                reference(&line, start),
                "round {round}: {line:?}"
            );
        }
    }

    #[test]
    fn matches_reference_on_all_short_lines() {
        // Exhaustive over a 7-symbol alphabet up to length 6.
        let alphabet = *b"| -.:@/\n";
        fn go(prefix: &mut Vec<u8>, alphabet: &[u8], depth: usize) {
            assert_eq!(parse_line(prefix, 7), reference(prefix, 7), "{prefix:?}");
            if depth == 0 {
                return;
            }
            for &b in alphabet {
                prefix.push(b);
                go(prefix, alphabet, depth - 1);
                prefix.pop();
            }
        }
        go(&mut Vec::new(), &alphabet, 6);
    }
}
