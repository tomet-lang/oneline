//! Differential check against tomet's own parser.
//!
//! oneline is meant to be a subset of tomet: a document it reads must come out as the same tree
//! `tomet_parser` produces. [`compare`] parses a text both ways and says whether the two
//! [`Document`]s are equal (`Span` equality always holds, so only structure and content count).
//! The rest groups a corpus into small units and into line shapes, so that disagreements can be
//! counted and looked at.

use tomet_ast::Document;

/// Result of parsing one text with tomet's parser and with oneline.
pub enum Outcome {
    Same,
    /// Both parsed, trees differ. The strings are the `Debug` forms with spans and empty fields removed.
    Different {
        real: String,
        ours: String,
    },
    /// tomet's own parser rejects the text, so there is nothing to compare against.
    ReferenceError(String),
}

pub fn compare(text: &str) -> Outcome {
    let real: Document = match tomet_parser::parse_document(text) {
        Ok(doc) => doc,
        Err(e) => return Outcome::ReferenceError(format!("{e:?}")),
    };
    let tree = oneline_simd::parse(text);
    let ours = match crate::to_tomet_ast(&tree) {
        Ok(doc) => doc,
        Err(e) => {
            return Outcome::Different {
                real: tidy(&format!("{real:?}")),
                ours: format!("conversion failed: {e}"),
            };
        }
    };
    if real == ours {
        Outcome::Same
    } else {
        Outcome::Different {
            real: tidy(&format!("{real:?}")),
            ours: tidy(&format!("{ours:?}")),
        }
    }
}

/// Removes `span: Span { .. }` and the fields that are empty, which makes a `Debug` dump readable.
pub fn tidy(debug: &str) -> String {
    let mut out = String::with_capacity(debug.len());
    let mut rest = debug;
    const SPAN: &str = "span: Span { start: ";
    while let Some(at) = rest.find(SPAN) {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        match tail.find('}') {
            Some(end) => rest = tail[end + 1..].trim_start_matches([',', ' ']),
            None => break,
        }
    }
    out.push_str(rest);
    for empty in [
        "args: None, ",
        "value: None, ",
        "id: None, ",
        "connects: [], ",
        "content: None, ",
        "sublist: None, ",
    ] {
        out = out.replace(empty, "");
    }
    out
}

/// Runs of non-blank lines (each keeps its line ends). Blank lines only separate; a tomet block
/// that contains a blank line is cut in two, which the comparison then reports as a disagreement.
pub fn chunks(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = None;
    let mut pos = 0;
    for line in text.split_inclusive('\n') {
        let blank = line.trim().is_empty();
        match (blank, start) {
            (false, None) => start = Some(pos),
            (true, Some(s)) => {
                out.push(&text[s..pos]);
                start = None;
            }
            _ => {}
        }
        pos += line.len();
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

/// What the first line of a chunk looks like, for grouping disagreements.
pub fn shape(chunk: &str) -> &'static str {
    let line = chunk.lines().next().unwrap_or("");
    if line.starts_with([' ', '\t']) {
        return "indented first line";
    }
    let t = line.trim_start_matches(['|', ' ']);
    match () {
        _ if line.starts_with("|[") => "table row |[",
        _ if line.starts_with('|') => "pipe line |",
        _ if t.starts_with("```") => "fence ```",
        _ if t.starts_with("-. ") => "ordered -.",
        _ if t.starts_with("- ") => "list -",
        _ if t.starts_with('=') => "heading =",
        _ if t.starts_with('@') => "element @",
        _ if t.starts_with(':') => "connect :",
        _ if t.starts_with('#') => "# heading / tag",
        _ if t.starts_with("//") || t.starts_with("/*") => "comment",
        _ if t.starts_with(';') => "break ;",
        _ => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_split_on_blank_lines_and_keep_everything() {
        let text = "a\nb\n\n\nc\n  \nd";
        let parts = chunks(text);
        assert_eq!(parts, vec!["a\nb\n", "c\n", "d"]);
        assert_eq!(chunks(""), Vec::<&str>::new());
        assert_eq!(chunks("\n\n"), Vec::<&str>::new());
    }

    #[test]
    fn tidy_drops_spans_and_empty_fields() {
        let dbg = "Element { sigil: Bare, args: None, span: Span { start: 3, end: 9 }, value: None, id: None }";
        assert_eq!(tidy(dbg), "Element { sigil: Bare, id: None }");
    }

    #[test]
    fn plain_element_with_pipe_children_is_the_same_tree() {
        assert!(matches!(
            compare("@card(title: \"x\")\n| inner text\n"),
            Outcome::Same
        ));
    }

    #[test]
    fn heading_level_is_the_equals_count() {
        assert!(matches!(compare("== Sub\n"), Outcome::Same));
        assert!(matches!(compare("= a\n=== c\n== b\n"), Outcome::Same));
    }

    #[test]
    fn shapes() {
        assert_eq!(shape("|[ a ][ b ]\n"), "table row |[");
        assert_eq!(shape("== x\n"), "heading =");
        assert_eq!(shape("  - sub\n"), "indented first line");
        assert_eq!(shape("hello\n"), "text");
    }
}

/// Why a chunk is not part of the line-local subset, or `None` if it is.
///
/// This is the definition of the subset in executable form; the harness reports how much of a real
/// corpus it covers (coverage) and whether the covered part comes out identical (fidelity).
pub fn outside_subset(chunk: &str) -> Option<&'static str> {
    for line in chunk.lines() {
        let t = line.trim_start_matches(['|', ' ', '\t']);
        if t.starts_with("```") {
            return Some("fence ```");
        }
        if t.starts_with("-|") {
            return Some("multiline list -|");
        }
        if t.starts_with("/*") {
            return Some("block comment /* */");
        }
        // Other list markers tomet accepts: `-[ x ]`, `-(x) c`, `-{..}`, `-@name(..)`, `-.[`..
        if let Some(after) = t.strip_prefix('-') {
            let after = after.strip_prefix('.').unwrap_or(after);
            if after.starts_with(['[', '(', '{', '@']) {
                return Some("list marker with group / combine");
            }
        }
        // An element, heading, list item, connect or pipe line whose brackets do not close on the line.
        let starts_block =
            line.starts_with('|') || matches!(t.chars().next(), Some('@' | '=' | '-' | ':'));
        if starts_block && bracket_delta(line) > 0 {
            return Some("multi-line brackets");
        }
    }
    None
}

/// Opening minus closing brackets on one line, ignoring double-quoted strings.
fn bracket_delta(line: &str) -> i32 {
    let (mut delta, mut in_str, mut escaped) = (0, false, false);
    for c in line.chars() {
        if in_str {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' | '[' | '{' => delta += 1,
            ')' | ']' | '}' => delta -= 1,
            _ => {}
        }
    }
    delta
}

#[cfg(test)]
mod subset_tests {
    use super::*;

    #[test]
    fn subset_detector() {
        assert_eq!(outside_subset("= title\ntext\n"), None);
        assert_eq!(outside_subset("@card(title: \"a)\")\n| x\n"), None);
        assert_eq!(
            outside_subset("@meta(\n  a: 1\n)\n"),
            Some("multi-line brackets")
        );
        assert_eq!(outside_subset("```\ncode\n```\n"), Some("fence ```"));
        assert_eq!(outside_subset("-| a\n | b\n"), Some("multiline list -|"));
        assert_eq!(
            outside_subset("-@x(a)\n"),
            Some("list marker with group / combine")
        );
        assert_eq!(outside_subset("- a\n- b\n"), None);
        assert_eq!(outside_subset("-. a\n"), None);
        // Prose with an unbalanced bracket does not open a region.
        assert_eq!(outside_subset("a smile :(\nnext\n"), None);
    }
}
