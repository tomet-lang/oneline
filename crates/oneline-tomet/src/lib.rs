pub mod compat;

use anyhow::Result;
use oneline_core::{FlatNode, FlatTree, NONE, NodeKind, TreeLinks};
use rayon::prelude::*;
use tomet_ast::{
    Block, Document, Inline, List, Paragraph, Section, Span, Text,
};

/// Converts FlatTree into a canonical tomet AST (tomet_ast::Document).
///
/// The parent/child structure comes from [`oneline_simd::build_links`] (parallel
/// ANSV over line depths), and the per-line blocks are built in parallel too; only
/// the final stitching of finished blocks into their parents is a sequential walk.
pub fn to_tomet_ast(tree: &FlatTree) -> Result<Document> {
    let links = oneline_simd::build_links(&tree.nodes);
    to_tomet_ast_with_links(tree, &links)
}

/// Like [`to_tomet_ast`] with precomputed links (e.g. when they are reused for queries).
pub fn to_tomet_ast_with_links(tree: &FlatTree, links: &TreeLinks) -> Result<Document> {
    assert_eq!(tree.nodes.len(), links.len(), "links do not belong to this tree");
    let n = tree.nodes.len();

    // tomet gives `|` lines a meaning only below an element, and a line with nothing on it ends that
    // content. So a `|` line is *literal text* unless its parent is an element that is itself real
    // and that no bare blank line separates from it.
    let (literal, parent) = literal_pipe_lines(tree, links);

    // Each line becomes one block on its own, so this is embarrassingly parallel.
    let mut blocks: Vec<Option<Block>> = tree
        .nodes
        .par_iter()
        .enumerate()
        .map(|(i, node)| {
            if matches!(node.kind(), NodeKind::Blank | NodeKind::Comment) {
                return None;
            }
            let raw_text = tree.content_of(node).trim();
            if raw_text.is_empty() {
                return None;
            }
            Some(if literal[i] { literal_block(tree, i) } else { parse_line_to_block(node, raw_text) })
        })
        .collect();

    // Children always have larger indices than their parent, so walking backwards
    // finishes every node after all of its children. `kids` collects children in
    // reverse order; they are put back in document order when a node is finished.
    let mut kids: Vec<Vec<Block>> = (0..n).map(|_| Vec::new()).collect();
    let mut roots: Vec<Block> = Vec::new();

    for i in (0..n).rev() {
        let mut children = std::mem::take(&mut kids[i]);
        children.reverse();
        let finished = match blocks[i].take() {
            Some(block) => finish(block, children),
            // A line without a block of its own: its children belong to its parent.
            None => children,
        };
        let sink = match parent[i] {
            NONE => &mut roots,
            p => &mut kids[p as usize],
        };
        sink.extend(finished.into_iter().rev());
    }
    roots.reverse();

    Ok(Document::new(nest_sections(roots), Span::dummy()))
}

/// `literal[i]`: line `i` starts with `|` but is not the content of an element, so tomet reads it as text.
/// Also returns the parents to stitch with: a line cut off from its element by a bare blank line
/// is no longer inside it (nor inside anything else, the blank ended every `|` context), so it
/// stands at the top level.
fn literal_pipe_lines(tree: &FlatTree, links: &TreeLinks) -> (Vec<bool>, Vec<u32>) {
    let n = tree.nodes.len();
    let mut literal = vec![false; n];
    let mut parent = links.parent.clone();
    // Index of the latest bare blank line (no `|` at all) before the current line.
    let mut last_blank: Option<usize> = None;
    for i in 0..n {
        let node = &tree.nodes[i];
        if node.kind() == NodeKind::Blank && node.depth == 0 {
            last_blank = Some(i);
            continue;
        }
        if node.depth == 0 {
            continue;
        }
        literal[i] = match links.parent[i] {
            NONE => true,
            p => {
                let p = p as usize;
                if last_blank.is_some_and(|b| b > p) {
                    parent[i] = NONE;
                    true
                } else {
                    tree.nodes[p].kind() != NodeKind::Element || literal[p]
                }
            }
        };
    }
    (literal, parent)
}

/// The block for a line that is plain text to tomet, whatever it looks like: the whole line, `|`
/// prefix included, goes through tomet's own inline parsing.
fn literal_block(tree: &FlatTree, i: usize) -> Block {
    let start = tree.line_offsets[i] as usize;
    let end = tree.line_offsets.get(i + 1).map_or(tree.source.len(), |&e| e as usize);
    let line = tree.source[start..end].trim_end_matches(['\r', '\n']);
    if let Ok(doc) = tomet_parser::parse_document(line) {
        if let Some(first_block) = doc.blocks.into_iter().next() {
            return first_block;
        }
    }
    Block::Paragraph(Paragraph::new(vec![Inline::Text(Text::new(line, Span::dummy()))], Span::dummy()))
}

/// Puts every block that follows a heading, up to the next heading of the same or a lower level,
/// into that heading's section. Applies to one block sequence (the document, or an element's content).
fn nest_sections(blocks: Vec<Block>) -> Vec<Block> {
    fn close_top(open: &mut Vec<Section>, out: &mut Vec<Block>) {
        let done = open.pop().expect("a section is open");
        match open.last_mut() {
            Some(parent) => parent.blocks.push(Block::Section(done)),
            None => out.push(Block::Section(done)),
        }
    }
    let mut out = Vec::new();
    let mut open: Vec<Section> = Vec::new();
    for block in blocks {
        match block {
            Block::Section(section) => {
                while open.last().is_some_and(|top| top.level >= section.level) {
                    close_top(&mut open, &mut out);
                }
                open.push(section);
            }
            other => match open.last_mut() {
                Some(top) => top.blocks.push(other),
                None => out.push(other),
            },
        }
    }
    while !open.is_empty() {
        close_top(&mut open, &mut out);
    }
    out
}

/// Parses a single line's content and NodeKind into an initial Block.
fn parse_line_to_block(node: &FlatNode, text: &str) -> Block {
    match node.kind() {
        NodeKind::Heading => {
            // "=[ Header ]", "== Header": the level is the number of `=`, the title is parsed by tomet.
            if let Ok(doc) = tomet_parser::parse_document(text) {
                if let Some(first_block @ Block::Section(_)) = doc.blocks.into_iter().next() {
                    return first_block;
                }
            }
            let level = text.chars().take_while(|&c| c == '=').count().max(1);
            let title_text = text.trim_start_matches('=').trim_start_matches('[').trim_end_matches(']').trim();
            let inline = Inline::Text(Text::new(title_text, Span::dummy()));
            Block::Section(Section::new(level, vec![inline], Span::dummy()))
        }
        NodeKind::Element => {
            // Use tomet-parser to parse element syntax like @card(title: "...").
            if let Ok(doc) = tomet_parser::parse_document(text) {
                if let Some(first_block) = doc.blocks.into_iter().next() {
                    return first_block;
                }
            }
            // Fallback on parse failure.
            Block::Paragraph(Paragraph::new(
                vec![Inline::Text(Text::new(text, Span::dummy()))],
                Span::dummy(),
            ))
        }
        NodeKind::List | NodeKind::OrderedList | NodeKind::Connect => {
            // Use tomet-parser to parse the list item or connect line.
            if let Ok(doc) = tomet_parser::parse_document(text) {
                if let Some(first_block) = doc.blocks.into_iter().next() {
                    return first_block;
                }
            }
            Block::Paragraph(Paragraph::new(
                vec![Inline::Text(Text::new(text, Span::dummy()))],
                Span::dummy(),
            ))
        }
        _ => {
            // Regular prose text.
            Block::Paragraph(Paragraph::new(
                vec![Inline::Text(Text::new(text, Span::dummy()))],
                Span::dummy(),
            ))
        }
    }
}

/// Attaches `children` (in document order) to `block`.
///
/// Returns the finished block followed by any children the block's AST type cannot
/// hold. Those are *hoisted* to become following siblings: their content is kept
/// (an earlier version silently dropped them), only their nesting is flattened.
fn finish(block: Block, children: Vec<Block>) -> Vec<Block> {
    if children.is_empty() {
        return vec![block];
    }
    match block {
        Block::Section(mut section) => {
            section.blocks = children;
            vec![Block::Section(section)]
        }
        Block::Element(mut element) => {
            element.content = Some(nest_sections(children));
            vec![Block::Element(element)]
        }
        Block::Paragraph(mut paragraph) => {
            // Consecutive deeper prose continues the paragraph as a soft break.
            let mut hoisted = Vec::new();
            for child in children {
                match child {
                    Block::Paragraph(child_p) => {
                        paragraph
                            .content
                            .push(Inline::SoftBreak(tomet_ast::SoftBreak { span: Span::dummy() }));
                        paragraph.content.extend(child_p.content);
                    }
                    other => hoisted.push(other),
                }
            }
            let mut out = vec![Block::Paragraph(paragraph)];
            out.extend(hoisted);
            out
        }
        Block::List(mut list) => {
            // A deeper list becomes the sublist of the last item; anything else
            // cannot live inside a list item and is hoisted.
            let mut hoisted = Vec::new();
            for child in children {
                match (child, list.items.last_mut()) {
                    (Block::List(sub), Some(item)) => attach_sublist(&mut item.sublist, sub),
                    (other, _) => hoisted.push(other),
                }
            }
            let mut out = vec![Block::List(list)];
            out.extend(hoisted);
            out
        }
    }
}

fn attach_sublist(slot: &mut Option<List>, sub: List) {
    match slot {
        Some(existing) => existing.items.extend(sub.items),
        None => *slot = Some(sub),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_tomet_ast_basic() {
        let text = "@card(title:\"tomet-shelf\")\n| Lightweight and fast\n";
        let tree = oneline_simd::parse(text);
        let doc = to_tomet_ast(&tree).unwrap();

        assert_eq!(doc.blocks.len(), 1);
        if let Block::Element(el) = &doc.blocks[0] {
            assert_eq!(el.sigil.name().unwrap().name, "card");
            assert!(el.content.is_some());
            let content_blocks = el.content.as_ref().unwrap();
            assert_eq!(content_blocks.len(), 1);
        } else {
            panic!("Expected Element block, got {:?}", doc.blocks[0]);
        }
    }

    #[test]
    fn children_of_leaves_are_kept_not_dropped() {
        // A paragraph with a deeper list and a deeper element beneath it, and a
        // list item with a deeper paragraph: the old builder lost all of these.
        let doc = "@note\n| lead prose\n| @card(title: \"kept\")\n| | - nested item\n| - item\n| | trailing detail\n";
        let tree = oneline_simd::parse(doc);
        let ast = to_tomet_ast(&tree).unwrap();
        let dump = format!("{ast:?}");
        for needle in ["lead prose", "nested item", "kept", "item", "trailing detail"] {
            assert!(dump.contains(needle), "{needle:?} was lost:\n{dump}");
        }
    }

}


#[cfg(test)]
mod differential {
    use super::compat::{Outcome, compare};
    use oneline_core::datagen::Rng;

    /// Documents built only from constructs oneline claims to read the way tomet does: headings of any
    /// level, elements with `|` content (nested too), single-line paragraphs and list items, `//`
    /// comments, all separated by blank lines where tomet would otherwise join or split differently.
    fn supported_doc(rng: &mut Rng, blocks: usize) -> String {
        let mut out = String::new();
        for _ in 0..blocks {
            match rng.below(11) {
                0 => {
                    let marks = "=".repeat(1 + rng.below(4) as usize);
                    let n = rng.below(100);
                    if rng.chance(30) {
                        out.push_str(&format!("{marks}[ Heading {n} ]\n"));
                    } else {
                        out.push_str(&format!("{marks} Heading {n}\n"));
                    }
                }
                1 => out.push_str(&format!("plain paragraph {}\n", rng.below(1000))),
                2 => out.push_str(&format!("- item {}\n", rng.below(1000))),
                3 => out.push_str(&format!("-. step {}\n", rng.below(1000))),
                4 => out.push_str(&format!("// comment {}\n", rng.below(1000))),
                5 => {
                    out.push_str("@card(title: \"t\")\n");
                    for _ in 0..rng.below(3) {
                        out.push_str(&format!("| child text {}\n|\n", rng.below(100)));
                    }
                }
                6 => {
                    out.push_str("@outer\n| @inner\n| | deep text\n|\n| after inner\n");
                }
                7 => out.push_str("@solo(a: 1)\n"),
                // `|` lines that are not the content of an element are plain text to tomet.
                8 => out.push_str(&format!("= Under {}\n| pipe under a heading\n", rng.below(10))),
                9 => out.push_str("- an item\n| pipe under a list item\n"),
                _ => out.push_str("@card(title: \"c\")\n| inside\n\n| after a blank line\n"),
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn matches_tomet_on_supported_documents() {
        let mut rng = Rng::new(2024);
        for round in 0..300 {
            let doc = supported_doc(&mut rng, 1 + (round % 12));
            match compare(&doc) {
                Outcome::Same => {}
                Outcome::ReferenceError(e) => panic!("round {round}: tomet rejects the generated document: {e}\n{doc}"),
                Outcome::Different { real, ours } => {
                    panic!("round {round}: trees differ\n{doc}\ntomet: {real}\nours:  {ours}")
                }
            }
        }
    }
}
