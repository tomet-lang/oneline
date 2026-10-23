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

    // Each line becomes one block on its own, so this is embarrassingly parallel.
    let mut blocks: Vec<Option<Block>> = tree
        .nodes
        .par_iter()
        .map(|node| {
            if node.kind() == NodeKind::Blank {
                return None;
            }
            let raw_text = tree.content_of(node).trim();
            if raw_text.is_empty() {
                return None;
            }
            Some(parse_line_to_block(node, raw_text))
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
        let sink = match links.parent[i] {
            NONE => &mut roots,
            p => &mut kids[p as usize],
        };
        sink.extend(finished.into_iter().rev());
    }
    roots.reverse();

    Ok(Document::new(roots, Span::dummy()))
}

/// Parses a single line's content and NodeKind into an initial Block.
fn parse_line_to_block(node: &FlatNode, text: &str) -> Block {
    match node.kind() {
        NodeKind::Heading => {
            // "=[ Header ]" or "= Header"
            let title_text = text
                .trim_start_matches('=')
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim();
            let inline = Inline::Text(Text::new(title_text, Span::dummy()));
            Block::Section(Section::new(node.depth.max(1) as usize, vec![inline], Span::dummy()))
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
        NodeKind::List | NodeKind::Connect => {
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
            element.content = Some(children);
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
    use oneline_core::datagen::Rng;
    use tomet_ast::Element;

    /// The previous stack-based builder, kept as an oracle for documents whose
    /// structure it handled without dropping anything.
    mod reference {
        use super::*;

        pub fn to_tomet_ast(tree: &FlatTree) -> Document {
            let mut root_blocks: Vec<Block> = Vec::new();
            let mut block_stack: Vec<(u32, Candidate)> = Vec::new();

            for node in &tree.nodes {
                if node.kind() == NodeKind::Blank {
                    continue;
                }
                let raw_text = tree.content_of(node).trim();
                if raw_text.is_empty() {
                    continue;
                }
                let block = parse_line_to_block(node, raw_text);

                while let Some((stack_depth, _)) = block_stack.last() {
                    if *stack_depth >= node.depth {
                        let (_, finished) = block_stack.pop().unwrap();
                        let finished_block = finished.into_block();
                        if let Some((_, parent)) = block_stack.last_mut() {
                            parent.add_child(finished_block);
                        } else {
                            root_blocks.push(finished_block);
                        }
                    } else {
                        break;
                    }
                }
                block_stack.push((node.depth, Candidate::new(block)));
            }
            while let Some((_, finished)) = block_stack.pop() {
                let finished_block = finished.into_block();
                if let Some((_, parent)) = block_stack.last_mut() {
                    parent.add_child(finished_block);
                } else {
                    root_blocks.push(finished_block);
                }
            }
            Document::new(root_blocks, Span::dummy())
        }

        enum Candidate {
            Section { section: Section, children: Vec<Block> },
            Element { element: Element, children: Vec<Block> },
            Other(Block),
        }

        impl Candidate {
            fn new(block: Block) -> Self {
                match block {
                    Block::Section(s) => Self::Section { section: s, children: Vec::new() },
                    Block::Element(e) => Self::Element { element: e, children: Vec::new() },
                    other => Self::Other(other),
                }
            }
            fn add_child(&mut self, child: Block) {
                match self {
                    Self::Section { children, .. } => children.push(child),
                    Self::Element { children, .. } => children.push(child),
                    Self::Other(Block::Paragraph(p)) => {
                        if let Block::Paragraph(child_p) = child {
                            p.content.push(Inline::SoftBreak(tomet_ast::SoftBreak { span: Span::dummy() }));
                            p.content.extend(child_p.content);
                        }
                    }
                    _ => {}
                }
            }
            fn into_block(self) -> Block {
                match self {
                    Self::Section { mut section, children } => {
                        section.blocks = children;
                        Block::Section(section)
                    }
                    Self::Element { mut element, children } => {
                        if !children.is_empty() {
                            element.content = Some(children);
                        }
                        Block::Element(element)
                    }
                    Self::Other(b) => b,
                }
            }
        }
    }

    #[test]
    fn test_to_tomet_ast_basic() {
        let text = "| @card(title:\"tomet-shelf\")\n|| Lightweight and fast\n";
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

    /// Random documents in which only containers (heading / element)
    /// have children — the shape the old builder handled losslessly.
    fn container_only_doc(rng: &mut Rng, lines: usize) -> String {
        let mut out = String::new();
        // (depth, is_container) of the currently open lines.
        let mut open: Vec<(u32, bool)> = Vec::new();
        for _ in 0..lines {
            let (text, container) = match rng.below(7) {
                0 => ("=[ Title ]", true),
                1 => ("@note(title: \"n\")", true),
                2 => ("@card(title: \"t\")", true),
                3 => ("- item", false),
                4 => ("plain prose", false),
                5 => ("", false), // blank: not part of the tree, must not touch `open`
                _ => ("- numbered", false),
            };
            // Pick a depth whose parent (nearest shallower open line) is a container.
            let depth = loop {
                let d = rng.below(5) as u32;
                match open.iter().rev().find(|&&(od, _)| od < d) {
                    Some(&(_, false)) if !text.is_empty() => continue,
                    _ => break d,
                }
            };
            if !text.is_empty() {
                while open.last().is_some_and(|&(od, _)| od >= depth) {
                    open.pop();
                }
            }
            for _ in 0..depth {
                out.push('|');
            }
            if depth > 0 {
                out.push(' ');
            }
            out.push_str(text);
            out.push('\n');
            if !text.is_empty() {
                open.push((depth, container));
            }
        }
        out
    }

    #[test]
    fn matches_previous_builder_on_container_documents() {
        let mut rng = Rng::new(77);
        for round in 0..60 {
            let doc = container_only_doc(&mut rng, 5 + round * 7);
            let tree = oneline_simd::parse(&doc);
            let want = reference::to_tomet_ast(&tree);
            let got = to_tomet_ast(&tree).unwrap();
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "round {round}\n{doc}");
        }
    }

    #[test]
    fn children_of_leaves_are_kept_not_dropped() {
        // A paragraph with a deeper list and a deeper element beneath it, and a
        // list item with a deeper paragraph: the old builder lost all of these.
        let doc = "| lead prose\n|| - nested item\n|| @card(title: \"kept\")\n| - item\n|| trailing detail\n";
        let tree = oneline_simd::parse(doc);
        let ast = to_tomet_ast(&tree).unwrap();
        let dump = format!("{ast:?}");
        for needle in ["lead prose", "nested item", "kept", "item", "trailing detail"] {
            assert!(dump.contains(needle), "{needle:?} was lost:\n{dump}");
        }
    }

    #[test]
    fn deeper_list_becomes_sublist() {
        let doc = "| - parent\n|| - child\n|| - child 2\n";
        let tree = oneline_simd::parse(doc);
        let ast = to_tomet_ast(&tree).unwrap();
        assert_eq!(ast.blocks.len(), 1, "{ast:?}");
        let Block::List(list) = &ast.blocks[0] else { panic!("{:?}", ast.blocks[0]) };
        let sub = list.items[0].sublist.as_ref().expect("sublist");
        assert_eq!(sub.items.len(), 2);
    }
}

