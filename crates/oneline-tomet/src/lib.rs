use anyhow::Result;
use oneline_core::{FlatNode, FlatTree, NodeKind};
use tomet_ast::{
    Block, Document, Element, Inline, Name, Paragraph, Section, Sigil, Span, Text,
};

/// Converts FlatTree into a canonical tomet AST (tomet_ast::Document).
pub fn to_tomet_ast(tree: &FlatTree) -> Result<Document> {
    let mut root_blocks: Vec<Block> = Vec::new();
    let mut block_stack: Vec<(u32, BlockCandidate)> = Vec::new();

    for node in &tree.nodes {
        if node.kind() == NodeKind::Blank {
            continue;
        }

        let raw_text = tree.content_of(node).trim();
        if raw_text.is_empty() {
            continue;
        }

        let block = parse_line_to_block(node, raw_text);

        // Adjust stack depth:
        // Pop blocks at or deeper than the current node's depth and attach to parent.
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

        // Push new block candidate onto the stack.
        block_stack.push((node.depth, BlockCandidate::new(block)));
    }

    // Unwind all remaining blocks on the stack.
    while let Some((_, finished)) = block_stack.pop() {
        let finished_block = finished.into_block();
        if let Some((_, parent)) = block_stack.last_mut() {
            parent.add_child(finished_block);
        } else {
            root_blocks.push(finished_block);
        }
    }

    Ok(Document::new(root_blocks, Span::dummy()))
}

/// Parses a single line's content and NodeKind into an initial Block.
fn parse_line_to_block(node: &FlatNode, text: &str) -> Block {
    match node.kind() {
        NodeKind::Heading => {
            // "=[ Header ]" or ">[ Header ]"
            let title_text = text
                .trim_start_matches('=')
                .trim_start_matches('>')
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
        NodeKind::Callout => {
            // ">[ Callout ]" -> construct @callout element.
            let callout_title = text
                .trim_start_matches('>')
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim();
            let mut el = Element::default();
            el.sigil = Sigil::Named(Name::bare("callout"));
            if !callout_title.is_empty() {
                el.args = Some(tomet_ast::Value::String(callout_title.to_string()));
            }
            Block::Element(el)
        }
        NodeKind::List => {
            // Use tomet-parser to parse list item.
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

/// Helper block candidate wrapper to accumulate nested children.
enum BlockCandidate {
    Section {
        section: Section,
        children: Vec<Block>,
    },
    Element {
        element: Element,
        children: Vec<Block>,
    },
    Other(Block),
}

impl BlockCandidate {
    fn new(block: Block) -> Self {
        match block {
            Block::Section(s) => Self::Section {
                section: s,
                children: Vec::new(),
            },
            Block::Element(e) => Self::Element {
                element: e,
                children: Vec::new(),
            },
            other => Self::Other(other),
        }
    }

    fn add_child(&mut self, child: Block) {
        match self {
            Self::Section { children, .. } => children.push(child),
            Self::Element { children, .. } => children.push(child),
            Self::Other(Block::Paragraph(p)) => {
                // If consecutive text follows a paragraph, append as softbreak.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
