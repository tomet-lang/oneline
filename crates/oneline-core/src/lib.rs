use bytemuck::{Pod, Zeroable};

pub mod datagen;
mod links;

pub use links::{NONE, TreeLinks};

/// Largest input (in bytes) the `u32` offsets of [`FlatNode`] can address.
pub const MAX_INPUT_LEN: usize = u32::MAX as usize;

/// Input does not fit the 32-bit offsets used by [`FlatNode`] / the GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputTooLarge {
    pub len: usize,
}

impl std::fmt::Display for InputTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input is {} bytes but 32-bit offsets address at most {} bytes",
            self.len, MAX_INPUT_LEN
        )
    }
}

impl std::error::Error for InputTooLarge {}

/// Fails when `len` cannot be addressed by `u32` offsets.
pub const fn check_input_len(len: usize) -> Result<(), InputTooLarge> {
    if len > MAX_INPUT_LEN {
        Err(InputTooLarge { len })
    } else {
        Ok(())
    }
}

/// Node kind identifiers synchronized with GPU Compute Shader definitions.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Blank = 0,
    Text = 1,
    Heading = 2,
    Element = 3,
    List = 4,
    Connect = 5,
    OrderedList = 6,
    Comment = 7,
    Unknown = 99,
}

impl NodeKind {
    pub const fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Blank,
            1 => Self::Text,
            2 => Self::Heading,
            3 => Self::Element,
            4 => Self::List,
            5 => Self::Connect,
            6 => Self::OrderedList,
            7 => Self::Comment,
            _ => Self::Unknown,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Blank => "blank",
            Self::Text => "text",
            Self::Heading => "heading",
            Self::Element => "element",
            Self::List => "list",
            Self::Connect => "connect",
            Self::OrderedList => "ordered-list",
            Self::Comment => "comment",
            Self::Unknown => "unknown",
        }
    }
}

/// Node representation with 1:1 memory layout matching the WGSL Compute Shader.
/// Fixed 16 bytes: depth(4B) + kind(4B) + span_start(4B) + span_len(4B).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Pod, Zeroable)]
pub struct FlatNode {
    /// Nesting depth specified by leading prefixes ('|' etc.).
    pub depth: u32,
    /// Kind of node (numeric value of NodeKind).
    pub kind: u32,
    /// Byte offset of the content start (after leading prefix and whitespace).
    pub span_start: u32,
    /// Byte length of the content.
    pub span_len: u32,
}

impl FlatNode {
    pub const fn new(depth: u32, kind: NodeKind, span_start: u32, span_len: u32) -> Self {
        Self {
            depth,
            kind: kind as u32,
            span_start,
            span_len,
        }
    }

    pub fn kind(&self) -> NodeKind {
        NodeKind::from_u32(self.kind)
    }
}

/// Flat tree holding parse results in a Structure of Arrays (SoA) layout.
///
/// The source text is borrowed, not copied: parsing never duplicates the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatTree<'a> {
    /// Original source text.
    pub source: &'a str,
    /// Array of flat nodes corresponding to each line.
    pub nodes: Vec<FlatNode>,
    /// Array of byte offsets pointing to the start of each line.
    pub line_offsets: Vec<u32>,
}

impl<'a> FlatTree<'a> {
    pub fn new(source: &'a str, nodes: Vec<FlatNode>, line_offsets: Vec<u32>) -> Self {
        Self {
            source,
            nodes,
            line_offsets,
        }
    }

    /// Retrieve the text slice pointed to by the node.
    pub fn content_of(&self, node: &FlatNode) -> &'a str {
        let start = node.span_start as usize;
        let end = start + node.span_len as usize;
        self.source.get(start..end).unwrap_or("")
    }

    /// Dump human-readable indented tree representation.
    pub fn dump_tree(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            let indent = "  ".repeat(node.depth as usize);
            let text = self.content_of(node);
            out.push_str(&format!(
                "{}[d:{} {:?}] \"{}\"\n",
                indent,
                node.depth,
                node.kind(),
                text.trim()
            ));
        }
        out
    }
}
