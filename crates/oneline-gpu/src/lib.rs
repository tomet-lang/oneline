//! GPU line parser and query engine (wgpu compute shaders).
//!
//! Pipeline, all inside VRAM:
//!
//! 1. **scan** (`scan.rs`, `scan.wgsl`): 3-pass parallel prefix scan that turns
//!    newline positions into a dense `line_offsets` array.
//! 2. **parse** (`parse.rs`): one thread per line writes a 16-byte [`oneline_core::FlatNode`].
//! 3. **query** (`query.rs`): filter by kind/depth, either fused with the parse
//!    (one-shot `query_raw`) or over nodes kept resident in a [`GpuDocument`].

use bytemuck::{Pod, Zeroable};

mod document;
mod parse;
mod parser;
mod query;
mod scan;
#[cfg(test)]
mod tests;
mod util;

pub use document::GpuDocument;
pub use parser::GpuParser;
pub use query::QueryResult;
pub use util::MAX_GPU_INPUT_LEN;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ShaderConfig {
    pub line_count: u32,
    pub text_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScanConfig {
    pub num_blocks: u32,
    pub text_len: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct QueryConfig {
    pub target_kind: u32,
    pub target_depth: u32,
    pub max_matches: u32,
    pub _pad: u32,
}
