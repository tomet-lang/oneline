use anyhow::Result;
use oneline_core::NodeKind;

use crate::parse::ParsedNodes;
use crate::parser::GpuParser;
use crate::query::QueryResult;
use crate::util::check_gpu_len;

/// A document parsed once and kept in VRAM, so that many queries pay the
/// upload + scan + parse cost only once.
///
/// This is the configuration where the GPU can beat the CPU: a one-shot query is
/// bounded by PCIe, a resident one only by VRAM bandwidth.
pub struct GpuDocument {
    text_len: u32,
    line_count: u32,
    /// `None` for an empty document (no buffers to hold).
    resident: Option<Resident>,
}

struct Resident {
    parsed: ParsedNodes,
    // Kept so the buffers outlive every bind group that references them.
    _input: wgpu::Buffer,
    _line_offsets: wgpu::Buffer,
}

impl GpuDocument {
    pub fn line_count(&self) -> usize {
        self.line_count as usize
    }

    pub fn text_len(&self) -> usize {
        self.text_len as usize
    }
}

impl GpuParser {
    /// Uploads, scans and parses `input`, keeping the result in VRAM.
    pub fn upload(&self, input: &str) -> Result<GpuDocument> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(GpuDocument { text_len: 0, line_count: 0, resident: None });
        }
        let text_len = input.len() as u32;

        let input_buffer = self.upload_text(input)?;
        let (line_offsets, line_count) = self.scan_to_buffer(&input_buffer, text_len)?;
        let parsed = self.parse_to_buffer(&input_buffer, &line_offsets, line_count, text_len)?;

        Ok(GpuDocument {
            text_len,
            line_count,
            resident: Some(Resident { parsed, _input: input_buffer, _line_offsets: line_offsets }),
        })
    }

    /// Queries a resident document. Only the match count and the (at most
    /// `max_matches`) indices cross PCIe.
    pub fn query_document(
        &self,
        doc: &GpuDocument,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<QueryResult> {
        match &doc.resident {
            None => Ok(QueryResult { total_matches: 0, indices: Vec::new() }),
            Some(r) => self.run_query(
                &self.filter_pipeline,
                &r.parsed.bind_group,
                doc.line_count,
                kind,
                depth,
                max_matches,
            ),
        }
    }
}
