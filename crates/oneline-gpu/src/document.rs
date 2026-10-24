use anyhow::Result;
use oneline_core::NodeKind;

use crate::parse::ParsedNodes;
use crate::parser::GpuParser;
use crate::query::QueryResult;
use crate::search::sample_byte_histogram;
use crate::timing::{Profile, StageTimer, mark};
use crate::util::check_gpu_len;

/// A document parsed once and kept in VRAM, so that many queries pay the
/// upload + scan + parse cost only once.
///
/// This is the configuration where the GPU can beat the CPU: a one-shot query is
/// bounded by PCIe, a resident one only by VRAM bandwidth.
pub struct GpuDocument {
    text_len: u32,
    line_count: u32,
    /// Byte histogram of a sample of the text, for choosing search anchors.
    byte_freq: Box<[u32; 256]>,
    /// `None` for an empty document (no buffers to hold).
    resident: Option<Resident>,
}

struct Resident {
    parsed: ParsedNodes,
    /// The text alone, for substring search.
    text_group: wgpu::BindGroup,
    // Kept so the buffers outlive every bind group that references them.
    _input: wgpu::Buffer,
    _line_offsets: wgpu::Buffer,
}

impl GpuDocument {
    pub fn line_count(&self) -> usize {
        self.line_count as usize
    }

    pub(crate) fn byte_freq(&self) -> &[u32; 256] {
        &self.byte_freq
    }

    pub(crate) fn text_group(&self) -> Option<&wgpu::BindGroup> {
        self.resident.as_ref().map(|r| &r.text_group)
    }

    pub fn text_len(&self) -> usize {
        self.text_len as usize
    }
}

impl GpuParser {
    /// Uploads, scans and parses `input`, keeping the result in VRAM.
    pub fn upload(&self, input: &str) -> Result<GpuDocument> {
        self.upload_inner(input, None)
    }

    fn upload_inner(&self, input: &str, timer: Option<&StageTimer>) -> Result<GpuDocument> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(GpuDocument {
                text_len: 0,
                line_count: 0,
                byte_freq: Box::new([0; 256]),
                resident: None,
            });
        }
        let text_len = input.len() as u32;

        let input_buffer = self.upload_text(input, timer)?;
        let (line_offsets, line_count) = self.scan_to_buffer(&input_buffer, text_len, timer)?;
        let parsed =
            self.parse_to_buffer(&input_buffer, &line_offsets, line_count, text_len, timer)?;
        let text_group = self.text_bind_group(&input_buffer);

        let byte_freq = sample_byte_histogram(input.as_bytes());
        mark(timer, "upload: sample byte histogram");

        Ok(GpuDocument {
            text_len,
            line_count,
            byte_freq,
            resident: Some(Resident {
                parsed,
                text_group,
                _input: input_buffer,
                _line_offsets: line_offsets,
            }),
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
        self.query_document_inner(doc, kind, depth, max_matches, None)
    }

    pub(crate) fn query_document_inner(
        &self,
        doc: &GpuDocument,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
        timer: Option<&StageTimer>,
    ) -> Result<QueryResult> {
        match &doc.resident {
            None => Ok(QueryResult {
                total_matches: 0,
                indices: Vec::new(),
            }),
            Some(r) => self.run_query(
                &self.filter_pipeline,
                &r.parsed.bind_group,
                (0, doc.line_count),
                kind,
                depth,
                max_matches,
                timer,
            ),
        }
    }

    /// Whether per-pass GPU timings are available (`TIMESTAMP_QUERY`).
    pub fn timestamps_supported(&self) -> bool {
        self.timestamps
    }

    /// `upload`, with a breakdown of where the time went. Waits for the parse pass
    /// so that the GPU work is fully accounted for.
    pub fn profile_upload(&self, input: &str) -> Result<(GpuDocument, Profile)> {
        let timer = StageTimer::new(self);
        let doc = self.upload_inner(input, Some(&timer))?;
        self.device.poll(wgpu::Maintain::wait());
        mark(Some(&timer), "upload: wait for parse pass");
        Ok((doc, timer.finish(self)?))
    }

    /// `query_document`, with a breakdown of where the time went.
    pub fn profile_query_document(
        &self,
        doc: &GpuDocument,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Profile> {
        let timer = StageTimer::new(self);
        self.query_document_inner(doc, kind, depth, max_matches, Some(&timer))?;
        timer.finish(self)
    }
}

impl GpuDocument {
    pub(crate) fn nodes_buffer(&self) -> Option<&wgpu::Buffer> {
        self.resident.as_ref().map(|r| &r.parsed.nodes)
    }
}

impl GpuParser {
    /// Filter over a range of lines of a resident document (`query_subtree` uses it).
    pub(crate) fn query_range(
        &self,
        doc: &GpuDocument,
        range: (u32, u32),
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<QueryResult> {
        match &doc.resident {
            None => Ok(QueryResult {
                total_matches: 0,
                indices: Vec::new(),
            }),
            Some(r) => self.run_query(
                &self.filter_pipeline,
                &r.parsed.bind_group,
                range,
                kind,
                depth,
                max_matches,
                None,
            ),
        }
    }
}
