use anyhow::{Result, anyhow};
use oneline_core::NodeKind;

use crate::parser::GpuParser;
use crate::util::{
    bind, check_gpu_len, compute_pass, dispatch_dims, map_all, staging_buffer, take_mapped,
};
use crate::{QueryConfig, ShaderConfig};

/// Result of a GPU query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult {
    /// Number of lines that matched (may exceed `indices.len()`).
    pub total_matches: u32,
    /// Matching line indices in ascending order. If more than `max_matches` lines
    /// matched, this is an arbitrary subset of size `max_matches`: the shader
    /// claims slots with an atomic counter, so which lines make the cut is not
    /// deterministic.
    pub indices: Vec<u32>,
}

impl GpuParser {
    /// Dispatches `pipeline` (a filter kernel) over `line_count` lines and reads
    /// back the match count and indices.
    pub(crate) fn run_query(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group0: &wgpu::BindGroup,
        line_count: u32,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<QueryResult> {
        let max_u32 = u32::try_from(max_matches).map_err(|_| anyhow!("max_matches {max_matches} does not fit in u32"))?;
        let indices_bytes = u64::from(max_u32) * 4;
        self.check_fits(indices_bytes, "the match indices")?;

        let q_config = self.uniform(
            "Query Config Buffer",
            &QueryConfig {
                target_kind: kind.map(|k| k as u32).unwrap_or(0),
                target_depth: depth.unwrap_or(0xFFFF_FFFF),
                max_matches: max_u32,
                _pad: 0,
            },
        );
        let match_count = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Match Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&match_count, 0, &0u32.to_ne_bytes());
        let matched = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matched Indices Buffer"),
            size: indices_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Query Bind Group"),
            layout: &self.query_layout,
            entries: &[bind(0, &q_config), bind(1, &match_count), bind(2, &matched)],
        });
        let staging_count = staging_buffer(&self.device, "Staging Count Buffer", 4);
        let staging_indices = staging_buffer(&self.device, "Staging Indices Buffer", indices_bytes);

        let (wg_x, wg_y) = dispatch_dims(line_count.div_ceil(64));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, "Oneline Filter Pass");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group0, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&match_count, 0, &staging_count, 0, 4);
        if indices_bytes > 0 {
            encoder.copy_buffer_to_buffer(&matched, 0, &staging_indices, 0, indices_bytes);
        }
        self.queue.submit(Some(encoder.finish()));

        map_all(&self.device, &[&staging_count, &staging_indices])?;
        let total_matches = take_mapped::<u32>(&staging_count, 1)[0];
        let valid = (total_matches as usize).min(max_matches);
        let mut indices = take_mapped::<u32>(&staging_indices, valid);
        indices.sort_unstable();
        Ok(QueryResult { total_matches, indices })
    }

    /// One-shot query on raw text: upload, scan, then a fused parse+filter kernel
    /// that never materializes the node array. Only the matches come back.
    pub fn query_raw(
        &self,
        input: &str,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let text_len = input.len() as u32;
        let input_buffer = self.upload_text(input)?;
        let (line_offsets, line_count) = self.scan_to_buffer(&input_buffer, text_len)?;
        self.query_fused(&input_buffer, &line_offsets, line_count, text_len, kind, depth, max_matches)
    }

    /// One-shot query with line offsets computed elsewhere (e.g. by the CPU scanner).
    pub fn query(
        &self,
        input: &str,
        line_offsets: &[u32],
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        check_gpu_len(input.len())?;
        if line_offsets.is_empty() {
            return Ok(Vec::new());
        }
        let input_buffer = self.upload_text(input)?;
        let offsets_buffer = self.upload_u32s("Line Offset Buffer", line_offsets)?;
        self.query_fused(
            &input_buffer,
            &offsets_buffer,
            line_offsets.len() as u32,
            input.len() as u32,
            kind,
            depth,
            max_matches,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn query_fused(
        &self,
        input_buffer: &wgpu::Buffer,
        line_offsets: &wgpu::Buffer,
        line_count: u32,
        text_len: u32,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Vec<u32>> {
        let config = self.uniform("Config Buffer", &ShaderConfig { line_count, text_len });
        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Query Bind Group"),
            layout: &self.fused_layout,
            entries: &[bind(0, input_buffer), bind(1, line_offsets), bind(2, &config)],
        });
        Ok(self
            .run_query(&self.fused_pipeline, &group0, line_count, kind, depth, max_matches)?
            .indices)
    }
}
