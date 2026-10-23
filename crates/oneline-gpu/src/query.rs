use anyhow::{Result, anyhow};
use oneline_core::NodeKind;

use crate::parser::GpuParser;
use crate::timing::{Profile, StageTimer, mark};
use crate::util::{
    bind, check_gpu_len, compute_pass, dispatch_dims, map_all, staging_buffer,
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

/// Indices fetched in the same round trip as the match count.
const FIRST_READ: u32 = 4096;

/// Buffers a counted-result kernel (query or search) needs, kept between calls so
/// a call allocates nothing.
pub(crate) struct QueryScratch {
    /// How many indices `matched` (and `staging`) can hold.
    capacity: u32,
    /// `QueryConfig` uniform for the node filters.
    pub(crate) config: wgpu::Buffer,
    /// `SearchConfig` uniform for substring search.
    pub(crate) search_config: wgpu::Buffer,
    match_count: wgpu::Buffer,
    matched: wgpu::Buffer,
    pub(crate) group1: wgpu::BindGroup,
    pub(crate) search_group1: wgpu::BindGroup,
    /// `[count: u32][indices: u32; capacity]`, read back with one map.
    staging: wgpu::Buffer,
}

impl GpuParser {
    fn new_query_scratch(&self, capacity: u32) -> Result<QueryScratch> {
        let bytes = u64::from(capacity) * 4;
        self.check_fits(bytes, "the match indices")?;
        let uniform = |label: &str, size: usize| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let config = uniform("Query Config Buffer", std::mem::size_of::<QueryConfig>());
        let search_config = uniform("Search Config Buffer", std::mem::size_of::<crate::SearchConfig>());
        let match_count = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Match Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let matched = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matched Indices Buffer"),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let group = |label: &str, cfg: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.query_layout,
                entries: &[bind(0, cfg), bind(1, &match_count), bind(2, &matched)],
            })
        };
        let group1 = group("Query Bind Group", &config);
        let search_group1 = group("Search Bind Group", &search_config);
        let staging = staging_buffer(&self.device, "Staging Query Result Buffer", 4 + bytes);
        Ok(QueryScratch { capacity, config, search_config, match_count, matched, group1, search_group1, staging })
    }

    /// Runs `f` with scratch buffers able to hold `max_matches` indices. The
    /// scratch is shared, so concurrent callers serialize here.
    pub(crate) fn with_scratch<R>(
        &self,
        max_matches: u32,
        timer: Option<&StageTimer>,
        f: impl FnOnce(&QueryScratch) -> Result<R>,
    ) -> Result<R> {
        let mut guard = self.query_scratch.lock().unwrap_or_else(|e| e.into_inner());
        mark(timer, "query: lock scratch");

        if guard.as_ref().is_none_or(|s| s.capacity < max_matches) {
            // Round up so growing max_matches a little does not reallocate every time.
            *guard = Some(self.new_query_scratch(max_matches.max(1024).next_power_of_two())?);
            mark(timer, "query: (re)allocate scratch buffers");
        }
        let scratch = guard.as_ref().expect("scratch was just ensured");

        // A failed readback leaves `staging` mapped or half-mapped; drop the
        // scratch so the next call starts from clean buffers.
        let result = f(scratch);
        if result.is_err() {
            *guard = None;
        }
        result
    }

    /// Dispatches `pipeline` (a filter kernel) over `line_count` lines and reads
    /// back the match count and indices.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_query(
        &self,
        pipeline: &wgpu::ComputePipeline,
        group0: &wgpu::BindGroup,
        range: (u32, u32),
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
        timer: Option<&StageTimer>,
    ) -> Result<QueryResult> {
        let max_u32 = u32::try_from(max_matches).map_err(|_| anyhow!("max_matches {max_matches} does not fit in u32"))?;
        self.with_scratch(max_u32, timer, |s| {
            let q_config = QueryConfig {
                target_kind: kind.map(|k| k as u32).unwrap_or(0),
                target_depth: depth.unwrap_or(0xFFFF_FFFF),
                max_matches: max_u32,
                range_start: range.0,
                range_end: range.1,
                _pad: [0; 3],
            };
            self.queue.write_buffer(&s.config, 0, bytemuck::bytes_of(&q_config));
            let dims = dispatch_dims(range.1.saturating_sub(range.0).div_ceil(64));
            let (total_matches, indices) =
                self.read_counted(s, pipeline, group0, &s.group1, dims, max_u32, "filter", timer)?;
            Ok(QueryResult { total_matches, indices })
        })
    }

    /// Resets the counter, runs a kernel that appends to `match_count` / `matched`,
    /// and reads the result back: `(total_matches, ascending indices)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_counted(
        &self,
        s: &QueryScratch,
        pipeline: &wgpu::ComputePipeline,
        group0: &wgpu::BindGroup,
        group1: &wgpu::BindGroup,
        (wg_x, wg_y): (u32, u32),
        max_matches: u32,
        pass_label: &'static str,
        timer: Option<&StageTimer>,
    ) -> Result<(u32, Vec<u32>)> {
        self.queue.write_buffer(&s.match_count, 0, &0u32.to_ne_bytes());
        mark(timer, "query: write config + reset counter");

        // Read the first few indices in the same round trip as the count. Whether
        // more are needed is only known once the count is back, and copying the
        // whole `max_matches` capacity every time costs far more than the kernel.
        let first = max_matches.min(FIRST_READ);
        let first_bytes = u64::from(first) * 4;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, pass_label, timer);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group0, &[]);
            pass.set_bind_group(1, group1, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&s.match_count, 0, &s.staging, 0, 4);
        if first_bytes > 0 {
            encoder.copy_buffer_to_buffer(&s.matched, 0, &s.staging, 4, first_bytes);
        }
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "query: encode + submit");

        map_all(&self.device, &[&s.staging])?;
        mark(timer, "query: wait for GPU + map readback");

        let total_matches = bytemuck::cast_slice::<u8, u32>(&s.staging.slice(..).get_mapped_range())[0];
        let valid = (total_matches as usize).min(max_matches as usize);

        if valid > first as usize {
            // More matches than the first read covered: fetch the rest.
            s.staging.unmap();
            let rest_from = first_bytes;
            let rest_to = valid as u64 * 4;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&s.matched, rest_from, &s.staging, 4 + rest_from, rest_to - rest_from);
            self.queue.submit(Some(encoder.finish()));
            map_all(&self.device, &[&s.staging])?;
            mark(timer, "query: second readback for the remaining matches");
        }

        let view = s.staging.slice(..).get_mapped_range();
        let mut indices = bytemuck::cast_slice::<u8, u32>(&view)[1..1 + valid].to_vec();
        drop(view);
        s.staging.unmap();
        sort_indices(&mut indices);
        mark(timer, "query: copy out + sort");
        Ok((total_matches, indices))
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
        Ok(self.query_raw_inner(input, kind, depth, max_matches, None)?.indices)
    }

    fn query_raw_inner(
        &self,
        input: &str,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
        timer: Option<&StageTimer>,
    ) -> Result<QueryResult> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(QueryResult { total_matches: 0, indices: Vec::new() });
        }
        let text_len = input.len() as u32;
        let input_buffer = self.upload_text(input, timer)?;
        let (line_offsets, line_count) = self.scan_to_buffer(&input_buffer, text_len, timer)?;
        self.query_fused(&input_buffer, &line_offsets, line_count, text_len, kind, depth, max_matches, timer)
    }

    /// `query_raw`, with a breakdown of where the time went.
    pub fn profile_query_raw(
        &self,
        input: &str,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<Profile> {
        let timer = StageTimer::new(self);
        self.query_raw_inner(input, kind, depth, max_matches, Some(&timer))?;
        timer.finish(self)
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
        let input_buffer = self.upload_text(input, None)?;
        let offsets_buffer = self.upload_u32s("Line Offset Buffer", line_offsets)?;
        Ok(self
            .query_fused(
                &input_buffer,
                &offsets_buffer,
                line_offsets.len() as u32,
                input.len() as u32,
                kind,
                depth,
                max_matches,
                None,
            )?
            .indices)
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
        timer: Option<&StageTimer>,
    ) -> Result<QueryResult> {
        let config = self.uniform("Config Buffer", &ShaderConfig { line_count, text_len });
        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Query Bind Group"),
            layout: &self.fused_layout,
            entries: &[bind(0, input_buffer), bind(1, line_offsets), bind(2, &config)],
        });
        self.run_query(&self.fused_pipeline, &group0, (0, line_count), kind, depth, max_matches, timer)
    }
}

/// Below this a serial sort beats the cost of waking the thread pool.
const PAR_SORT_MIN: usize = 1 << 15;

/// The shader claims result slots with an atomic counter, so indices arrive in
/// arbitrary order. Ascending order is part of the API; for large result sets the
/// sort is the dominant host cost, so use all cores.
fn sort_indices(indices: &mut [u32]) {
    if indices.len() >= PAR_SORT_MIN {
        use rayon::slice::ParallelSliceMut;
        indices.par_sort_unstable();
    } else {
        indices.sort_unstable();
    }
}
