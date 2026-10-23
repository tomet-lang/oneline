use anyhow::{Result, anyhow};
use oneline_core::{NONE, NodeKind, TreeLinks};

use crate::document::GpuDocument;
use crate::parser::GpuParser;
use crate::query::QueryResult;
use crate::timing::{Profile, StageTimer, mark};
use crate::util::{bind, compute_pass, dispatch_dims, map_all, staging_buffer, take_mapped};

/// Depth levels the tree kernels track (`D` in `tree.wgsl`). A content line nested deeper
/// than this makes `build_tree` fail rather than return a wrong tree.
pub const MAX_TREE_DEPTH: u32 = 32;
/// Lines per thread (`RUN` in `tree.wgsl`).
const RUN: u32 = 32;
/// Threads per workgroup of the run kernels.
const RUNS_PER_GROUP: u32 = 64;

/// The six kernels of `tree.wgsl`.
pub(crate) struct TreePipelines {
    pub extract_keys: wgpu::ComputePipeline,
    pub summarize_forward: wgpu::ComputePipeline,
    pub scan_forward: wgpu::ComputePipeline,
    pub assign_parent: wgpu::ComputePipeline,
    pub summarize_backward: wgpu::ComputePipeline,
    pub scan_backward: wgpu::ComputePipeline,
    pub assign_end: wgpu::ComputePipeline,
}

/// `parent` and `subtree_end` of a [`GpuDocument`], resident in VRAM.
///
/// Same meaning as [`TreeLinks`]: the subtree of line `i` is the contiguous range
/// `i..subtree_end[i]`. Blank lines are outside the tree (`parent == NONE`,
/// `subtree_end == i + 1`).
pub struct GpuTree {
    line_count: u32,
    /// Lines the output buffers can hold (>= line_count), so a later build can reuse them.
    capacity: u32,
    pub(crate) parent: wgpu::Buffer,
    pub(crate) subtree_end: wgpu::Buffer,
}

impl GpuTree {
    pub fn line_count(&self) -> usize {
        self.line_count as usize
    }
}

impl GpuParser {
    /// Builds the tree links of a resident document without leaving VRAM.
    ///
    /// Fails if a non-blank line is nested deeper than [`MAX_TREE_DEPTH`] - 1 levels (use
    /// the CPU `oneline_simd::build_links` then).
    pub fn build_tree(&self, doc: &GpuDocument) -> Result<GpuTree> {
        self.build_tree_inner(doc, None, None)
    }

    /// Like [`build_tree`](Self::build_tree), but reuses the output buffers of `previous` when they
    /// are large enough, so rebuilding for a new or edited document of similar size allocates (and
    /// clears) nothing. The working buffers are always reused across calls.
    pub fn build_tree_reusing(&self, doc: &GpuDocument, previous: Option<GpuTree>) -> Result<GpuTree> {
        self.build_tree_inner(doc, previous, None)
    }

    /// `build_tree_reusing`, with a breakdown of where the time went.
    pub fn profile_build_tree_reusing(
        &self,
        doc: &GpuDocument,
        previous: Option<GpuTree>,
    ) -> Result<(GpuTree, Profile)> {
        let timer = StageTimer::new(self);
        let tree = self.build_tree_inner(doc, previous, Some(&timer))?;
        Ok((tree, timer.finish(self)?))
    }

    /// `build_tree`, with a breakdown of where the time went.
    pub fn profile_build_tree(&self, doc: &GpuDocument) -> Result<Profile> {
        Ok(self.profile_build_tree_reusing(doc, None)?.1)
    }

    fn build_tree_inner(
        &self,
        doc: &GpuDocument,
        previous: Option<GpuTree>,
        timer: Option<&StageTimer>,
    ) -> Result<GpuTree> {
        let n = doc.line_count() as u32;
        let Some(nodes) = doc.nodes_buffer() else {
            // Empty document: nothing to compute, but keep the type uniform.
            let empty = || self.storage_scratch("Empty Tree Buffer", 4);
            return Ok(GpuTree { line_count: 0, capacity: 0, parent: empty(), subtree_end: empty() });
        };
        let runs = n.div_ceil(RUN);
        let links_bytes = u64::from(n) * 4;
        self.check_fits(links_bytes, "the tree links")?;

        // Output buffers: the previous tree's when they fit, else new ones.
        let (parent, subtree_end, capacity) = match previous {
            Some(t) if t.capacity >= n => (t.parent, t.subtree_end, t.capacity),
            _ => (
                self.storage_scratch_copy("Tree Parent Buffer", links_bytes),
                self.storage_scratch_copy("Tree Subtree End Buffer", links_bytes),
                n,
            ),
        };

        // Working buffers: kept across calls, grown when a bigger document comes.
        let mut guard = self.tree_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_none_or(|s| s.line_capacity < n) {
            *guard = Some(self.new_tree_scratch(n)?);
        }
        let scratch = guard.as_ref().expect("just ensured");
        self.queue.write_buffer(&scratch.cfg, 0, bytemuck::bytes_of(&TreeConfig { line_count: n, runs, _pad: [0; 2] }));
        self.queue.write_buffer(&scratch.status, 0, &[0u8; 8]);

        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tree Bind Group 0"),
            layout: &self.tree_layout0,
            entries: &[
                bind(0, nodes),
                bind(1, &scratch.cfg),
                bind(2, &scratch.summary),
                bind(3, &scratch.scanned),
                bind(4, &scratch.status),
                bind(5, &scratch.keys),
            ],
        });
        let out_group = |label: &str, buffer: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.tree_layout1,
                entries: &[bind(0, buffer)],
            })
        };
        let parent_group = out_group("Tree Parent Output", &parent);
        let end_group = out_group("Tree Subtree End Output", &subtree_end);
        mark(timer, "tree: prepare buffers + bind groups");

        let run_dims = dispatch_dims(runs.div_ceil(RUNS_PER_GROUP));
        let key_dims = dispatch_dims(n.div_ceil(4).div_ceil(256));
        let p = &self.tree;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        // parent: summarize runs, scan across runs per level, assign.
        for (label, pipeline, groups, out) in [
            ("tree: extract key bytes", &p.extract_keys, key_dims, None),
            ("tree: summarize runs (forward)", &p.summarize_forward, run_dims, None),
            ("tree: scan runs per level (forward)", &p.scan_forward, (MAX_TREE_DEPTH, 1), None),
            ("tree: assign parent", &p.assign_parent, run_dims, Some(&parent_group)),
            ("tree: summarize runs (backward)", &p.summarize_backward, run_dims, None),
            ("tree: scan runs per level (backward)", &p.scan_backward, (MAX_TREE_DEPTH, 1), None),
            ("tree: assign subtree end", &p.assign_end, run_dims, Some(&end_group)),
        ] {
            let mut pass = compute_pass(&mut encoder, label, timer);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group0, &[]);
            // The scan kernels never touch group 1, but a bind group must exist for the
            // assign kernels; the others simply ignore it.
            if let Some(g) = out {
                pass.set_bind_group(1, g, &[]);
            } else {
                pass.set_bind_group(1, &parent_group, &[]);
            }
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        encoder.copy_buffer_to_buffer(&scratch.status, 0, &scratch.staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "tree: encode + submit");

        map_all(&self.device, &[&scratch.staging])?;
        let flag = take_mapped::<u32>(&scratch.staging, 1)[0];
        mark(timer, "tree: wait for GPU + read status");
        if flag != 0 {
            return Err(anyhow!(
                "the document has a non-blank line nested {MAX_TREE_DEPTH} or more levels deep; \
                 the GPU tree kernels track {MAX_TREE_DEPTH} levels (use oneline_simd::build_links)"
            ));
        }
        Ok(GpuTree { line_count: n, capacity, parent, subtree_end })
    }

    /// Working buffers for documents of up to `lines` lines (rounded up, so a slightly bigger
    /// document does not reallocate).
    fn new_tree_scratch(&self, lines: u32) -> Result<TreeScratch> {
        let line_capacity = lines.next_multiple_of(1 << 16).max(1 << 16);
        let runs = line_capacity.div_ceil(RUN);
        let summary_bytes = u64::from(runs) * u64::from(MAX_TREE_DEPTH) * 4;
        self.check_fits(summary_bytes, "the tree run summaries")?;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(4), usage, mapped_at_creation: false })
        };
        Ok(TreeScratch {
            line_capacity,
            summary: buffer("Tree Run Summary Buffer", summary_bytes, wgpu::BufferUsages::STORAGE),
            scanned: buffer("Tree Run Start Buffer", summary_bytes, wgpu::BufferUsages::STORAGE),
            keys: buffer("Tree Key Bytes Buffer", u64::from(line_capacity.div_ceil(4)) * 4, wgpu::BufferUsages::STORAGE),
            status: buffer("Tree Status Buffer", 8, rw), // [error flag, deepest line]
            cfg: buffer("Tree Config Buffer", std::mem::size_of::<TreeConfig>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            staging: staging_buffer(&self.device, "Tree Status Staging Buffer", 4),
        })
    }

    /// Reads the tree links back to the host (for checking, or to hand to the CPU).
    pub fn download_links(&self, tree: &GpuTree) -> Result<TreeLinks> {
        let n = tree.line_count();
        if n == 0 {
            return Ok(TreeLinks { parent: Vec::new(), subtree_end: Vec::new() });
        }
        Ok(TreeLinks {
            parent: self.download_u32s(&tree.parent, n)?,
            subtree_end: self.download_u32s(&tree.subtree_end, n)?,
        })
    }

    /// Lines of the subtree rooted at `root` (a contiguous range of lines) that match the
    /// kind/depth filter. Everything stays in VRAM except one 4-byte read of `subtree_end[root]`
    /// to size the dispatch, so the cost scales with the subtree, not the document.
    pub fn query_subtree(
        &self,
        doc: &GpuDocument,
        tree: &GpuTree,
        root: usize,
        kind: Option<NodeKind>,
        depth: Option<u32>,
        max_matches: usize,
    ) -> Result<QueryResult> {
        if root >= tree.line_count() || tree.line_count() != doc.line_count() {
            return Err(anyhow!(
                "line {root} is out of range (document has {} lines, tree has {})",
                doc.line_count(),
                tree.line_count()
            ));
        }
        let end = self.download_u32_at(&tree.subtree_end, root)?;
        self.query_range(doc, (root as u32, end), kind, depth, max_matches)
    }

    fn download_u32_at(&self, src: &wgpu::Buffer, index: usize) -> Result<u32> {
        let staging = staging_buffer(&self.device, "Staging Single Value Buffer", 4);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, index as u64 * 4, &staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));
        map_all(&self.device, &[&staging])?;
        Ok(take_mapped::<u32>(&staging, 1)[0])
    }

    fn storage_scratch(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn storage_scratch_copy(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeConfig {
    line_count: u32,
    runs: u32,
    _pad: [u32; 2],
}

/// `NONE` as the shaders use it.
const _: () = assert!(NONE == 0xFFFF_FFFF);

/// Working buffers of the tree kernels, kept between builds.
pub(crate) struct TreeScratch {
    line_capacity: u32,
    summary: wgpu::Buffer,
    scanned: wgpu::Buffer,
    keys: wgpu::Buffer,
    status: wgpu::Buffer,
    cfg: wgpu::Buffer,
    staging: wgpu::Buffer,
}
