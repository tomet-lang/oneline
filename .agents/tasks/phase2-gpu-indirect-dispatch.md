# phase2-gpu-indirect-dispatch

Goal: Eliminate CPU synchronization roundtrips by using GPU Indirect Dispatch (`dispatch_workgroups_indirect`) and GPU-written configs, enabling 100% autonomous single-submission GPU execution.

## Motivation & Current Bottleneck

Currently in Stage 1:
1. Pass 1 and Pass 2 run on the GPU.
2. The CPU maps and reads `scan_meta[0]` (4 bytes) over PCIe to obtain `total_lines` and calculate the workgroup grid size `(wg_x, wg_y)`.
3. The CPU then launches Pass 3, Stage 2 (`parse_lines`), and Stage 3 (`filter_nodes`).

Although reading 4 bytes takes only ~0.1 ms, it stalls the CPU thread and incurs a PCIe synchronization boundary.
By utilizing WebGPU indirect compute dispatch, the GPU can calculate its own workgroup dispatch dimensions and execute the entire pipeline without CPU intervention.

## Technical Design

### 1. Indirect Dispatch Buffer Format (WebGPU Specification)
An indirect dispatch buffer contains tightly packed 12 bytes (`[u32; 3]`):
- Offset 0: `workgroup_count_x` (u32)
- Offset 4: `workgroup_count_y` (u32)
- Offset 8: `workgroup_count_z` (u32)
Buffer must have `wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE`.

### 2. WGSL Pass 2 Extension (`scan.wgsl`)
```wgsl
struct IndirectArgs {
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
    _pad: u32,
};

@group(0) @binding(5) var<storage, read_write> indirect_buf: IndirectArgs;

@compute @workgroup_size(1)
fn scan_block_counts() {
    ...
    let total_wgs = (total_lines + 63u) / 64u;
    if (total_wgs <= 65535u) {
        indirect_buf.wg_x = total_wgs;
        indirect_buf.wg_y = 1u;
        indirect_buf.wg_z = 1u;
    } else {
        indirect_buf.wg_x = 65535u;
        indirect_buf.wg_y = (total_wgs + 65534u) / 65535u;
        indirect_buf.wg_z = 1u;
    }
}
```

### 3. Dynamic Storage Config
Convert `ShaderConfig` (`line_count`, `text_len`) from `var<uniform>` to `var<storage, read_write>` so Pass 2 writes `config.line_count = total_lines` directly in VRAM.

### 4. Single-Submission Execution Flow
A single `wgpu::CommandEncoder` records:
- ComputePass 1: Pass 1 (`count_block_newlines`)
- ComputePass 2: Pass 2 (`scan_block_counts`) -> writes `indirect_buf` and `config`
- ComputePass 3: Pass 3 (`scatter_line_offsets`)
- ComputePass 4: Stage 2 (`parse_lines`) -> `compute_pass.dispatch_workgroups_indirect(&indirect_buf, 0)`
- ComputePass 5 (Query): Stage 3 (`filter_nodes`) -> `compute_pass.dispatch_workgroups_indirect(&indirect_buf, 0)`
- Copy results to staging buffer
- Single `queue.submit(...)`

## Steps

- [ ] Add `indirect_buf` with `BufferUsages::INDIRECT | BufferUsages::STORAGE` to scan pipelines.
- [ ] Update `scan_block_counts` in `scan.wgsl` to compute and write `IndirectArgs`.
- [ ] Update `shader.wgsl` to consume `config` via storage buffer.
- [ ] Update `query_raw` in `crates/oneline-gpu/src/lib.rs` (or `query.rs`) to dispatch all passes in 1 command encoder with zero CPU readbacks.
- [ ] Verify `just test` passes with exact matching results.
- [ ] Benchmark `just query-bench 100` and `just query-bench 200` to measure latency reduction from zero-CPU sync.

## Status: ON HOLD — the design has a sizing problem

"Zero CPU readback" cannot work as written. `line_offsets` and `nodes` are sized from
`total_lines`, which only the GPU knows after pass 2, and a buffer cannot be allocated from a
GPU-written value. Sizing for the worst case (every byte a newline) costs 20x the input
(4 B offset + 16 B node per byte), which is unusable at GB scale.

The thing being avoided is a 4-byte readback (~0.1 ms by this document's own estimate), against
kernels that run for milliseconds. Before any of this: add timestamp queries per pass and show
that the sync, not upload/allocation, is the bottleneck. Cheaper wins already landed instead:
single-copy text upload (`mapped_at_creation`), a fused parse+filter kernel for one-shot queries,
and `GpuDocument` so repeated queries skip upload/scan/parse entirely.
