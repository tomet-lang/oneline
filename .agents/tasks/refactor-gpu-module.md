# refactor-gpu-module

Goal: Refactor `crates/oneline-gpu/src/lib.rs` (currently ~1,460 lines) into modular submodules so that no source file exceeds 400 lines, improving maintainability, testability, and code navigation.

## Context

`oneline-gpu` currently contains all shader loading, compute pipeline layouts, parallel prefix scan routines (Stage 1), line parsing (Stage 2), GPU-resident query filtering (Stage 3), and integration tests inside a single `lib.rs`.
As we add GPU indirect dispatch and query enhancements, modular separation is required.

## Target Architecture

```
crates/oneline-gpu/src/
├── lib.rs          # Public types, re-exports, module tree, ShaderConfig / ScanConfig / QueryConfig
├── parser.rs       # GpuParser struct definition, device setup, core parse() / parse_async()
├── scan.rs         # Stage 1 Prefix Scan (scan_line_offsets_gpu_buffer, scan_line_offsets, parse_raw)
├── query.rs        # Stage 3 GPU-Resident Query (query, query_raw)
├── shader.wgsl     # WGSL Compute Shader for parse_lines and filter_nodes
├── scan.wgsl       # WGSL Compute Shader for 3-pass parallel prefix scan
└── tests.rs        # Full GPU integration and regression tests
```

## Steps

- [x] Create `crates/oneline-gpu/src/scan.rs` and extract Stage 1 parallel prefix scan pipelines & methods:
  - `scan_pipeline_count`, `scan_pipeline_scan`, `scan_pipeline_scatter`
  - `scan_line_offsets_gpu_buffer`
  - `scan_line_offsets` / `scan_line_offsets_async`
  - `parse_raw` / `parse_raw_async`
- [x] Create `crates/oneline-gpu/src/query.rs` and extract query pipelines & methods:
  - `filter_pipeline`, `query_bind_group_layout`
  - `query` / `query_async`
  - `query_raw` / `query_raw_async`
- [x] Create `crates/oneline-gpu/src/parser.rs` with `GpuParser::new()`, `GpuParser::new_async()`, device request, and basic `parse` / `parse_async`.
- [x] Move integration tests from `lib.rs` into `crates/oneline-gpu/src/tests.rs` (or `tests/integration.rs`).
- [x] Keep `lib.rs` clean (~50-100 lines) with clear doc comments, public API re-exports, and data POD structures.
- [x] Verify `just test` passes with zero regressions across all 11 tests.
- [x] Benchmark with `just bench-mb 100 5` to ensure no runtime overhead from modularization.

## Status: done (layout differs from the sketch above)

```
crates/oneline-gpu/src/
├── lib.rs        # module tree, re-exports, POD configs
├── util.rs       # layout/bind helpers, dispatch_dims, staging + readback, length limit
├── parser.rs     # GpuParser, device + pipeline setup, text/uniform upload, buffer-limit check
├── scan.rs       # Stage 1 prefix scan, scan_line_offsets
├── parse.rs      # Stage 2, parse / parse_raw
├── query.rs      # Stage 3, query / query_raw (fused) / run_query
├── document.rs   # GpuDocument: upload once, query many
└── tests.rs
```

Besides the split, the five copies of "create buffers + bind group + 2D dispatch" collapsed into
`util.rs` helpers, the `*_async` variants were dropped (they blocked on `device.poll` anyway;
only `GpuParser::new_async` remains), and tests grew from 6 to 10 with CPU-vs-GPU differential
checks. Those found a real bug: the shader did not classify numbered list items (`1. x`) that
the CPU parser does.
