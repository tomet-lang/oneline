# benches-criterion

Goal: Establish standard Criterion benchmarks for `oneline` in a dedicated workspace package `benches/` (`oneline-benches`).

## Motivation

CLI benchmarks (`just bench-mb`) provide rapid wall-clock throughput measurements, but Criterion provides:
1. Statistically rigorous measurement (p-values, standard deviation, outlier detection).
2. Throughput tracking over time and automatic regression detection.
3. Isolated dependencies: Criterion and plotters will not enter the core crates' test graph, keeping `cargo test --workspace` sub-second and noise-free.

## Architecture

```
benches/
├── Cargo.toml          # Package name: oneline-benches, criterion dependency
└── benches/
    ├── parse_simd.rs   # Single-core vs Multi-core Rayon AVX2 parse benchmarks
    ├── parse_gpu.rs    # GPU Prefix Scan, GPU Parse Raw, GPU Query benchmarks
    └── comparison.rs   # Direct CPU vs GPU throughput comparison matrix
```

## Inputs & Matrix

- Document scales:
  - Small: 64 KB (cache resident)
  - Medium: 1 MB, 10 MB (L3 cache boundary)
  - Large: 50 MB, 100 MB (DRAM bandwidth saturation)
- Engines:
  - `simd_scalar`: Fallback SWAR scanner
  - `simd_avx2_single`: Single-core AVX2 line scanner & parser
  - `simd_avx2_par`: Multi-core Rayon AVX2 parallel parser
  - `gpu_full_pipeline`: Complete GPU prefix scan + GPU line parse
  - `gpu_resident_query`: VRAM-resident line filtering

## Steps

- [x] Add `benches` to workspace members in root `Cargo.toml`.
- [x] Add `criterion = { version = "0.5", default-features = false, features = ["cargo_bench_support"] }` to `[workspace.dependencies]` (without `cargo_bench_support`, `cargo bench` cannot pass its `--bench` flag; plotters/HTML reports stay off).
- [x] Create `benches/Cargo.toml` with `[[bench]]` targets (harness = false).
- [x] Implement `benches/benches/parse_simd.rs` measuring single vs multi-core throughput in GB/s.
- [x] Implement `benches/benches/parse_gpu.rs` with graceful fallback when headless/no Vulkan adapter.
- [x] Add `just bench-criterion` recipe to `justfile` with appropriate `LD_LIBRARY_PATH` export.
- [x] Verify `cargo bench -p oneline-benches` runs and produces reports.

## Status: done

Layout as built: `benches/benches/{parse_simd,parse_gpu,comparison}.rs` plus the shared
`benches/benches/common/mod.rs` (a `common/` directory, not `common.rs`, so cargo does not
treat it as a bench target). Datasets come from `oneline_core::datagen` (profiles `repeat`,
`short-lines`, `mixed`, `long-lines`), not from one repeated sample. Run `just bench-criterion`
(or `just bench-criterion parse_simd`); add `-- --quick` for a fast pass.

Rows of `comparison.rs`: `cold/*` is raw text in, indices out (CPU parse+filter vs GPU one-shot);
`warm/*` is an already parsed document queried repeatedly (CPU parallel filter vs GPU resident).

Caveat: on a software adapter (llvmpipe) the GPU rows only prove the benches run. Real GPU
numbers need real hardware; `common::gpu()` prints a warning when the adapter is software.
