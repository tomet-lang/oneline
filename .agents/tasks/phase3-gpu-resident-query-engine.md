# phase3-gpu-resident-query-engine

Goal: Expand the GPU-resident query engine to support substring/keyword search, structural hierarchical filtering, and in-VRAM aggregation statistics over gigabyte-scale documents (VRAM bandwidth is the ceiling; see Expectations).

## Motivation

Once a document is parsed inside GPU VRAM, downloading millions of nodes to the CPU for searching is a waste of PCIe bandwidth.
By moving query execution into compute shaders, queries execute directly against VRAM, returning only the small subset of matching line indices or summary metrics.

## Features to Implement

### 1. In-VRAM Substring / Keyword Search
- Support searching for arbitrary text patterns (e.g. `--pattern "TODO"` or `--pattern "warning"`).
- Pass search pattern as a u32-packed byte buffer in uniform/storage binding.
- Shader checks `input_text` within each line's `[span_start..span_start + span_len]` for pattern matches using Boyer-Moore or rolling hash / SWAR byte search.

### 2. In-VRAM Document Statistics & Histogram
- Compute structural summary of the document entirely in GPU VRAM:
  - Total line count
  - Count of Headings, Elements, Lists, Callouts, Text blocks, Blank lines
  - Max nesting depth
- Output: Single 32-byte POD struct `DocStats` returned to CPU in microseconds.

### 3. Structural & Hierarchical Matching
- Query nodes based on tree relationship:
  - `depth <= N`
  - Sub-tree filtering (e.g. all nodes nested under a specific Heading or Callout).

## Architecture

### WGSL Query Extension (`shader.wgsl`)
```wgsl
struct AdvancedQueryConfig {
    target_kind: u32,
    target_depth: u32,
    max_matches: u32,
    pattern_len: u32,
    pattern_bytes: array<u32, 16>, // Up to 64 bytes pattern
};

struct DocStats {
    total_lines: u32,
    count_text: atomic<u32>,
    count_heading: atomic<u32>,
    count_element: atomic<u32>,
    count_list: atomic<u32>,
    count_callout: atomic<u32>,
    count_raw: atomic<u32>,
    max_depth: atomic<u32>,
};
```

## Steps

- [ ] Add `AdvancedQueryConfig` and `DocStats` structures in `shader.wgsl` and Rust.
- [ ] Implement substring search routine inside `filter_nodes` compute shader.
- [ ] Implement `compute_stats` compute shader for instant document telemetry.
- [ ] Expose `query_text(&self, input: &str, pattern: &str) -> Result<Vec<u32>>` on `GpuParser`.
- [ ] Expose `stats(&self, input: &str) -> Result<DocStats>` on `GpuParser`.
- [ ] Add `--pattern` and `--stats` flags to `oneline query` CLI.
- [ ] Add unit tests verifying keyword matching and statistics accuracy.
- [x] Benchmark query throughput: substring search reaches ~250 GB/s kernel-only, 5x the CPU on resident text (see `gpu-substring-search.md`).

## Status / prerequisites (updated)

- **Done:** `GpuParser::upload` -> `GpuDocument` + `query_document` (kind/depth filter over nodes kept in
  VRAM, returns `total_matches` + indices). Substring search, `DocStats` and the `--pattern/--stats` flags are not started.
- **Expectations:** the "300+ GB/s" in the motivation is the VRAM bandwidth ceiling, not a target; the
  "5+ GB/s" step is the realistic goal. Neither is measured yet (only a software adapter was available).
  Residency only pays after enough queries; `oneline bench` prints the break-even once run on a real GPU.
- **Subtree queries need no new tree walk:** `oneline_simd::build_links` gives `parent[]` and
  `subtree_end[]`, and the subtree of line `i` is the contiguous range `i..subtree_end[i]`. A "nodes under
  heading H" query is a range filter. A GPU version of `build_links` is the same ANSV algorithm
  (see `crates/oneline-simd/src/links.rs`: local pass, small serial combine, resolve pass).

## Re-prioritization from real-GPU measurements (2026-10-23)

Kind/depth filters only read the 16 B/line node array, which is too little memory traffic for a GPU to win:
on an RTX 3060 the CPU's parallel filter over parsed nodes is 5-15x faster than a resident GPU query at
1-128 MB (see `gpu-profiling-and-overhead.md`). The experiment that can still show a GPU advantage is
**substring search**: it scans the *text* (100 MB+ per query, no node array), where VRAM bandwidth (~360 GB/s)
vs a 12-core CPU (~60 GB/s) is a real difference. Do `gpu-profiling-and-overhead.md` first (the ~1 ms query
floor would swamp a sub-millisecond kernel), then implement substring search as the deciding benchmark, with a
CPU baseline (`memchr`-style SIMD search on 12 threads) in `benches/benches/comparison.rs`.

## Status update: substring search done (see `gpu-substring-search.md`)

Substring search is implemented and was the deciding experiment: on resident text with selective patterns
the GPU is 4-5x faster than a 12-thread memmem (0.65 ms vs 3.3 ms at 128 MB); with dense matches or cold it
loses. `DocStats` and the `--pattern/--stats` flags on `oneline query` are still not done; the structural
`--stats` histogram reads only nodes, which the profiling results say is at best parity with the CPU, so it is
low priority. Line-oriented search (match positions mapped to lines, restricted to the content span) is also
open: today `search_*` works on raw bytes.
