# parse-line-simd

Goal: cut the per-line cost of `oneline_simd::parse_line`, which dominates single-core parsing.

## Evidence

On 100 MB, newline scanning is ~3.4 ms (30 GB/s) but the full single-core parse is 34 ms (`mixed`) to 82 ms
(`short-lines`): ~7 ns/line on a repeating pattern, ~17 ns/line on random short lines. Random data is slower
than the repeating sample because of branch mispredictions in `parse_line` / `classify_line`.

## Ideas (measure each; keep what wins)

- [ ] Branch-light prefix handling: count the leading `|` run and skip following blanks with one SIMD compare
      + `trailing_zeros` on a 16/32-byte window instead of byte loops.
- [ ] Trim trailing `\r\n` with a single check of the last two bytes instead of a loop.
- [ ] Turn `classify_line` into a 256-entry lookup on the first byte, with the rare multi-byte cases
      (`- `, `1. `, ``` ``` ```) in a cold path.
- [ ] Fuse newline discovery and classification per 32-byte block so each line's bytes are still in L1.
- [ ] Check `-C target-cpu=native` vs runtime dispatch: keep the AVX2 detection, do not assume it.

## Constraints

`parse`, `parse_par` and the GPU shader must keep agreeing node for node. The differential tests
(`oneline-simd` `all_profiles_match_reference_across_chunking`, `oneline-gpu` `gpu_parse_matches_cpu`) are the
guard; any change to classification rules must change `shader.wgsl` in the same commit.
