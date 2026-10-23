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

## Status (2026-10-23): done where it paid; the rest is a memory-bandwidth problem

### Where the time goes (1 GiB of HTML, 7.9M lines, single core, before this work: 166 ms)

The task assumed `parse_line` dominated. It does not. Stage-by-stage, adding one piece at a time:

| stage | time |
|---|---|
| `count_newlines` (the pass that sizes the output) | 42 ms (the memory-read floor, ~25 GB/s) |
| list newline positions (was: `while mask != 0` per 32-byte block) | 72 ms, i.e. **+30 ms over a plain count** |
| walk the positions, write `offsets` | +0 ms |
| also write the `FlatNode` (20 B/line of stores) | +8 ms |
| real `parse_line` | **+35 ms** |

So `parse_line` was ~22% of the single-core parse, and the newline listing cost more than it.

### What was changed

- **Branch-free newline listing** (`flatten_newlines`, simdjson's `flatten_bits` idea): an unpredictable
  `while mask != 0` loop mispredicts about once per newline (~24% of 32-byte blocks hold one here). Instead a fixed
  group of positions is written every time and only the count advances. First try, groups of 8, was a wash (69 ms
  to list 1 GiB); the average 64-byte block holds < 1 newline, so groups of **4** (then 8, 16 only when needed)
  were measurably better (69 -> 52 ms). The fill loop works in 16 KiB tiles: list, then classify while the tile is
  in cache.
- **`parse_line`**: iterator-based prefix/blank counting and a 256-entry first-byte table instead of a comparison
  chain.

Single-core `parse`, medians (before this work -> now):

| data | before | now |
|---|---|---|
| mixed, 100 MB | 35.5 ms | **25.8 ms** (-27%) |
| short-lines, 100 MB | 89 ms | **71 ms** (-20%) |
| long-lines, 100 MB | 11.2 ms | 11.0 ms |
| 1 GiB HTML | 166 ms | **145 ms** (-13%) |

`parse_par` (12 threads) moved by 0-6%; at 1 GiB it is unchanged at ~68.5 ms.

### Why `parse_par` does not move: the two-pass design is DRAM-bound

`parse_par` reads the text twice (count, then fill). At 1 GiB each pass costs ~27 ms across 12 threads (~40 GB/s DRAM),
so the floor of this design is ~54 ms of the measured 68 ms. Making the per-line work faster cannot help much.

### Tried and not worth it

- A 16-wide group for the first unconditional writes, and the group-of-8 version: slower than 4.
- Anything SIMD inside `parse_line` itself: the leading-blank run averages 2.2 bytes, so a vector compare would not
  amortize. The scalar iterator version already captured the available win (3-11%).

### Ideas, in order of expected value (not done)

1. **Avoid the second DRAM read for large inputs**: parse each chunk into per-chunk buffers in one pass and gather them
   into the final arrays with a parallel copy afterwards (output is ~15% of the text for long-line data, so
   160 MB of copying replaces 1 GiB of re-reading). Expected parse_par 68 -> ~45 ms at 1 GiB; no gain for short
   lines, where the output is as large as the text. Decoupled look-back would avoid the copy but needs the output
   size up front.
2. Non-temporal stores for `nodes`/`offsets` (they are written once and not re-read).
3. `scan_line_offsets` still uses the old branchy loop (it is only used by tests and the GPU benchmarks).

### Guard rails added

`parse_line` is checked against the previous implementation (kept in the test module) exhaustively over a 7-symbol
alphabet up to length 6 and on 300k random lines; `flatten_newlines` is checked against a naive scan at every
alignment and density 0-100%. The CPU/GPU differential tests are unchanged, so a classification change cannot
silently diverge from `shader.wgsl`.

## Follow-up: one-pass `parse_par` ("gather"), done (2026-10-23)

Idea 1 above, implemented as `ParStrategy::{Auto, TwoPass, Gather}` / `try_parse_par_with`. Gather parses each chunk once into
its own buffers, then copies them into the final arrays in parallel; TwoPass is the previous count-then-fill. `Auto`
(the default of `parse_par`) estimates the output (20 B per line) from four 64 KiB windows and gathers only for inputs of
at least 32 MiB whose output is at most half the size of the text, i.e. average line length >= 40 bytes.

12 threads, medians; the gain tracks "text read / output written":

| data | two-pass | gather | |
|---|---|---|---|
| long-lines 100 MB (655 B/line) | 4.42 ms | 2.49 ms | -44% |
| mixed 100 MB (80 B/line) | 6.77 ms | 5.01 ms | -26% |
| nested 100 MB / prose 100 MB | 6.48 / 6.49 ms | 4.91 / 5.28 ms | -24% / -19% |
| 1 GiB HTML (135 B/line) | 68.3 ms | 59.6 ms | **-13%** |
| 1.85 GiB HTML (157 B/line) | 121.5 ms | 99.0 ms | **-19%** |
| short-lines 100 MB (22 B/line) | 11.96 ms | 16.06 ms | **+34% (worse)** |
| repeat 100 MB (15 B/line) | 14.1 ms | 21.0 ms | **+48% (worse)** |

`Auto` matches the better of the two in every row above (the last two stay on TwoPass).

The estimate for 1 GiB was 68 -> ~45 ms (-35%); the measured -13% is less than that. The cost not in the estimate is the
extra output-sized buffer: its first-touch page faults and the gather copy. Where exactly was not measured. At 1 GiB
the remaining ~59 ms is close to one DRAM read of the text (27 ms) plus the node writes and the copy.

Tests: every strategy is checked against the naive reference on every data profile and on awkward shapes (no final
newline, one giant line, CRLF only, all newlines); the heuristic has its own test (long lines -> gather, short lines /
small input -> two-pass).

Still open from the list above: non-temporal stores; `scan_line_offsets` still uses the old branchy loop.
