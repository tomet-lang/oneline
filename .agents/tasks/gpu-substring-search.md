# gpu-substring-search

Goal: decide whether the GPU can beat a strong CPU baseline on a query whose cost is proportional to the
*text*, not to the node count. This was the deciding experiment left by `gpu-profiling-and-overhead.md`.

## Result (RTX 3060, 100-128 MB, release build)

**Yes, on resident text and sparse matches: 3.4-6.6x faster than a 12-thread `memchr::memmem` on five real corpora (4-5x on the synthetic data). Not when matches are dense, and never cold.**

Warm = text already in RAM (CPU) / resident in VRAM (GPU). `prose` data, `cargo bench -p oneline-benches
--bench comparison -- compare/search`, 128 MB:

| query | CPU (12 threads) | GPU | |
|---|---|---|---|
| rare word (~1.3k hits), warm, offsets back | 3.30 ms | **0.65 ms** | GPU 5.1x faster |
| rare word, warm, count only | 3.29 ms | **0.61 ms** | GPU 5.4x faster |
| most frequent word (~1M hits), warm, count only | **3.76 ms** | 5.33 ms | CPU 1.4x faster |
| most frequent word, warm, all offsets back | **4.55 ms** | 9.01 ms | CPU 2.0x faster |
| rare word, **cold** (text only in host memory) | **3.34 ms** | 45.0 ms | CPU 13x faster |

`oneline search --bench-mb 100 --profile prose` (all numbers are medians of 20 runs, 100 MB):

| pattern (hits) | CPU 12 thr | GPU resident | GPU kernel alone |
|---|---|---|---|
| rare word, 6 B (1,205) | 2.55 ms | 0.53 ms | 0.42 ms (~250 GB/s) |
| mid word, 9 B (8,263) | 3.08 ms | 0.77 ms | 0.49 ms |
| two-word phrase, 18 B (11,714) | 2.72 ms | 1.08 ms | 0.73 ms |
| most frequent word (835,301) | 3.27 ms (count) | 4.27 ms (count) | 3.77 ms |

Where it stops working:

- **Cold, always.** Uploading 100 MB costs ~20 ms (see `gpu-profiling-and-overhead.md`); memmem does the whole
  search in ~3 ms. A GPU search only pays when the text stays resident across many queries.
- **Dense matches.** At one hit per ~120 bytes the kernel is output-bound: every match claims a slot, the
  pattern is verified at each, and the host then sorts and reads back up to millions of offsets
  (sort ~2.6 ms at 835k). The CPU baseline does not pay for any of that when it only counts.
- **Small text.** At 16 MB the CPU's whole working set fits in cache (memmem hits ~100 GB/s) and the GPU's
  ~130 us per-query floor eats most of the lead: 142 us vs 185 us.
- **Adversarial data.** On the 23-word `mixed` vocabulary (every pattern prefix occurs every ~160 bytes) the
  GPU still wins sparse queries but only by ~1.4x (1.8 ms vs 2.6 ms) and ties on dense ones.

So the "GPU-resident query engine" thesis holds for exactly one shape of workload: a large corpus kept in
VRAM and probed repeatedly with selective text searches. It does not hold for structural (kind/depth) filters
(parity, see the profiling doc) nor for one-shot queries.


## Real corpora (checked 2026-10-23): the synthetic result holds, and is a little better

The synthetic `prose` data was the main caveat, so the same benchmark was run on text that exists on the
development machine, ~210-260 MB each, read-only, concatenated and cut at 256 MiB (`head -c`):

| corpus | built from |
|---|---|
| rust-src (208 MB) | every `.rs` file in the cargo registry |
| c-headers (256 MB) | `*.h` files in the nix store (contains invalid UTF-8, converted lossily) |
| html-docs (256 MB) | `*.html` in the nix store (markup-heavy documentation) |
| text-docs (232 MB) | `*.txt` + `*.md` in the nix store (licenses, readmes; closest to natural language) |
| json (256 MB) | `*.json` in the nix store (repetitive structured data) |

Method: `oneline search FILE --bench` (median of 15). **Patterns are not hand-picked**: for each corpus the tool counts
words in a ~16 MB sample and takes the word whose estimated number of hits in the whole text is closest to
300 / 5,000 / 100,000 / 1,000,000, plus a 24-byte slice from the middle of the text. Both sides return every
offset (or only the count). CPU = `memchr::memmem` on 12 threads with the text already in RAM; GPU = text resident
in VRAM. Speedup = CPU time / GPU time.

| hits (bytes per hit) | offsets back | count only |
|---|---|---|
| sparse, up to ~27k (>= 10 KB/hit), 14 patterns | **3.4x - 6.6x** (CPU 5.5-6.9 ms, GPU 0.8-1.8 ms) | **3.8x - 6.6x** |
| medium, 55k - 190k (~1.4-4.4 KB/hit), 6 patterns | **1.8x - 3.7x** | **3.0x - 4.7x** |
| dense, 570k - 1.1M (~245-470 B/hit), 4 patterns | **0.8x - 1.3x** (parity; CPU wins on 3 of 4) | **1.2x - 2.4x** |
| cold: upload + search, any pattern | GPU 89-116 ms vs CPU 5.5-8 ms, i.e. **14-20x slower** | |

Per-corpus detail (speedup, offsets / count): rust-src 6.6/6.6, 6.2/6.2, 2.2/3.7, 0.9/1.2 for 74, 1.9k, 125k, 864k hits;
c-headers 6.3/6.4, 6.3/6.6, 5.1/5.5, 2.3/3.9 for 1, 643, 5.2k, 103k hits; html-docs 6.1/6.4, 4.0/3.9, 2.2/3.8, 1.3/2.4,
0.8/1.5 for 104, 13k, 137k, 573k, 1.09M hits; text-docs 5.1/5.2, 6.4/5.9, 4.8/5.2, 3.7/4.7, 1.8/3.4 for 3, 2.2k, 6.9k,
55k, 150k hits; json 6.3/6.6, 5.4/5.6, 3.4/3.8, 3.9/4.7, 1.9/3.0, 0.9/1.7 for 255, 674, 10k, 27k, 186k, 682k hits.

What changed against the synthetic data: the *dense* case is less bad. Synthetic prose had a hit every ~120 bytes (the
most frequent word of a Zipf vocabulary); the densest real patterns are one per 245-470 bytes, and there the GPU wins
the count-only query (1.2-2.4x) instead of losing it. The ordering of regimes is the same: sparse >> medium >> dense,
and cold is always CPU.

Caveats that remain: all five corpora are code, markup or data plus some licence/readme text; there are no
server logs and no large natural-language (e.g. Japanese) text here. A single machine (RTX 3060, 12 threads).

## What was built

- `search.wgsl`, `GpuParser::search_document` (resident), `search_raw` (one-shot), `profile_search_*`,
  `SearchResult { total_matches, offsets }`; patterns 1..=64 bytes; every occurrence (overlapping included) as a
  raw-text byte offset, ascending. Raw text means no line semantics: a pattern may contain `\n`, and matches may
  fall in the `|` prefix.
- `oneline_simd::{find_all, find_all_par, count_all_par}`: the CPU baseline (`memchr::memmem`, Rayon,
  overlapping chunks). Strong on purpose.
- `oneline search` (file mode prints line:column; `--bench-mb` runs the CPU/GPU comparison with an agreement
  check first), `oneline profile --pattern`, criterion `compare/search-{cold,warm}/*`.
- `datagen::Profile::Prose` + `prose_vocabulary()`: 4096-word Zipf-like vocabulary. Needed because the
  23-word `mixed` data makes every pattern prefix common, which is unrepresentative of real text and is the
  worst case for the filter.
- 7 GPU search tests (plus 2 pure unit tests for window choice and sampling), differential against `find_all` on the real GPU, covering word alignments, overlaps,
  text tail, NUL patterns, 1-byte..64-byte patterns, truncation, bad patterns, windows implying a start before
  the text. Four kernel invariants were
  mutation-checked: dropping an alignment, the start guard, the start-position report, or the tail verification
  each makes a test fail.

## How the kernel got from 1.6 ms to 0.42 ms (100 MB, rare pattern)

A plain read of the text takes ~350 us (~300 GB/s, i.e. the memory bandwidth), so that is the floor.

| version | kernel time | what it taught |
|---|---|---|
| 1 thread per word, atomic per workgroup, first-byte filter | 1.64 ms | |
| 16 words/thread, SWAR "any byte equals first byte" gate | 1.32 ms | barriers were not the problem |
| coalesced row layout | 1.36 ms | memory access pattern was not the problem |
| ablation: read + gate only | 0.35 ms | the floor |
| ablation: + verify | 1.13 ms | **the verify path is what costs** |
| ablation: verify only 2 bytes instead of all | 1.34 ms (unchanged) | not its length, **how often it is entered** |
| rare-byte anchor, global memory | 1.30 ms | 4.4% of words still flagged |
| + shared-memory tile for verification | 1.24-1.40 ms | verification latency was not it either |
| **4-byte window compared branch-free at 4 alignments, window picked rarest from a text sample** | **0.42 ms** | |

Lessons worth keeping:

1. **On SIMT hardware the cost of a rare branch is `P(any of 32 lanes takes it)`, not `P(one lane takes it)`.**
   Filtering on one byte flagged ~12% of words, so ~98% of warp iterations entered the slow path. The filter must
   be selective *per warp*, i.e. 4 bytes compared at once, and the window must be chosen from the data.
2. Measure by ablation before theorizing: two of the three explanations tried first (barriers, memory access
   pattern, then verification latency) were wrong, and each was refuted by a 1-minute experiment.
3. The first 4-byte-window attempt was *slower* (1.45-2.3 ms) because per-alignment `&&` chains became many
   branches. Folding the 4 comparisons into one bitmask with a single branch is what made it work.

## Limitations / next steps

- [ ] Output-bound case: results come back unsorted (atomic slot claims) and are sorted on the host. Ordered
      output on the GPU needs per-workgroup counts + a scan (two kernels) and a thread-major layout; only worth it
      if dense-result queries matter.
- [ ] Patterns longer than 64 bytes (the pattern travels in a uniform buffer).
- [ ] Case-insensitive / multi-pattern search would change the filter (and likely the economics).
- [ ] The doc must stay resident: `GpuDocument` holds the text (and nodes) in VRAM. A text-only resident
      object would halve the memory for search-only use.
- [x] Real corpora: done, see "Real corpora" above (five corpora, same ordering of regimes).
- [ ] Still untested: server logs and large natural-language text (e.g. Japanese); other GPUs (one RTX 3060 only).
- [ ] 1 GB+: only checked at 100-128 MB.

## Scaling to 1-2 GiB (2026-10-23)

Same benchmark on HTML documentation concatenated to 1 GiB and 1.85 GiB (median of 8 / 5 runs, text resident). The
advantage **grows** with size because the CPU is DRAM-bound (~40 GB/s) while the kernel holds 210-296 GB/s:

| text | hits (bytes/hit) | CPU 12 thr | GPU offsets / count | speedup |
|---|---|---|---|---|
| 1 GiB | 1.3k - 12k (>= 85 KB/hit) | 27 ms | 4.4-5.1 ms | 5.3-6.1x |
| 1 GiB | 122k (8.8 KB/hit) | 27.6 ms | 6.3 / 4.9 ms | 4.4 / 5.6x |
| 1.85 GiB | 16 - 3.2k | 50 ms | 6.7-7.8 ms | 6.3-7.5x |
| 1.85 GiB | 196k (10 KB/hit) | 51 ms | 9.7 / 8.8 ms | 5.3 / 5.7x |
| 1.85 GiB | 1.63M (1.2 KB/hit) | 54 ms | 24.5 / 17.0 ms | 2.2 / 3.1x |
| 1.85 GiB | 5.0M (380 B/hit) | 70 / 54 ms | 55 / 36 ms | 1.3 / 1.5x |

Cold (upload + search) remains 14-16x slower than the CPU (434 ms and 771 ms). Even the densest case (one hit per
380 bytes, 5M hits) is still a GPU win here, consistent with the real-corpus finding that density, not size, is
what erodes the advantage. The GPU result was checked against the CPU result (count and offsets) before each timing.
