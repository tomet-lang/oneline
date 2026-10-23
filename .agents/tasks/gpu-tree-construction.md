# gpu-tree-construction

Goal: build the tree (parent + subtree range) from the per-line depths *on the GPU*, so that the whole pipeline
text -> lines -> nodes -> tree -> queries runs inside VRAM. Done; verified against the CPU on every data profile.

## Result (RTX 3060)

`GpuParser::build_tree(&doc) -> GpuTree` (`parent[]`, `subtree_end[]`, same meaning as `TreeLinks`),
`download_links`, `query_subtree(doc, tree, root, kind, depth, max)`; kernels in `tree.wgsl`.

| document | GPU kernels | GPU wall (incl. buffer allocation) | CPU `build_links` (12 thr) |
|---|---|---|---|
| 100 MB `mixed`, 1.3M lines | 0.58 ms | 0.82 ms | 4.3 ms (5.3x) |
| 1 GiB `mixed`, 13M lines | **4.6 ms** | 8.6 ms | 40.2 ms (**4.7x**; 8.7x kernels only) |
| 1 GiB `nested` | 4.8 ms | 8.3 ms | 40.7 ms |
| criterion `nested`, 128 MB | | 1.89 ms | 5.84 ms (3.1x) |
| criterion `nested`, 16 MB | | 1.02 ms | 1.08 ms (parity: fixed costs) |

Of the 8.6 ms wall at 1 GiB, 2.5 ms is allocating the working buffers and ~1 ms is the final sync; both could be
amortized by keeping the buffers (not done).

**Subtree queries are not a GPU win except for very large subtrees.** `query_subtree` (kind filter over
`root..subtree_end[root]`, 1 GiB `nested` document, 12M+ lines):

| subtree lines | GPU | CPU 1 thread | CPU 12 threads |
|---|---|---|---|
| 10 | 89 us | 0.02 us | 43 us |
| 967 | 109 us | 0.7 us | 34 us |
| 99,964 | 121 us | 33 us | 41 us |
| 922,208 | 131 us | 345 us | **107 us** |

The GPU pays ~90-130 us per query (it reads `subtree_end[root]` back, 4 bytes, to size the dispatch, then runs and
reads back the result); the CPU range filter on a cache-resident range is far cheaper. They meet only around a
million lines; the 13M-line whole-document filter is where the GPU leads (4.5x, see `gpu-profiling-and-overhead.md`).
So the point of building the tree on the GPU is *completeness and the 4.7x build*, not faster small subtree lookups.

## Algorithm

The CPU version is a monotonic stack plus a persistent-stack combine; a serial combine over ~50k blocks would
dominate on a GPU. Depth is a small integer, which allows an O(n) formulation with no stack:

- keep, per depth level `k`, the position of the last line with depth exactly `k`; then
  `parent[i] = max over k < depth[i] of last[k]`. `subtree_end` is the mirror (first[k], min over `k <= depth[i]`).
- Lines are cut into runs, one thread per run: (1) each run records `last[k]` for every level, (2) per level an
  exclusive prefix-max across runs gives each run its starting state (one workgroup per level), (3) each run
  re-walks its lines from that state and writes the answers. Same shape as the line-offset scan.
- A one-byte-per-line key (depth, or a marker for blank / too-deep lines) is extracted first so later passes read
  1 byte per line, not the 16-byte node.
- A non-blank line nested 32 or more levels deep cannot be represented: `build_tree` returns an error instead of a
  wrong tree (use `oneline_simd::build_links`). Blank lines may be arbitrarily "deep"; they are outside the tree.

## How it got from 27.7 ms to 4.6 ms (1 GiB `mixed`, kernels only)

| version | kernels | what it taught |
|---|---|---|
| thread per 256-line run, reads 16-byte nodes | 27.7 ms | assign passes 9.8 + 10.3 ms for ~260 MB of traffic: ~26 GB/s |
| one-byte keys | 23.8 ms | summarize 3.1 -> 0.7 ms, assign unchanged |
| ablation of `assign_parent` | | **per-element result writes cost 8.3 of 9.3 ms**; key loads ~5.8 ms; the level loop ~2 ms |
| runs of 32 lines; shared-memory tiles for keys and results (coalesced global access); level-major summaries | 10.2 ms | assign 9.3 -> 1.4 ms, but the scan got 2.5x slower (8x more runs) |
| process only the levels in use (deepest line recorded on the GPU) | **4.6 ms** | scans 3.0 -> 0.7 ms; summaries and state init shrink too |

Lessons:

1. When each thread walks its own contiguous run, the threads of a warp touch addresses a whole run apart. Writes
   were 90% of a kernel's time. The cure is to move the strided access into shared memory (rows padded by one word so
   a warp hits different banks) and do the global loads/stores cooperatively and contiguously.
2. Shorter runs fix coalescing but multiply per-run state; bound that state by what the data uses.
3. Both slow versions were found by ablation (replace one part with a constant and watch the time), not by reasoning:
   the first guess (node loads) was only 60% of the answer.

## Tests

`gpu_tree_*` (6 tests, real GPU): all data profiles at 3 MB vs the CPU, run/workgroup boundaries (1 line ... 6145 lines
around 32 and 2048), adversarial shapes (flat, header first/last, increasing/decreasing to the depth limit,
alternating, only blanks, deepest allowed), the error path (depth 32 rejected, depth 31 accepted, deep *blank* line fine,
parser usable afterwards), empty document, and `query_subtree` vs the CPU range filter for ~160 (root, filter) pairs.
Five kernel invariants were mutation-checked (level count, tile padding, result mapping, scan exclusivity, the depth
check); each makes a test fail.

## Not done

- [ ] Reuse the working buffers across calls (saves ~3.5 ms of the 8.6 ms at 1 GiB).
- [ ] `query_subtree` without the 4-byte readback (indirect dispatch from `subtree_end[root]`): it would remove the
      round trip but not the ~90 us floor, so it will not change the conclusion above.
- [ ] More than 32 levels (a second pass for deep lines, or a fallback to the CPU stack for just those lines).
