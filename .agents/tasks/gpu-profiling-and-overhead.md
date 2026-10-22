# gpu-profiling-and-overhead

Goal: find out where GPU time actually goes, then remove the fixed costs, before adding GPU features.
Replaces `phase2-gpu-indirect-dispatch` as the next GPU task.

## Evidence (RTX 3060, PCIe, release build, 2026-10-23)

100 MB `mixed` document (1.3 M lines), `oneline bench --synthetic-mb 100 --profile mixed`:

| stage | time |
|---|---|
| CPU `parse_par` (12 threads) | 6.6 ms |
| GPU `parse_raw` (upload + scan + parse + download) | 36.8 ms |
| GPU `query_raw` (fused, one-shot) | 29.7 ms |
| GPU `upload` (resident: upload + scan + parse) | 23.8 ms |
| GPU resident query | 3.8 ms (CLI avg) |

Warm queries on an already-parsed document (`cargo bench -p oneline-benches --bench comparison -- compare/warm`):

| text size | CPU parallel filter over nodes | GPU resident query |
|---|---|---|
| 1 MB | 75 us | 1.11 ms |
| 16 MB | 161 us | 1.19 ms |
| 128 MB | 405 us | 1.98 ms |

Reading: the GPU resident query has a ~1 ms floor that does not scale with size (64 KB costs 0.8 ms), so
the kernel is not the bottleneck. A query touches only the 16 B/line node array (~21 MB for 100 MB of text),
so 3.8 ms is ~5 GB/s of actual traffic, nowhere near the 3060's ~360 GB/s. On this machine the CPU wins every
measured regime, cold and warm. This is a finding to report, not a bug.

## Steps (original list; see Status below for the outcome)

- [ ] Add per-pass GPU timestamps (`wgpu::Features::TIMESTAMP_QUERY`; supported on the NVIDIA Vulkan driver).
      Report scan pass 1/2/3, parse, filter separately from submit/poll/map latency.
- [ ] Time the host side of `upload_text` separately (the `mapped_at_creation` memcpy plus wgpu's internal
      staging copy to device-local memory) from the PCIe transfer and the kernels.
- [ ] Find the ~1 ms query floor. Suspects: per-call buffer creation (`match_count`, indices, two staging
      buffers, bind group), zero-initialization of fresh buffers, `device.poll(wait)` latency.
- [ ] Preallocate and reuse query buffers (e.g. in `GpuDocument` or a `QueryContext`); re-measure.
- [ ] Measure pass 2 (hierarchical scan) against the old single-thread loop on a >1 GB input.
- [ ] Try `array<vec4<u32>>` loads in the scan kernels (never measured; was impossible on a software adapter).
- [ ] Benchmark at 1 GB+ where CPU memory bandwidth (~60 GB/s) vs VRAM bandwidth should start to matter.
- [ ] Revisit `phase2` only if the 4-byte readback shows up in the timestamps.

## Done when

A table of per-stage GPU times for 100 MB and 1 GB, and a statement of the break-even point (if any)
between CPU and GPU for cold and warm queries, with the adapter named.

## Status (2026-10-23): measured, floor removed; cold path is structurally PCIe-bound

Tooling added: `oneline profile` (per-pass GPU timestamps + host-side steps, median of N runs,
`--synthetic-kb`, `--max-matches`), `GpuParser::profile_upload / profile_query_document / profile_query_raw`,
`GpuParser::timestamps_supported`. Normal calls pass `None` and are unaffected.

### Where the time goes (RTX 3060, 100 MB `mixed`, median of 20)

Resident upload (upload + scan + parse), host wall 20.5 ms:

| step | time |
|---|---|
| create mapped buffer (wgpu zero-fills 100 MB) | 3.4 ms |
| host memcpy into mapped memory | 5.6 ms |
| wait for scan passes 1-2 (includes the staging -> VRAM DMA) | 9.8 ms |
| everything else (bind groups, submits, parse wait) | ~1.5 ms |
| **GPU execution, all passes** | **1.09 ms (5%)** |

GPU passes: scan pass 1 309 us (~340 GB/s, i.e. the 3060's VRAM bandwidth), pass 2 12 us, pass 3 529 us,
parse 242 us, filter 74-97 us. **The kernels are fast and bandwidth-bound; the cost is moving bytes to the GPU.**
A 100 MB cold query is ~20 ms of data movement around ~1 ms of compute, so the cold GPU path cannot beat the
12-thread CPU parse (6.6 ms) on this machine, whatever the kernels do. Not worth optimizing further.

### The ~1 ms warm-query floor: found and removed

Resident query, host wall: **526 us -> 262 us** (1 MB-100 MB, 1024 matches). Buffer + bind group creation
was 323 us (61%); queries now reuse scratch buffers (`QueryScratch`) and read back with one map. What is left
is ~190 us of submit -> completion -> map latency around a 76 us kernel.

Result sets matter: copying the whole `max_matches` capacity every query cost ~1 ms at 1<<20. Now the first
4096 indices ride along with the count and a second readback fetches the rest only when needed; the sort
(atomics deliver indices unordered) uses a parallel sort above 32768 results.

Warm query, GPU resident vs CPU parallel filter over parsed nodes (criterion `compare/warm`, all matches returned):

| text size | CPU | GPU resident (before) | GPU resident (now) |
|---|---|---|---|
| 1 MB | 85 us | 1.11 ms | **66 us** |
| 16 MB | 196 us | 1.19 ms | 204 us |
| 128 MB | 522 us | 1.98 ms | 580 us |

Parity, not a win: a kind/depth filter reads only 16 B/line, too little traffic for VRAM bandwidth to matter.
At 128 MB with ~83k matches the remainder is readback (~280 us) and the sort (~440 us), not the kernel (97 us).

### Tried and reverted

Parallel `memcpy` into the mapped upload buffer: no change (5.64 -> 5.73 ms). The limit is the destination
memory, not CPU threads.

### Steps

- [x] Per-pass GPU timestamps.
- [x] Host-side breakdown of `upload_text` (create / memcpy / unmap) and every pipeline step.
- [x] Find the query floor (buffer creation, then readback size, then sort).
- [x] Reuse query buffers; re-measure.
- [ ] Measure pass 2 (hierarchical scan) against the old single-thread loop on a >1 GB input.
- [ ] Try `array<vec4<u32>>` loads in the scan kernels (pass 1 is already at ~340 GB/s, so expect little).
- [ ] Benchmark at 1 GB+ (needs a PCIe/VRAM budget check; `max_storage_buffer_binding_size` applies).
- [x] Revisit `phase2`: the 4-byte count readback is ~0.05 ms of 20 ms (host steps "scan: wait..." is the DMA, not the sync). Stay on hold.
- [ ] Open question: can the cold path overlap upload with scanning by streaming chunks? It can hide at most ~1 ms of
      compute, since the DMA and copies (~19 ms) dominate; probably not worth it.

## Consequence for the research question

The GPU's possible win is a query whose cost is proportional to the *text*, not to the node count. See the
re-prioritization in `phase3-gpu-resident-query-engine.md`: substring search is the deciding experiment, now that
the fixed per-query cost is ~260 us and the kernels are known to run at memory bandwidth.
