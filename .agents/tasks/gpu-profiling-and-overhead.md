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

## Steps

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
