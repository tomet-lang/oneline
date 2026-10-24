use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use wgpu::util::DeviceExt;

use crate::document::GpuDocument;
use crate::parser::GpuParser;
use crate::search::choose_window;
use crate::timing::{Profile, StageTimer, mark};
use crate::util::{bind, compute_pass, dispatch_dims, map_all, staging_buffer, take_mapped};

/// Most patterns one set can hold (the per-pattern counters live in shared memory).
pub const MAX_PATTERNS: usize = 2048;
/// Shortest and longest pattern a set accepts. The filter compares a 4-byte window, so a
/// pattern needs at least 4 bytes; the verifier reads at most 64.
pub const MIN_SET_PATTERN_LEN: usize = 4;
pub const MAX_SET_PATTERN_LEN: usize = 64;

/// Sets with at least this many patterns use the compact strategy (see `many_compact.wgsl`);
/// measured on a 1 GiB text, the two strategies cross between 256 patterns (fused 12.3 ms, compact
/// 14.2 ms) and 512 (fused 23.2 ms, compact 14.5 ms);
/// smaller ones have so few candidates that the fused kernel is already at the memory floor.
pub(crate) const COMPACT_MIN_PATTERNS: usize = 384;
/// Shared memory the compact collection kernel uses besides the filter: its candidate tile.
const COMPACT_TILE_BYTES: u32 = 4000 * 4 + 16;
/// Words per thread of the collection kernel (`WORDS_PER_THREAD` for that pipeline).
const COMPACT_WORDS_PER_THREAD: u32 = 32;

/// Filter size for a set of `n` patterns: the smallest power of two giving about a 1-in-1000
/// chance that a random text position looks like some pattern's window, within 2^12..2^18 bits.
/// A small filter leaves shared memory (and so occupancy) to the kernel; the cap is what a
/// workgroup can hold.
fn filter_bits_for(n: usize) -> u32 {
    (10 + n.next_power_of_two().trailing_zeros()).clamp(12, 18)
}

/// Words per thread of the fused kernel: a big filter is loaded once per workgroup, so give it
/// more text to amortize it.
fn words_per_thread_for(filter_bits: u32) -> u32 {
    if filter_bits >= 18 { 64 } else { 32 }
}

/// Candidate-list capacity for a text of `text_len` bytes: up to one candidate per 8 bytes.
/// More than that is an adversarial filter-saturating input; the fused kernel handles it.
fn candidate_capacity(text_len: usize) -> usize {
    (text_len / 8).clamp(1 << 16, 1 << 26)
}

/// The shared kernel pieces; one set of pipelines per (filter bits, histogram slots).
pub(crate) struct ManyKernel {
    pub shader_fused: wgpu::ShaderModule,
    pub shader_compact: wgpu::ShaderModule,
    pub layout_fused: wgpu::PipelineLayout,
    pub layout_compact: wgpu::PipelineLayout,
    pub pipelines: Mutex<HashMap<(u32, u32), ManyPipelines>>,
}

#[derive(Clone)]
pub(crate) struct ManyPipelines {
    fused: wgpu::ComputePipeline,
    collect: Option<wgpu::ComputePipeline>,
    make_args: Option<wgpu::ComputePipeline>,
    verify: Option<wgpu::ComputePipeline>,
}

fn hash_window(w: u32, shift: u32) -> u32 {
    w.wrapping_mul(0x9E37_79B1) >> shift
}

fn hash_window_2(w: u32, shift: u32) -> u32 {
    (w ^ (w >> 15)).wrapping_mul(0x85EB_CA6B) >> shift
}

/// One occurrence of one pattern of a set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ManyHit {
    /// Index of the pattern in the slice given to `compile_patterns`.
    pub pattern: u32,
    /// Byte offset where the pattern starts.
    pub offset: u32,
}

/// Result of a dictionary search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManyResult {
    /// Occurrences of each pattern.
    pub counts: Vec<u32>,
    /// Total occurrences (the sum of `counts`).
    pub total_hits: u64,
    /// Occurrences sorted by `(pattern, offset)`. Empty when `max_hits` was 0. If there are more
    /// than `max_hits`, an arbitrary subset of that size (slots are claimed with atomics).
    pub hits: Vec<ManyHit>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ManyConfig {
    text_len: u32,
    pattern_count: u32,
    max_hits: u32,
    flags: u32,
    cand_capacity: u32,
    _pad: [u32; 3],
}

/// Buffers of the compact strategy.
struct CompactState {
    capacity: usize,
    /// Kept alive for the bind group below.
    _candidates: wgpu::Buffer,
    /// Bind group 3 of `many_compact.wgsl` over the buffers above and the set's small ones.
    group: wgpu::BindGroup,
}

/// A compiled set of patterns: the filter bitmap, hash table and pattern bytes, in VRAM,
/// plus the buffers a search writes its counts into. Build one with
/// [`GpuParser::compile_patterns`] and reuse it for many searches.
pub struct GpuPatternSet {
    pipelines: ManyPipelines,
    words_per_thread: u32,
    count: u32,
    /// Bind group 2 of `many.wgsl`: the filter, hash table and patterns.
    set_group: wgpu::BindGroup,
    counts: wgpu::Buffer,
    hit_count: wgpu::Buffer,
    /// Compact strategy: number of candidates, "something was dropped" flag, indirect dispatch args.
    cand_count: wgpu::Buffer,
    status: wgpu::Buffer,
    verify_args: wgpu::Buffer,
    compact: Mutex<Option<CompactState>>,
    /// `[hit_count][status][cand_count][counts...]`, read back with one map.
    staging: wgpu::Buffer,
    /// Hit records, grown on demand.
    hits: Mutex<Option<(usize, wgpu::Buffer)>>,
}

impl GpuPatternSet {
    pub fn len(&self) -> usize {
        self.count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn uses_compact(&self) -> bool {
        self.count as usize >= COMPACT_MIN_PATTERNS && self.pipelines.collect.is_some()
    }
}

impl GpuParser {
    /// Compiles `patterns` (1..=[`MAX_PATTERNS`] of them, each
    /// [`MIN_SET_PATTERN_LEN`]..=[`MAX_SET_PATTERN_LEN`] bytes) for dictionary search over `doc`.
    ///
    /// `doc` only supplies a sample of its byte frequencies, used to pick each pattern's rarest
    /// 4-byte window; the set can be searched against any document.
    pub fn compile_patterns(&self, doc: &GpuDocument, patterns: &[&[u8]]) -> Result<GpuPatternSet> {
        if patterns.is_empty() || patterns.len() > MAX_PATTERNS {
            return Err(anyhow!(
                "a set needs 1..={MAX_PATTERNS} patterns, got {}",
                patterns.len()
            ));
        }
        if let Some(bad) = patterns
            .iter()
            .position(|p| !(MIN_SET_PATTERN_LEN..=MAX_SET_PATTERN_LEN).contains(&p.len()))
        {
            return Err(anyhow!(
                "pattern {bad} is {} bytes; patterns must be {MIN_SET_PATTERN_LEN}..={MAX_SET_PATTERN_LEN} bytes",
                patterns[bad].len()
            ));
        }
        let freq = doc.byte_freq();

        // Size the kernels for this set.
        let bits = filter_bits_for(patterns.len());
        let shift = 32 - bits;
        let filter_words = 1u32 << (bits - 5);
        let hist_slots = (patterns.len() as u32).next_multiple_of(64);
        let words_per_thread = words_per_thread_for(bits);
        let limit = self.limits.max_compute_workgroup_storage_size;
        let fused_bytes = filter_words * 4 + hist_slots * 4 + 16;
        if fused_bytes > limit {
            return Err(anyhow!(
                "this set needs {fused_bytes} bytes of workgroup shared memory; the adapter offers {limit}"
            ));
        }
        let compact_ok = filter_words * 4 + COMPACT_TILE_BYTES <= limit;
        let pipelines = self.many_pipelines(bits, hist_slots, words_per_thread, compact_ok);

        let mut words: Vec<u32> = Vec::new();
        const NONE: u32 = u32::MAX;
        let mut table = vec![[0u32, 0, 0, NONE]; 1usize << bits];
        let mut overflow: Vec<[u32; 4]> = Vec::new();
        let mut bitmap = vec![0u32; filter_words as usize];
        for (pid, p) in patterns.iter().enumerate() {
            let window = choose_window(p, freq) as usize;
            let head = u32::from_le_bytes(p[window..window + 4].try_into().unwrap());
            let h = hash_window(head, shift);
            let slot = [
                head,
                p.len() as u32 | (window as u32) << 8 | (pid as u32) << 16,
                words.len() as u32,
                NONE,
            ];
            for chunk in p.chunks(4) {
                let mut w = [0u8; 4];
                w[..chunk.len()].copy_from_slice(chunk);
                words.push(u32::from_le_bytes(w));
            }
            // Both Bloom bits; the kernels check one or both (`BLOOM_HASHES`).
            let h2 = hash_window_2(head, shift);
            bitmap[(h >> 5) as usize] |= 1 << (h & 31);
            bitmap[(h2 >> 5) as usize] |= 1 << (h2 & 31);
            // Place the slot: directly in the table if free, else at the end of that index's chain.
            if table[h as usize][1] == 0 {
                table[h as usize] = slot;
            } else {
                let new_index = overflow.len() as u32;
                overflow.push(slot);
                // Walk to the end of the chain (by index: the table and `overflow` are both mutated).
                let mut at_table = true;
                let mut index = 0usize;
                loop {
                    let next = if at_table {
                        table[h as usize][3]
                    } else {
                        overflow[index][3]
                    };
                    if next == NONE {
                        if at_table {
                            table[h as usize][3] = new_index;
                        } else {
                            overflow[index][3] = new_index;
                        }
                        break;
                    }
                    at_table = false;
                    index = next as usize;
                }
            }
        }
        if overflow.is_empty() {
            overflow.push([0, 0, 0, NONE]); // a binding must not be empty
        }

        let storage = |label: &str, bytes: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let bitmap_buf = storage("Pattern Filter Bitmap", bytemuck::cast_slice(&bitmap));
        let table_buf = storage("Pattern Hash Table", bytemuck::cast_slice(&table));
        let overflow_buf = storage("Pattern Hash Overflow", bytemuck::cast_slice(&overflow));
        let words_buf = storage("Pattern Bytes", bytemuck::cast_slice(&words));
        let set_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pattern Set Bind Group"),
            layout: &self.many_set_layout,
            entries: &[
                bind(0, &bitmap_buf),
                bind(1, &table_buf),
                bind(2, &overflow_buf),
                bind(3, &words_buf),
            ],
        });

        let rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let small = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let n = patterns.len() as u64;
        Ok(GpuPatternSet {
            pipelines,
            words_per_thread,
            count: patterns.len() as u32,
            set_group,
            counts: small("Pattern Counts Buffer", n * 4, rw),
            hit_count: small("Pattern Hit Count Buffer", 4, rw),
            cand_count: small("Pattern Candidate Count Buffer", 4, rw),
            status: small("Pattern Status Buffer", 4, rw),
            verify_args: small(
                "Pattern Verify Args Buffer",
                12,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
            compact: Mutex::new(None),
            staging: staging_buffer(&self.device, "Pattern Result Staging Buffer", 12 + n * 4),
            hits: Mutex::new(None),
        })
    }

    fn many_pipelines(
        &self,
        bits: u32,
        hist_slots: u32,
        fused_wpt: u32,
        compact_ok: bool,
    ) -> ManyPipelines {
        let mut cache = self
            .many
            .pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(found) = cache.get(&(bits, hist_slots)) {
            if found.collect.is_some() || !compact_ok {
                return found.clone();
            }
        }
        let build = |module: &wgpu::ShaderModule,
                     layout: &wgpu::PipelineLayout,
                     entry: &str,
                     wpt: u32,
                     hashes: u32| {
            let constants = HashMap::from([
                ("WORDS_PER_THREAD".to_string(), f64::from(wpt)),
                ("FILTER_WORDS".to_string(), f64::from(1u32 << (bits - 5))),
                ("HASH_SHIFT".to_string(), f64::from(32 - bits)),
                ("HIST_SLOTS".to_string(), f64::from(hist_slots)),
                ("BLOOM_HASHES".to_string(), f64::from(hashes)),
            ]);
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(layout),
                    module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    cache: None,
                })
        };
        let m = &self.many;
        let pipelines = ManyPipelines {
            fused: build(
                &m.shader_fused,
                &m.layout_fused,
                "search_many",
                fused_wpt,
                1,
            ),
            // The compact strategy pays per candidate, so it uses the two-hash filter.
            collect: compact_ok.then(|| {
                build(
                    &m.shader_compact,
                    &m.layout_compact,
                    "collect_candidates",
                    COMPACT_WORDS_PER_THREAD,
                    2,
                )
            }),
            make_args: compact_ok.then(|| {
                build(
                    &m.shader_compact,
                    &m.layout_compact,
                    "make_verify_args",
                    1,
                    2,
                )
            }),
            verify: compact_ok.then(|| {
                build(
                    &m.shader_compact,
                    &m.layout_compact,
                    "verify_candidates",
                    1,
                    2,
                )
            }),
        };
        cache.insert((bits, hist_slots), pipelines.clone());
        pipelines
    }

    /// Searches `doc` for every pattern of `set` in one pass over the text.
    ///
    /// `max_hits == 0` only counts (fastest); otherwise up to `max_hits` occurrences are also
    /// returned.
    pub fn search_many(
        &self,
        doc: &GpuDocument,
        set: &GpuPatternSet,
        max_hits: usize,
    ) -> Result<ManyResult> {
        self.search_many_inner(doc, set, max_hits, None)
    }

    /// `search_many`, with a breakdown of where the time went.
    pub fn profile_search_many(
        &self,
        doc: &GpuDocument,
        set: &GpuPatternSet,
        max_hits: usize,
    ) -> Result<Profile> {
        let timer = StageTimer::new(self);
        self.search_many_inner(doc, set, max_hits, Some(&timer))?;
        timer.finish(self)
    }

    fn search_many_inner(
        &self,
        doc: &GpuDocument,
        set: &GpuPatternSet,
        max_hits: usize,
        timer: Option<&StageTimer>,
    ) -> Result<ManyResult> {
        let max_u32 = u32::try_from(max_hits)
            .map_err(|_| anyhow!("max_hits {max_hits} does not fit in u32"))?;
        let Some(text_group) = doc.text_group() else {
            return Ok(ManyResult {
                counts: vec![0; set.len()],
                total_hits: 0,
                hits: Vec::new(),
            });
        };
        self.check_fits(u64::from(max_u32) * 8, "the hit records")?;

        // The hit buffer, grown on demand.
        let mut hits_slot = set.hits.lock().unwrap_or_else(|e| e.into_inner());
        if hits_slot
            .as_ref()
            .is_none_or(|(cap, _)| *cap < max_hits.max(1))
        {
            let cap = max_hits.max(1).next_power_of_two();
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pattern Hits Buffer"),
                size: cap as u64 * 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            *hits_slot = Some((cap, buffer));
        }
        let hits_buffer = &hits_slot.as_ref().expect("just ensured").1;

        let text_len = doc.text_len();
        #[allow(unused_mut)]
        let mut capacity = candidate_capacity(text_len);
        #[cfg(test)]
        {
            let cap = self
                .candidate_capacity_cap
                .load(std::sync::atomic::Ordering::Relaxed);
            if cap != 0 {
                capacity = cap;
            }
        }
        let cfg = self.uniform(
            "Many Config Buffer",
            &ManyConfig {
                text_len: text_len as u32,
                pattern_count: set.count,
                max_hits: max_u32,
                flags: u32::from(max_hits > 0),
                cand_capacity: capacity as u32,
                _pad: [0; 3],
            },
        );
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Many Bind Group 1"),
            layout: &self.many_run_layout,
            entries: &[
                bind(0, &cfg),
                bind(1, &set.counts),
                bind(2, &set.hit_count),
                bind(3, hits_buffer),
            ],
        });
        mark(timer, "many: create config + bind group");

        if set.uses_compact() {
            let mut state = set.compact.lock().unwrap_or_else(|e| e.into_inner());
            if state.as_ref().is_none_or(|s| s.capacity < capacity) {
                let candidates = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Pattern Candidates Buffer"),
                    size: capacity as u64 * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Many Bind Group 3"),
                    layout: &self.many_compact_layout,
                    entries: &[
                        bind(0, &candidates),
                        bind(1, &set.cand_count),
                        bind(2, &set.status),
                        bind(3, &set.verify_args),
                    ],
                });
                *state = Some(CompactState {
                    capacity,
                    _candidates: candidates,
                    group,
                });
            }
            let group3 = &state.as_ref().expect("just ensured").group;
            self.reset_set_buffers(set);
            mark(timer, "many: reset counters");

            let blocks = (text_len as u32)
                .div_ceil(4)
                .div_ceil(256 * COMPACT_WORDS_PER_THREAD);
            let (wg_x, wg_y) = dispatch_dims(blocks);
            let p = &set.pipelines;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for (label, pipeline, dims) in [
                (
                    "many: collect candidates",
                    p.collect.as_ref().expect("compact"),
                    (wg_x, wg_y),
                ),
                (
                    "many: verify args",
                    p.make_args.as_ref().expect("compact"),
                    (1, 1),
                ),
            ] {
                let mut pass = compute_pass(&mut encoder, label, timer);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, text_group, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.set_bind_group(2, &set.set_group, &[]);
                pass.set_bind_group(3, group3, &[]);
                pass.dispatch_workgroups(dims.0, dims.1, 1);
            }
            {
                let mut pass = compute_pass(&mut encoder, "many: verify candidates", timer);
                pass.set_pipeline(p.verify.as_ref().expect("compact"));
                pass.set_bind_group(0, text_group, &[]);
                pass.set_bind_group(1, &group1, &[]);
                pass.set_bind_group(2, &set.set_group, &[]);
                pass.set_bind_group(3, group3, &[]);
                // Sized on the GPU by `make_verify_args`: no CPU round trip in between.
                pass.dispatch_workgroups_indirect(&set.verify_args, 0);
            }
            self.copy_results(&mut encoder, set);
            self.queue.submit(Some(encoder.finish()));
            mark(timer, "many: encode + submit (compact)");
            let (recorded, status, counts) = self.read_results(set)?;
            mark(timer, "many: wait for GPU + map readback");
            if status == 0 {
                return self.finish_many(set, hits_buffer, max_hits, recorded, counts, timer);
            }
            // A candidate was dropped (tile or buffer full): the answer is incomplete. Redo it with the
            // fused kernel, which has no such limit.
        }

        self.reset_set_buffers(set);
        let blocks = (text_len as u32)
            .div_ceil(4)
            .div_ceil(256 * set.words_per_thread);
        let (wg_x, wg_y) = dispatch_dims(blocks);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = compute_pass(&mut encoder, "search many (fused)", timer);
            pass.set_pipeline(&set.pipelines.fused);
            pass.set_bind_group(0, text_group, &[]);
            pass.set_bind_group(1, &group1, &[]);
            pass.set_bind_group(2, &set.set_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.copy_results(&mut encoder, set);
        self.queue.submit(Some(encoder.finish()));
        mark(timer, "many: encode + submit (fused)");
        let (recorded, _, counts) = self.read_results(set)?;
        mark(timer, "many: wait for GPU + map readback");
        self.finish_many(set, hits_buffer, max_hits, recorded, counts, timer)
    }

    fn reset_set_buffers(&self, set: &GpuPatternSet) {
        self.queue
            .write_buffer(&set.counts, 0, &vec![0u8; set.count as usize * 4]);
        for buffer in [&set.hit_count, &set.cand_count, &set.status] {
            self.queue.write_buffer(buffer, 0, &0u32.to_ne_bytes());
        }
    }

    fn copy_results(&self, encoder: &mut wgpu::CommandEncoder, set: &GpuPatternSet) {
        encoder.copy_buffer_to_buffer(&set.hit_count, 0, &set.staging, 0, 4);
        encoder.copy_buffer_to_buffer(&set.status, 0, &set.staging, 4, 4);
        encoder.copy_buffer_to_buffer(&set.cand_count, 0, &set.staging, 8, 4);
        encoder.copy_buffer_to_buffer(&set.counts, 0, &set.staging, 12, u64::from(set.count) * 4);
    }

    /// `(recorded hit slots, status, counts)`.
    fn read_results(&self, set: &GpuPatternSet) -> Result<(u32, u32, Vec<u32>)> {
        map_all(&self.device, &[&set.staging])?;
        let out = {
            let view = set.staging.slice(..).get_mapped_range();
            let words = bytemuck::cast_slice::<u8, u32>(&view);
            (
                words[0],
                words[1],
                words[3..3 + set.count as usize].to_vec(),
            )
        };
        set.staging.unmap();
        Ok(out)
    }

    fn finish_many(
        &self,
        set: &GpuPatternSet,
        hits_buffer: &wgpu::Buffer,
        max_hits: usize,
        recorded: u32,
        counts: Vec<u32>,
        timer: Option<&StageTimer>,
    ) -> Result<ManyResult> {
        let total_hits: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        debug_assert!(max_hits == 0 || u64::from(recorded) == total_hits);
        let _ = set;

        let mut hits = Vec::new();
        let valid = (recorded as usize).min(max_hits);
        if valid > 0 {
            let bytes = valid as u64 * 8;
            let staging = staging_buffer(&self.device, "Pattern Hits Staging Buffer", bytes);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(hits_buffer, 0, &staging, 0, bytes);
            self.queue.submit(Some(encoder.finish()));
            map_all(&self.device, &[&staging])?;
            hits = take_mapped::<[u32; 2]>(&staging, valid)
                .into_iter()
                .map(|[pattern, offset]| ManyHit { pattern, offset })
                .collect();
            hits.sort_unstable();
            mark(timer, "many: read back and sort the hits");
        }
        Ok(ManyResult {
            counts,
            total_hits,
            hits,
        })
    }
}
