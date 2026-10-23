
// ---- strategy 2: compact ---------------------------------------------------------
//
// The fused kernel makes a whole warp wait for the one lane whose word has a candidate; at
// 1.3% of words that is about a third of all warp iterations (measured: ~15 ns of whole-GPU
// time per hit, 58 ms for 3.6M hits in 1 GiB, 11x the memory-bound floor).
//
// This strategy separates the two jobs:
//   1. `collect_candidates`: a bandwidth-bound pass over the text. Every position that passes
//      the filter appends its position to a list. Candidates are gathered in shared memory per
//      workgroup and flushed with one global atomic, so there is no divergence beyond the append.
//   2. `make_verify_args`: one thread turns the candidate count into an indirect dispatch size,
//      so the CPU never has to look at it.
//   3. `verify_candidates`: ONE THREAD PER CANDIDATE looks the window up in the hash table and
//      verifies the pattern. Millions of independent threads hide the latency of the table
//      read, and every lane of a warp does real work.
//
// If a workgroup finds more candidates than fit in shared memory, or the list outgrows its
// buffer, `status` is set and the host re-runs the query with the fused kernel. Nothing is
// ever silently dropped.

@group(3) @binding(0) var<storage, read_write> candidates: array<u32>;
@group(3) @binding(1) var<storage, read_write> cand_count: atomic<u32>;
// 0 = ok, 1 = a candidate was dropped (shared tile or buffer full).
@group(3) @binding(2) var<storage, read_write> status: atomic<u32>;
// Indirect dispatch arguments for `verify_candidates`.
@group(3) @binding(3) var<storage, read_write> verify_args: array<u32>;

// Candidates one workgroup can hold before flushing. With the largest filter (32 KiB) this
// keeps the kernel inside 48 KiB of workgroup shared memory.
const TILE_CANDIDATES: u32 = 4000u;
const VERIFY_GROUP: u32 = 256u;

var<workgroup> cand_tile: array<u32, 4000>;
var<workgroup> tile_n: atomic<u32>;
var<workgroup> tile_base: u32;

@compute @workgroup_size(256)
fn collect_candidates(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block_start = (workgroup_id.y * 65535u + workgroup_id.x) * (256u * WORDS_PER_THREAD);
    let words = arrayLength(&text);

    load_filter(lid);
    if (lid == 0u) {
        atomicStore(&tile_n, 0u);
    }
    workgroupBarrier();

    for (var w = 0u; w < WORDS_PER_THREAD; w = w + 1u) {
        let wi = block_start + w * 256u + lid;
        if (wi * 4u < cfg.text_len) {
            var next = 0u;
            if (wi + 1u < words) {
                next = text[wi + 1u];
            }
            let w0 = text[wi];
            for (var k = 0u; k < 4u; k = k + 1u) {
                if (filter_has(window_at(w0, next, k))) {
                    let i = atomicAdd(&tile_n, 1u);
                    if (i < TILE_CANDIDATES) {
                        cand_tile[i] = wi * 4u + k;
                    }
                }
            }
        }
    }
    workgroupBarrier();

    let found = atomicLoad(&tile_n);
    let n = min(found, TILE_CANDIDATES);
    if (lid == 0u) {
        if (found > TILE_CANDIDATES) {
            atomicMax(&status, 1u);
        }
        tile_base = atomicAdd(&cand_count, n);
    }
    workgroupBarrier();
    for (var i = lid; i < n; i = i + 256u) {
        let g = tile_base + i;
        if (g < cfg.cand_capacity) {
            candidates[g] = cand_tile[i];
        } else {
            atomicMax(&status, 1u);
        }
    }
}

@compute @workgroup_size(1)
fn make_verify_args() {
    let total = min(atomicLoad(&cand_count), cfg.cand_capacity);
    let groups = (total + VERIFY_GROUP - 1u) / VERIFY_GROUP;
    if (groups <= 65535u) {
        verify_args[0] = groups;
        verify_args[1] = 1u;
    } else {
        verify_args[0] = 65535u;
        verify_args[1] = (groups + 65534u) / 65535u;
    }
    verify_args[2] = 1u;
}

@compute @workgroup_size(256)
fn verify_candidates(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    for (var i = lid; i < HIST_SLOTS; i = i + 256u) {
        atomicStore(&local_counts[i], 0u);
    }
    workgroupBarrier();

    let total = min(atomicLoad(&cand_count), cfg.cand_capacity);
    let index = (workgroup_id.y * 65535u + workgroup_id.x) * VERIFY_GROUP + lid;
    if (index < total) {
        let pos = candidates[index];
        let window_value = text_word_at(pos);
        var s = table[hash_window(window_value)];
        loop {
            if (s.descriptor == 0u) {
                break;
            }
            if (slot_matches(s, pos, window_value)) {
                let pid = s.descriptor >> 16u;
                atomicAdd(&local_counts[pid], 1u);
                if ((cfg.flags & 1u) != 0u) {
                    let slot = atomicAdd(&hit_count, 1u);
                    if (slot < cfg.max_hits) {
                        hits[slot] = vec2<u32>(pid, pos - ((s.descriptor >> 8u) & 0xFFu));
                    }
                }
            }
            if (s.next == NONE) {
                break;
            }
            s = overflow[s.next];
        }
    }
    workgroupBarrier();

    for (var i = lid; i < cfg.pattern_count; i = i + 256u) {
        let c = atomicLoad(&local_counts[i]);
        if (c != 0u) {
            atomicAdd(&counts[i], c);
        }
    }
}
