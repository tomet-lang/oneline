
// ---- strategy 1: fused ---------------------------------------------------------
//
// One kernel: filter, verify and count in the same thread. A warp waits for its slowest lane,
// so every word that has a candidate stalls 31 other words; this is the fastest strategy for
// small dictionaries (few candidates) and the fallback for any set, because it needs no
// buffer sized by the number of candidates.
//
// A workgroup owns a contiguous block of 256 * WORDS_PER_THREAD words; thread i reads word i of
// each 256-word row (coalesced). Per-pattern hit counts are accumulated in a shared-memory
// histogram and added to the global counts once per workgroup, so even a very frequent pattern
// costs no contended global atomics. Optionally every hit is also recorded; slots are reserved
// per workgroup with one global atomic, exactly like the single-pattern search.

var<workgroup> wg_total: atomic<u32>;
var<workgroup> wg_base: u32;

// Records one verified hit.
fn record_hit(s: Slot, pos: u32, tally: bool, write: bool, slot: ptr<function, u32>) {
    let pid = s.descriptor >> 16u;
    if (tally) {
        atomicAdd(&local_counts[pid], 1u);
    }
    if (write) {
        if (*slot < cfg.max_hits) {
            hits[*slot] = vec2<u32>(pid, pos - ((s.descriptor >> 8u) & 0xFFu));
        }
        *slot = *slot + 1u;
    }
}

// Finds the hits whose filter window starts in this word. Returns how many there are.
// `tally`: add them to the workgroup histogram. `write`: record them from `slot_in` on.
fn process_word(w0: u32, w1: u32, base_pos: u32, tally: bool, write: bool, slot_in: u32) -> u32 {
    var cand = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (filter_has(window_at(w0, w1, k))) {
            cand = cand | (1u << k);
        }
    }
    if (cand == 0u) {
        return 0u;
    }
    var found = 0u;
    var slot = slot_in;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if ((cand & (1u << k)) == 0u) {
            continue;
        }
        let window_value = window_at(w0, w1, k);
        let pos = base_pos + k;
        var s = table[hash_window(window_value)];
        // One read gave the whole first slot; chains (same hash, different windows) are rare.
        loop {
            if (s.descriptor == 0u) {
                break;
            }
            if (slot_matches(s, pos, window_value)) {
                found = found + 1u;
                record_hit(s, pos, tally, write, &slot);
            }
            if (s.next == NONE) {
                break;
            }
            s = overflow[s.next];
        }
    }
    return found;
}

@compute @workgroup_size(256)
fn search_many(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block_start = (workgroup_id.y * 65535u + workgroup_id.x) * (256u * WORDS_PER_THREAD);
    let words = arrayLength(&text);

    load_filter(lid);
    for (var i = lid; i < HIST_SLOTS; i = i + 256u) {
        atomicStore(&local_counts[i], 0u);
    }
    workgroupBarrier();

    // Pass 1: count (and tally into the workgroup histogram).
    var count = 0u;
    for (var w = 0u; w < WORDS_PER_THREAD; w = w + 1u) {
        let wi = block_start + w * 256u + lid;
        if (wi * 4u < cfg.text_len) {
            var next = 0u;
            if (wi + 1u < words) {
                next = text[wi + 1u];
            }
            count = count + process_word(text[wi], next, wi * 4u, true, false, 0u);
        }
    }

    if ((cfg.flags & 1u) != 0u) {
        // Reserve a contiguous range of hit slots for the whole workgroup.
        if (lid == 0u) {
            atomicStore(&wg_total, 0u);
        }
        workgroupBarrier();
        var local_slot = 0u;
        if (count > 0u) {
            local_slot = atomicAdd(&wg_total, count);
        }
        workgroupBarrier();
        if (lid == 0u) {
            let total = atomicLoad(&wg_total);
            if (total > 0u) {
                wg_base = atomicAdd(&hit_count, total);
            }
        }
        workgroupBarrier();

        // Pass 2: record the hits.
        if (count > 0u) {
            var slot = wg_base + local_slot;
            for (var w = 0u; w < WORDS_PER_THREAD; w = w + 1u) {
                let wi = block_start + w * 256u + lid;
                if (wi * 4u < cfg.text_len) {
                    var next = 0u;
                    if (wi + 1u < words) {
                        next = text[wi + 1u];
                    }
                    slot = slot + process_word(text[wi], next, wi * 4u, false, true, slot);
                }
            }
        }
    }

    // Add this workgroup's per-pattern counts to the global counts.
    workgroupBarrier();
    for (var i = lid; i < cfg.pattern_count; i = i + 256u) {
        let c = atomicLoad(&local_counts[i]);
        if (c != 0u) {
            atomicAdd(&counts[i], c);
        }
    }
}
