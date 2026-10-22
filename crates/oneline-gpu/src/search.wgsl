// Substring search over the raw text: every position where the pattern occurs
// (overlapping occurrences included).
//
// Candidate filter: compare one 4-byte WINDOW of the pattern (up to 4 bytes) against
// the text at each of the 4 byte alignments of a 32-bit word, built from the current
// and the next word with a funnel shift. The 4 comparisons are branch-free and fold
// into one bit mask, so there is exactly one rarely-taken branch per word. Only words
// whose mask is non-zero verify the whole pattern at the start position the window
// implies.
//
// The window is not simply the pattern's head: the host picks the 4-byte window whose
// bytes are rarest in a sample of the text. On SIMT hardware a warp pays for every
// lane that takes the slow path, so what matters is how often ANY of 32 lanes does;
// the filter has to be selective on the actual data.
//
// (Measured on the RTX 3060, 100 MB: a plain read of the text takes ~350 us. Filtering
// on a single byte took 1.3 ms, because ~12% of words contain any given letter and so
// ~98% of warp iterations entered the slow path. The cost was the number of slow-path
// entries, not the length of the verification: verifying 2 bytes instead of all took
// the same time.)
//
// A workgroup owns a contiguous block of 256 * WORDS_PER_THREAD words. In each round
// thread i reads word i of a 256-word row, so a load instruction touches 1 KiB of
// contiguous memory (coalesced).
//
// Threads of a workgroup agree on how many slots they need and the workgroup
// reserves them with ONE global atomic, instead of one atomic per match (which
// serializes on frequent patterns). A thread that found something re-scans its
// words to write the positions at its reserved slots.

struct SearchConfig {
    text_len: u32,
    pattern_len: u32,
    max_matches: u32,
    // Index in the pattern of the 4-byte filter window (0 if the pattern is shorter).
    window: u32,
    // Up to 64 bytes, packed little-endian into 16 words (vec4 for uniform layout).
    pattern: array<vec4<u32>, 4>,
};

@group(0) @binding(0) var<storage, read> text: array<u32>;

@group(1) @binding(0) var<uniform> cfg: SearchConfig;
@group(1) @binding(1) var<storage, read_write> match_count: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> matches: array<u32>;

const WORDS_PER_THREAD: u32 = 16u;

var<workgroup> wg_total: atomic<u32>;
var<workgroup> wg_base: u32;

fn text_byte(i: u32) -> u32 {
    return (text[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

fn pattern_byte(i: u32) -> u32 {
    let word = cfg.pattern[i >> 4u][(i >> 2u) & 3u];
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

// Does the pattern occur at start position `p`?
fn matches_at(p: u32) -> bool {
    if (p + cfg.pattern_len > cfg.text_len) {
        return false;
    }
    for (var i = 0u; i < cfg.pattern_len; i = i + 1u) {
        if (text_byte(p + i) != pattern_byte(i)) {
            return false;
        }
    }
    return true;
}

// The 4 text bytes starting `k` bytes (0..=3) into `w0`; `w1` is the next word.
fn window_at(w0: u32, w1: u32, k: u32) -> u32 {
    // The shift is masked so k == 0 never shifts by 32 (that lane is w0 anyway).
    return select((w0 >> (k * 8u)) | (w1 << ((32u - k * 8u) & 31u)), w0, k == 0u);
}

// Bit k set: the pattern occurs at the start position implied by a window hit at
// base_pos + k. Each occurrence has exactly one window position, so it is reported by
// exactly one word.
fn matches_in_word(w0: u32, w1: u32, base_pos: u32, window: u32, head: u32, head_mask: u32) -> u32 {
    var cand = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        cand = cand | select(0u, 1u << k, (window_at(w0, w1, k) & head_mask) == head);
    }
    if (cand == 0u) {
        return 0u;
    }
    var mask = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if ((cand & (1u << k)) != 0u && base_pos + k >= window && matches_at(base_pos + k - window)) {
            mask = mask | (1u << k);
        }
    }
    return mask;
}

@compute @workgroup_size(256)
fn search(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let block_start = (workgroup_id.y * 65535u + workgroup_id.x) * (256u * WORDS_PER_THREAD);
    let words = arrayLength(&text);
    let window = cfg.window;
    // The filter window: pattern bytes window..window+n (n = min(pattern_len, 4)).
    let n = min(cfg.pattern_len, 4u);
    var head = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        head = head | (pattern_byte(window + i) << (i * 8u));
    }
    var head_mask = 0xFFFFFFFFu;
    if (n < 4u) {
        head_mask = (1u << (n * 8u)) - 1u;
    }

    // Pass 1: count this thread's matches.
    var count = 0u;
    for (var w = 0u; w < WORDS_PER_THREAD; w = w + 1u) {
        let wi = block_start + w * 256u + local_index;
        if (wi * 4u < cfg.text_len) {
            var next = 0u;
            if (wi + 1u < words) {
                next = text[wi + 1u];
            }
            count = count + countOneBits(matches_in_word(text[wi], next, wi * 4u, window, head, head_mask));
        }
    }

    // Reserve a contiguous range for the whole workgroup. No early return above:
    // every thread must reach these barriers.
    if (local_index == 0u) {
        atomicStore(&wg_total, 0u);
    }
    workgroupBarrier();
    var local_slot = 0u;
    if (count > 0u) {
        local_slot = atomicAdd(&wg_total, count);
    }
    workgroupBarrier();
    if (local_index == 0u) {
        let total = atomicLoad(&wg_total);
        if (total > 0u) {
            wg_base = atomicAdd(&match_count, total);
        }
    }
    workgroupBarrier();

    // Pass 2: write start positions.
    if (count > 0u) {
        var slot = wg_base + local_slot;
        for (var w = 0u; w < WORDS_PER_THREAD; w = w + 1u) {
            let wi = block_start + w * 256u + local_index;
            if (wi * 4u < cfg.text_len) {
                var next = 0u;
                if (wi + 1u < words) {
                    next = text[wi + 1u];
                }
                let mask = matches_in_word(text[wi], next, wi * 4u, window, head, head_mask);
                for (var k = 0u; k < 4u; k = k + 1u) {
                    if ((mask & (1u << k)) != 0u) {
                        if (slot < cfg.max_matches) {
                            matches[slot] = wi * 4u + k - window;
                        }
                        slot = slot + 1u;
                    }
                }
            }
        }
    }
}
