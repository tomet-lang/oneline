// Dictionary search: every occurrence of every one of up to 2048 patterns (each 4..=64
// bytes) in one pass over the text. Shared pieces; the two strategies are in
// `many_fused.wgsl` and `many_compact.wgsl`.
//
// Single-pattern search compares one 4-byte window per position. With many patterns the
// per-pattern work would multiply, so instead each pattern contributes ONE 4-byte window
// (its rarest, chosen on the host) to a Bloom filter: a bitmap indexed by hashes of the
// window. A text position costs one funnel shift, a multiply and a bit test in shared memory,
// whatever the number of patterns. Only positions that pass the filter look in the hash table
// (built on the host) and verify the whole pattern in the text.
//
// This is why a GPU can hold its memory-bandwidth speed while an Aho-Corasick automaton on a
// CPU slows down roughly in proportion to its state table: the filter is the same size for 10
// patterns as for 2000.

struct ManyConfig {
    text_len: u32,
    pattern_count: u32,
    max_hits: u32,
    // bit 0: also record every hit (otherwise only count).
    flags: u32,
    // Entries the candidate buffer can hold (compact strategy).
    cand_capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// One slot of the hash table (16 bytes, one read). A slot holds a pattern whose filter window
// hashes to that index; patterns that share an index are chained through `overflow`.
//   head:  the 4 window bytes, little-endian
//   descriptor:  pattern length | window index << 8 | pattern id << 16   (0 = empty slot)
//   first_word: index into `pattern_words` of the pattern's first word
//   next:  index into `overflow` of the next pattern in the chain, or NONE
struct Slot {
    head: u32,
    descriptor: u32,
    first_word: u32,
    next: u32,
};

@group(0) @binding(0) var<storage, read> text: array<u32>;

@group(1) @binding(0) var<uniform> cfg: ManyConfig;
@group(1) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> hit_count: atomic<u32>;
@group(1) @binding(3) var<storage, read_write> hits: array<vec2<u32>>;

@group(2) @binding(0) var<storage, read> filter_bits: array<u32>;     // FILTER_WORDS words
@group(2) @binding(1) var<storage, read> table: array<Slot>;           // 2^(32 - HASH_SHIFT) slots
@group(2) @binding(2) var<storage, read> overflow: array<Slot>;
@group(2) @binding(3) var<storage, read> pattern_words: array<u32>;

// Specialized per pattern set (see `many.rs`): the filter is as small as the set allows, so a
// small dictionary leaves shared memory free for occupancy and a big one gets a big filter.
override WORDS_PER_THREAD: u32 = 32u;
override FILTER_WORDS: u32 = 8192u;   // filter bits / 32
override HASH_SHIFT: u32 = 14u;       // 32 - log2(filter bits)
override HIST_SLOTS: u32 = 2048u;     // >= pattern count
// Hash functions checked by the filter (the host always sets both bits of every window).
// 1: cheapest. 2: squares the false-positive rate, which matters when every false positive
// costs a candidate to be verified.
override BLOOM_HASHES: u32 = 1u;

const NONE: u32 = 0xFFFFFFFFu;

var<workgroup> filter_tile: array<u32, FILTER_WORDS>;
var<workgroup> local_counts: array<atomic<u32>, HIST_SLOTS>;

fn hash_window(w: u32) -> u32 {
    return (w * 0x9E3779B1u) >> HASH_SHIFT;
}

fn hash_window_2(w: u32) -> u32 {
    return ((w ^ (w >> 15u)) * 0x85EBCA6Bu) >> HASH_SHIFT;
}

fn window_at(w0: u32, w1: u32, k: u32) -> u32 {
    // The shift is masked so k == 0 never shifts by 32 (that lane is w0 anyway).
    return select((w0 >> (k * 8u)) | (w1 << ((32u - k * 8u) & 31u)), w0, k == 0u);
}

fn filter_has(w: u32) -> bool {
    let h1 = hash_window(w);
    var bits = filter_tile[h1 >> 5u] >> (h1 & 31u);
    if (BLOOM_HASHES == 2u) {
        let h2 = hash_window_2(w);
        bits = bits & (filter_tile[h2 >> 5u] >> (h2 & 31u));
    }
    return (bits & 1u) != 0u;
}

// The 4 text bytes starting at `byte_pos` (any alignment); bytes past the end read as 0.
fn text_word_at(byte_pos: u32) -> u32 {
    let i = byte_pos >> 2u;
    let shift = (byte_pos & 3u) * 8u;
    let lo = text[i];
    if (shift == 0u) {
        return lo;
    }
    var hi = 0u;
    if (i + 1u < arrayLength(&text)) {
        hi = text[i + 1u];
    }
    return (lo >> shift) | (hi << (32u - shift));
}

// Does the pattern of slot `s` occur with its window at text position `pos`? Compared 4 bytes at a
// time (the pattern words of one pattern are contiguous, so these reads are independent).
fn slot_matches(s: Slot, pos: u32, window_value: u32) -> bool {
    let len = s.descriptor & 0xFFu;
    let window = (s.descriptor >> 8u) & 0xFFu;
    if (s.head != window_value || pos < window) {
        return false;
    }
    let start = pos - window;
    if (start + len > cfg.text_len) {
        return false;
    }
    for (var i = 0u; i < len; i = i + 4u) {
        let remaining = len - i;
        var mask = 0xFFFFFFFFu;
        if (remaining < 4u) {
            mask = (1u << (remaining * 8u)) - 1u;
        }
        if (((text_word_at(start + i) ^ pattern_words[s.first_word + (i >> 2u)]) & mask) != 0u) {
            return false;
        }
    }
    return true;
}

// Copies the filter into shared memory. Callers must `workgroupBarrier()` afterwards.
fn load_filter(lid: u32) {
    for (var i = lid; i < FILTER_WORDS; i = i + 256u) {
        filter_tile[i] = filter_bits[i];
    }
}
