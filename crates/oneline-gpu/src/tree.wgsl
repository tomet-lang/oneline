// Tree structure from per-line depth, entirely on the GPU.
//
//   parent[i]      = nearest previous content line with depth <  depth[i]
//   subtree_end[i] = nearest next     content line with depth <= depth[i]   (or line_count)
//
// (the same two all-nearest-smaller-values relations as `TreeLinks` on the CPU.)
//
// The CPU version is a monotonic stack. Here the depth is a small integer (< D), which
// allows an O(n) formulation that is data-parallel and needs no stack:
//
//   keep, per depth level k, the position of the last line seen with depth exactly k;
//   then  parent[i] = max over k < depth[i] of last[k].
//
// Steps:
//   0. extract keys: pack each line into ONE byte (its depth, or a marker for blank /
//      too-deep lines). The 16-byte nodes are read once, coalesced; every later pass
//      reads 1 byte per line instead of 16.
//   1. summarize: lines are cut into runs of RUN lines, one thread per run; each run
//      records last[k] for every level (D values, stored level-major);
//   2. scan: per level, an exclusive prefix-max over the runs gives each run the `last`
//      state at its start (one workgroup per level);
//   3. assign: each run re-walks its lines from that state and writes the answers.
// `subtree_end` is the mirror image (first[k], suffix-min, walk right to left).
//
// Memory access. A thread walks its own run, so reading keys and writing results straight
// from/to global memory makes the 64 threads of a workgroup touch addresses RUN elements
// apart: uncoalesced (writes alone cost ~8 of 9.3 ms at 13M lines). So a workgroup owns
// 64 consecutive runs = 2048 consecutive lines, loads their keys into a shared-memory
// tile with coalesced reads, and stages results in a second tile that is copied out with
// coalesced writes. Rows are padded (+1 word) so that threads of a warp hit different
// banks. Summaries are stored level-major (`k * runs + r`) so that the scan, which walks
// one level at a time, reads contiguous memory.
//
// Blank lines are not part of the tree (kind 0): they never update the state, and get
// parent NONE / subtree_end = i + 1. A content line with depth >= D cannot be handled
// exactly, so it raises `status` and the host falls back to the CPU.

const D: u32 = 32u;
const RUN: u32 = 32u;
const GROUP: u32 = 64u;
const GROUP_LINES: u32 = 2048u;          // GROUP * RUN
const NONE: u32 = 0xFFFFFFFFu;
// Key bytes >= D mean "not part of the tree".
const KEY_TOO_DEEP: u32 = 254u;
const KEY_BLANK: u32 = 255u;

struct FlatNode {
    depth: u32,
    kind: u32,
    span_start: u32,
    span_len: u32,
};

struct TreeConfig {
    line_count: u32,
    runs: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<FlatNode>;
@group(0) @binding(1) var<uniform> cfg: TreeConfig;
// D * runs values, level-major: per level and run, the last (forward) / first (backward)
// line with that depth.
@group(0) @binding(2) var<storage, read_write> summary: array<u32>;
// D * runs values, level-major: the same state at the *start* of each run, from the scan.
@group(0) @binding(3) var<storage, read_write> scanned: array<u32>;
// [0]: 0 = ok, 1 = a content line was deeper than D - 1.
// [1]: the deepest content line (so later passes handle only that many levels).
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>, 2>;
// One key byte per line, four to a word.
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
// parent or subtree_end, depending on the dispatch.
@group(1) @binding(0) var<storage, read_write> result: array<u32>;

var<workgroup> wg_shared: array<u32, 256>;
// 64 runs x 8 key words each, rows padded to 9 words.
var<workgroup> key_tile: array<u32, 576>;
// 64 runs x 32 results each, rows padded to 33 words.
var<workgroup> out_tile: array<u32, 2112>;

fn thread_index(global_id: vec3<u32>, group_size: u32) -> u32 {
    return global_id.y * 65535u * group_size + global_id.x;
}

fn workgroup_linear(workgroup_id: vec3<u32>) -> u32 {
    return workgroup_id.y * 65535u + workgroup_id.x;
}

// Coalesced load of this workgroup's 2048 keys (512 words) into `key_tile`. Past the end
// of the key array the tile holds blanks. Callers must `workgroupBarrier()` afterwards.
fn load_key_tile(group_base_line: u32, lid: u32) {
    let words = (cfg.line_count + 3u) / 4u;
    for (var idx = lid; idx < 512u; idx = idx + GROUP) {
        let g = group_base_line / 4u + idx;
        var value = 0xFFFFFFFFu;
        if (g < words) {
            value = keys[g];
        }
        key_tile[(idx / 8u) * 9u + (idx % 8u)] = value;
    }
}

// Key of element `m` (0..RUN) of run `lid`'s run.
fn tile_key(lid: u32, m: u32) -> u32 {
    return (key_tile[lid * 9u + (m >> 2u)] >> ((m & 3u) * 8u)) & 0xFFu;
}

// Coalesced copy of `out_tile` to `result`. Callers must `workgroupBarrier()` before.
fn store_out_tile(group_base_line: u32, lid: u32) {
    for (var idx = lid; idx < GROUP_LINES; idx = idx + GROUP) {
        let line = group_base_line + idx;
        if (line < cfg.line_count) {
            result[line] = out_tile[(idx / RUN) * 33u + (idx % RUN)];
        }
    }
}

// Number of depth levels in use (deepest content line + 1). Valid after `extract_keys`.
fn levels() -> u32 {
    return atomicLoad(&status[1]) + 1u;
}

// ---- step 0 ------------------------------------------------------------------

// One thread per word of keys, i.e. per four lines.
@compute @workgroup_size(256)
fn extract_keys(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word = thread_index(global_id, 256u);
    if (word * 4u >= cfg.line_count) {
        return;
    }
    var packed = 0u;
    var bad = 0u;
    var max_depth = 0u;
    for (var b = 0u; b < 4u; b = b + 1u) {
        let i = word * 4u + b;
        var key = KEY_BLANK;
        if (i < cfg.line_count) {
            let node = nodes[i];
            if (node.kind != 0u) {
                if (node.depth >= D) {
                    key = KEY_TOO_DEEP;
                    bad = 1u;
                } else {
                    key = node.depth;
                    max_depth = max(max_depth, node.depth);
                }
            }
        }
        packed = packed | (key << (b * 8u));
    }
    keys[word] = packed;
    if (max_depth > 0u) {
        atomicMax(&status[1], max_depth);
    }
    if (bad != 0u) {
        atomicMax(&status[0], 1u);
    }
}

// ---- forward: parent ---------------------------------------------------------

// Values are line + 1 so that 0 can mean "none" and plain max works.
@compute @workgroup_size(64)
fn summarize_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group_base = workgroup_linear(workgroup_id) * GROUP_LINES;
    load_key_tile(group_base, lid);
    workgroupBarrier();

    let r = workgroup_linear(workgroup_id) * GROUP + lid;
    let n_levels = levels();
    var last: array<u32, 32>;
    for (var k = 0u; k < n_levels; k = k + 1u) {
        last[k] = 0u;
    }
    for (var m = 0u; m < RUN; m = m + 1u) {
        let key = tile_key(lid, m);
        if (key < D) {
            last[key] = r * RUN + m + 1u;
        }
    }
    if (r < cfg.runs) {
        for (var k = 0u; k < n_levels; k = k + 1u) {
            summary[k * cfg.runs + r] = last[k];
        }
    }
}

// One workgroup per level: exclusive prefix-max over runs.
@compute @workgroup_size(256)
fn scan_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = workgroup_id.x;
    if (k >= levels()) {
        return; // this level never occurs; uniform across the workgroup
    }
    let level = k * cfg.runs;
    let per_thread = (cfg.runs + 255u) / 256u;
    let begin = min(lid * per_thread, cfg.runs);
    let end = min(begin + per_thread, cfg.runs);

    var strip = 0u;
    for (var r = begin; r < end; r = r + 1u) {
        strip = max(strip, summary[level + r]);
    }
    wg_shared[lid] = strip;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset = offset << 1u) {
        var v = wg_shared[lid];
        if (lid >= offset) {
            v = max(v, wg_shared[lid - offset]);
        }
        workgroupBarrier();
        wg_shared[lid] = v;
        workgroupBarrier();
    }
    var running = 0u;
    if (lid > 0u) {
        running = wg_shared[lid - 1u];
    }
    for (var r = begin; r < end; r = r + 1u) {
        scanned[level + r] = running;
        running = max(running, summary[level + r]);
    }
}

@compute @workgroup_size(64)
fn assign_parent(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group_base = workgroup_linear(workgroup_id) * GROUP_LINES;
    load_key_tile(group_base, lid);
    workgroupBarrier();

    let r = workgroup_linear(workgroup_id) * GROUP + lid;
    let n_levels = levels();
    var state: array<u32, 32>;
    for (var k = 0u; k < n_levels; k = k + 1u) {
        state[k] = 0u;
        if (r < cfg.runs) {
            state[k] = scanned[k * cfg.runs + r];
        }
    }
    for (var m = 0u; m < RUN; m = m + 1u) {
        let key = tile_key(lid, m);
        var value = NONE;
        if (key < D) {
            var p = 0u;
            for (var j = 0u; j < key; j = j + 1u) {
                p = max(p, state[j]);
            }
            value = select(p - 1u, NONE, p == 0u);
            state[key] = r * RUN + m + 1u;
        }
        out_tile[lid * 33u + m] = value;
    }
    workgroupBarrier();
    store_out_tile(group_base, lid);
}

// ---- backward: subtree_end ---------------------------------------------------

// Values are line indices; NONE means "none" and plain min works.
@compute @workgroup_size(64)
fn summarize_backward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group_base = workgroup_linear(workgroup_id) * GROUP_LINES;
    load_key_tile(group_base, lid);
    workgroupBarrier();

    let r = workgroup_linear(workgroup_id) * GROUP + lid;
    let n_levels = levels();
    var first: array<u32, 32>;
    for (var k = 0u; k < n_levels; k = k + 1u) {
        first[k] = NONE;
    }
    for (var m = RUN; m > 0u; m = m - 1u) {
        let key = tile_key(lid, m - 1u);
        if (key < D) {
            first[key] = r * RUN + m - 1u;
        }
    }
    if (r < cfg.runs) {
        for (var k = 0u; k < n_levels; k = k + 1u) {
            summary[k * cfg.runs + r] = first[k];
        }
    }
}

// One workgroup per level: exclusive suffix-min over runs (runs to the right of each run).
@compute @workgroup_size(256)
fn scan_backward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let k = workgroup_id.x;
    if (k >= levels()) {
        return; // this level never occurs; uniform across the workgroup
    }
    let level = k * cfg.runs;
    let per_thread = (cfg.runs + 255u) / 256u;
    let begin = min(lid * per_thread, cfg.runs);
    let end = min(begin + per_thread, cfg.runs);

    var strip = NONE;
    for (var r = begin; r < end; r = r + 1u) {
        strip = min(strip, summary[level + r]);
    }
    wg_shared[lid] = strip;
    workgroupBarrier();
    // Inclusive suffix scan across threads.
    for (var offset = 1u; offset < 256u; offset = offset << 1u) {
        var v = wg_shared[lid];
        if (lid + offset < 256u) {
            v = min(v, wg_shared[lid + offset]);
        }
        workgroupBarrier();
        wg_shared[lid] = v;
        workgroupBarrier();
    }
    var running = NONE;
    if (lid < 255u) {
        running = wg_shared[lid + 1u];
    }
    for (var r = end; r > begin; r = r - 1u) {
        scanned[level + r - 1u] = running;
        running = min(running, summary[level + r - 1u]);
    }
}

@compute @workgroup_size(64)
fn assign_end(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let group_base = workgroup_linear(workgroup_id) * GROUP_LINES;
    load_key_tile(group_base, lid);
    workgroupBarrier();

    let r = workgroup_linear(workgroup_id) * GROUP + lid;
    let n_levels = levels();
    var state: array<u32, 32>;
    for (var k = 0u; k < n_levels; k = k + 1u) {
        state[k] = NONE;
        if (r < cfg.runs) {
            state[k] = scanned[k * cfg.runs + r];
        }
    }
    for (var m = RUN; m > 0u; m = m - 1u) {
        let line = r * RUN + m - 1u;
        let key = tile_key(lid, m - 1u);
        var value = line + 1u;
        if (key < D) {
            // Nearest later line with depth <= this one (levels 0..=key).
            var q = NONE;
            for (var j = 0u; j <= key; j = j + 1u) {
                q = min(q, state[j]);
            }
            value = select(q, cfg.line_count, q == NONE);
            state[key] = line;
        }
        out_tile[lid * 33u + m - 1u] = value;
    }
    workgroupBarrier();
    store_out_tile(group_base, lid);
}
