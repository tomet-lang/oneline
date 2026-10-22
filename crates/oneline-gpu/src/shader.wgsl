struct Config {
    line_count: u32,
    text_len: u32,
};

struct FlatNode {
    depth: u32,
    kind: u32,
    span_start: u32,
    span_len: u32,
};

@group(0) @binding(0) var<storage, read> input_text: array<u32>;
@group(0) @binding(1) var<storage, read> line_offsets: array<u32>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> nodes: array<FlatNode>;

// Helper to extract a single byte from the u32-packed storage buffer.
fn get_byte(byte_offset: u32) -> u32 {
    if (byte_offset >= config.text_len) {
        return 0u;
    }
    let word_idx = byte_offset >> 2u;
    let byte_in_word = byte_offset & 3u;
    let word = input_text[word_idx];
    return (word >> (byte_in_word * 8u)) & 0xFFu;
}

fn thread_index(global_id: vec3<u32>) -> u32 {
    return global_id.y * 65535u * 64u + global_id.x;
}

// Classifies one line. Pure function of the text, so it can run fused with a query.
fn parse_one(tid: u32) -> FlatNode {
    let line_start = line_offsets[tid];
    var line_end = config.text_len;
    if (tid + 1u < config.line_count) {
        line_end = line_offsets[tid + 1u];
    }

    // Exclude trailing '\r' and '\n'.
    var eff_end = line_end;
    while (eff_end > line_start) {
        let last_b = get_byte(eff_end - 1u);
        if (last_b == 10u || last_b == 13u) {
            eff_end = eff_end - 1u;
        } else {
            break;
        }
    }

    if (eff_end <= line_start) {
        return FlatNode(0u, 0u, line_start, 0u); // Blank
    }

    // Count consecutive prefix '|' (0x7C) characters (nesting depth).
    var depth = 0u;
    var cursor = line_start;
    while (cursor < eff_end && get_byte(cursor) == 124u) { // 124 == '|'
        depth = depth + 1u;
        cursor = cursor + 1u;
    }

    // Skip whitespace immediately following prefix (' ' = 32, '\t' = 9).
    while (cursor < eff_end) {
        let b = get_byte(cursor);
        if (b == 32u || b == 9u) {
            cursor = cursor + 1u;
        } else {
            break;
        }
    }

    if (cursor >= eff_end) {
        return FlatNode(depth, 0u, cursor, 0u); // Blank line with prefix only
    }

    let span_start = cursor;
    let span_len = eff_end - cursor;
    let first_b = get_byte(cursor);

    var kind = 1u; // Text (default)

    if (first_b == 61u) {
        // '=' (Heading)
        kind = 2u;
    } else if (first_b == 62u) {
        // '>' (Callout / Header)
        kind = 5u;
    } else if (first_b == 64u) {
        // '@' (Element)
        kind = 3u;
    } else if (first_b == 45u || first_b == 42u) {
        // '-' (45) or '*' (42) (List)
        if (cursor + 1u < eff_end) {
            let next_b = get_byte(cursor + 1u);
            if (next_b == 32u || next_b == 9u) {
                kind = 4u;
            }
        }
    } else if (first_b == 96u) {
        // '`' (Codeblock)
        if (cursor + 2u < eff_end && get_byte(cursor + 1u) == 96u && get_byte(cursor + 2u) == 96u) {
            kind = 6u;
        }
    } else if (first_b >= 48u && first_b <= 57u) {
        // Numbered list item: digits, '.', ' ' (e.g. "1. "). Mirrors `classify_line`
        // in oneline-simd; this branch was missing and GPU/CPU disagreed.
        var idx = cursor + 1u;
        while (idx < eff_end) {
            let d = get_byte(idx);
            if (d < 48u || d > 57u) {
                break;
            }
            idx = idx + 1u;
        }
        if (idx + 1u < eff_end && get_byte(idx) == 46u && get_byte(idx + 1u) == 32u) {
            kind = 4u;
        }
    }

    return FlatNode(depth, kind, span_start, span_len);
}

@compute @workgroup_size(64)
fn parse_lines(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = thread_index(global_id);
    if (tid >= config.line_count) {
        return;
    }
    nodes[tid] = parse_one(tid);
}

struct QueryConfig {
    target_kind: u32,
    target_depth: u32,
    max_matches: u32,
    _pad: u32,
};

@group(1) @binding(0) var<uniform> query: QueryConfig;
@group(1) @binding(1) var<storage, read_write> match_count: atomic<u32>;
@group(1) @binding(2) var<storage, read_write> matched_indices: array<u32>;

fn node_matches(node: FlatNode) -> bool {
    if (query.target_kind != 0u && node.kind != query.target_kind) {
        return false;
    }
    if (query.target_depth != 0xFFFFFFFFu && node.depth != query.target_depth) {
        return false;
    }
    return true;
}

fn record_match(tid: u32) {
    let slot = atomicAdd(&match_count, 1u);
    if (slot < query.max_matches) {
        matched_indices[slot] = tid;
    }
}

// Query over nodes already resident in VRAM (see `GpuDocument`).
@compute @workgroup_size(64)
fn filter_nodes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = thread_index(global_id);
    if (tid >= config.line_count) {
        return;
    }
    if (node_matches(nodes[tid])) {
        record_match(tid);
    }
}

// One-shot query: classify the line and test it in the same thread, so the
// 16-byte-per-line `nodes` array is never written or read. Does not touch the
// `nodes` binding, so its bind group may omit it.
@compute @workgroup_size(64)
fn filter_fused(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = thread_index(global_id);
    if (tid >= config.line_count) {
        return;
    }
    if (node_matches(parse_one(tid))) {
        record_match(tid);
    }
}
