struct ScanConfig {
    num_blocks: u32,
    text_len: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> scan_input: array<u32>;
@group(0) @binding(1) var<uniform> scan_cfg: ScanConfig;
@group(0) @binding(2) var<storage, read_write> block_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> scan_meta: array<u32>;

@group(1) @binding(0) var<storage, read_write> line_offsets: array<u32>;

var<workgroup> wg_shared: array<u32, 256>;

// Pass 1: Count newlines in parallel across blocks
@compute @workgroup_size(256)
fn count_block_newlines(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let block_id = group_id.y * 65535u + group_id.x;
    if (block_id >= scan_cfg.num_blocks) {
        return;
    }

    if (scan_cfg.text_len == 0u) {
        if (local_id.x == 0u) {
            block_counts[block_id] = 0u;
        }
        return;
    }

    let scan_limit = scan_cfg.text_len - 1u;
    let total_words = (scan_cfg.text_len + 3u) >> 2u;

    var count = 0u;
    let thread_base_word = block_id * 4096u + local_id.x * 16u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let word_idx = thread_base_word + w;
        if (word_idx < total_words) {
            let val = scan_input[word_idx];
            let m = val ^ 0x0A0A0A0Au;
            if (((m - 0x01010101u) & (~m) & 0x80808080u) != 0u) {
                let p0 = word_idx * 4u;
                if (p0 < scan_limit && (val & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p1 = p0 + 1u;
                if (p1 < scan_limit && ((val >> 8u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p2 = p0 + 2u;
                if (p2 < scan_limit && ((val >> 16u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p3 = p0 + 3u;
                if (p3 < scan_limit && ((val >> 24u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
            }
        }
    }

    wg_shared[local_id.x] = count;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_id.x < stride) {
            wg_shared[local_id.x] = wg_shared[local_id.x] + wg_shared[local_id.x + stride];
        }
        workgroupBarrier();
    }

    if (local_id.x == 0u) {
        block_counts[block_id] = wg_shared[0];
    }
}

// Pass 2: Exclusive prefix scan of block sums.
//
// One workgroup. Each of the 256 threads serially sums a contiguous strip of
// blocks, the 256 strip sums are scanned in shared memory, and each thread then
// rewrites its strip as exclusive offsets. This replaces the old single-thread
// loop over every block: serial depth drops from `num_blocks` to
// `num_blocks / 256 + log2(256)`.
@compute @workgroup_size(256)
fn scan_block_counts(@builtin(local_invocation_id) local_id: vec3<u32>) {
    if (scan_cfg.text_len == 0u) {
        if (local_id.x == 0u) {
            scan_meta[0] = 0u;
        }
        return;
    }

    let nb = scan_cfg.num_blocks;
    let per_thread = (nb + 255u) / 256u;
    let begin = min(local_id.x * per_thread, nb);
    let end = min(begin + per_thread, nb);

    var strip_sum = 0u;
    for (var i = begin; i < end; i = i + 1u) {
        strip_sum = strip_sum + block_counts[i];
    }

    // Inclusive Hillis-Steele scan of the strip sums.
    wg_shared[local_id.x] = strip_sum;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset = offset << 1u) {
        var v = wg_shared[local_id.x];
        if (local_id.x >= offset) {
            v = v + wg_shared[local_id.x - offset];
        }
        workgroupBarrier();
        wg_shared[local_id.x] = v;
        workgroupBarrier();
    }

    var running = 0u;
    if (local_id.x > 0u) {
        running = wg_shared[local_id.x - 1u];
    }
    for (var i = begin; i < end; i = i + 1u) {
        block_offsets[i] = running;
        running = running + block_counts[i];
    }

    if (local_id.x == 255u) {
        scan_meta[0] = 1u + wg_shared[255];
    }
}

// Pass 3: Scatter newline byte offsets into ordered array
@compute @workgroup_size(256)
fn scatter_line_offsets(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let block_id = group_id.y * 65535u + group_id.x;
    if (block_id >= scan_cfg.num_blocks) {
        return;
    }

    if (scan_cfg.text_len == 0u) {
        return;
    }

    let scan_limit = scan_cfg.text_len - 1u;
    let total_words = (scan_cfg.text_len + 3u) >> 2u;

    if (block_id == 0u && local_id.x == 0u) {
        line_offsets[0] = 0u;
    }

    var count = 0u;
    let thread_base_word = block_id * 4096u + local_id.x * 16u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let word_idx = thread_base_word + w;
        if (word_idx < total_words) {
            let val = scan_input[word_idx];
            let m = val ^ 0x0A0A0A0Au;
            if (((m - 0x01010101u) & (~m) & 0x80808080u) != 0u) {
                let p0 = word_idx * 4u;
                if (p0 < scan_limit && (val & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p1 = p0 + 1u;
                if (p1 < scan_limit && ((val >> 8u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p2 = p0 + 2u;
                if (p2 < scan_limit && ((val >> 16u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
                let p3 = p0 + 3u;
                if (p3 < scan_limit && ((val >> 24u) & 0xFFu) == 10u) {
                    count = count + 1u;
                }
            }
        }
    }

    wg_shared[local_id.x] = count;
    workgroupBarrier();

    for (var offset = 1u; offset < 256u; offset = offset << 1u) {
        var val = wg_shared[local_id.x];
        if (local_id.x >= offset) {
            val = val + wg_shared[local_id.x - offset];
        }
        workgroupBarrier();
        wg_shared[local_id.x] = val;
        workgroupBarrier();
    }

    var thread_prefix = 0u;
    if (local_id.x > 0u) {
        thread_prefix = wg_shared[local_id.x - 1u];
    }

    let base_slot = 1u + block_offsets[block_id];
    var write_slot = base_slot + thread_prefix;

    if (count > 0u) {
        for (var w = 0u; w < 16u; w = w + 1u) {
            let word_idx = thread_base_word + w;
            if (word_idx < total_words) {
                let val = scan_input[word_idx];
                let m = val ^ 0x0A0A0A0Au;
                if (((m - 0x01010101u) & (~m) & 0x80808080u) != 0u) {
                    let p0 = word_idx * 4u;
                    if (p0 < scan_limit && (val & 0xFFu) == 10u) {
                        line_offsets[write_slot] = p0 + 1u;
                        write_slot = write_slot + 1u;
                    }
                    let p1 = p0 + 1u;
                    if (p1 < scan_limit && ((val >> 8u) & 0xFFu) == 10u) {
                        line_offsets[write_slot] = p1 + 1u;
                        write_slot = write_slot + 1u;
                    }
                    let p2 = p0 + 2u;
                    if (p2 < scan_limit && ((val >> 16u) & 0xFFu) == 10u) {
                        line_offsets[write_slot] = p2 + 1u;
                        write_slot = write_slot + 1u;
                    }
                    let p3 = p0 + 3u;
                    if (p3 < scan_limit && ((val >> 24u) & 0xFFu) == 10u) {
                        line_offsets[write_slot] = p3 + 1u;
                        write_slot = write_slot + 1u;
                    }
                }
            }
        }
    }
}
