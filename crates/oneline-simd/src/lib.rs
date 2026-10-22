use oneline_core::{FlatNode, FlatTree, NodeKind};

/// Scans line breaks quickly and returns starting byte offsets for each line.
pub fn scan_line_offsets(bytes: &[u8]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(bytes.len() / 40 + 1);
    if bytes.is_empty() {
        return offsets;
    }

    // Line 1 always starts at offset 0.
    offsets.push(0);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                scan_line_offsets_avx2(bytes, &mut offsets);
            }
            return offsets;
        }
    }

    scan_line_offsets_swar(bytes, &mut offsets);
    offsets
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_line_offsets_avx2(bytes: &[u8], offsets: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut i = 0;
    let newline_vec = _mm256_set1_epi8(b'\n' as i8);

    while i + 32 <= len {
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i) };
        let cmp = _mm256_cmpeq_epi8(chunk, newline_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0 {
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros();
                let next = (i as u32) + bit + 1;
                if (next as usize) < len {
                    offsets.push(next);
                }
                m &= m - 1; // Clear lowest set bit
            }
        }
        i += 32;
    }

    // Process remaining trailing bytes in scalar mode.
    while i < len {
        if bytes[i] == b'\n' {
            let next = (i + 1) as u32;
            if (next as usize) < len {
                offsets.push(next);
            }
        }
        i += 1;
    }
}

fn scan_line_offsets_swar(bytes: &[u8], offsets: &mut Vec<u32>) {
    let mut i = 0;
    let len = bytes.len();

    // 8-byte SWAR (SIMD Within A Register) scan for '\n'.
    while i + 8 <= len {
        let chunk = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
        let match_mask = chunk ^ 0x0a0a0a0a_0a0a0a0a;
        let zero_byte_mask = (match_mask.wrapping_sub(0x01010101_01010101)) & (!match_mask) & 0x80808080_80808080;

        if zero_byte_mask != 0 {
            for b in 0..8 {
                if bytes[i + b] == b'\n' {
                    let next = (i + b + 1) as u32;
                    if (next as usize) < len {
                        offsets.push(next);
                    }
                }
            }
        }
        i += 8;
    }

    // Process remaining trailing bytes in scalar mode.
    while i < len {
        if bytes[i] == b'\n' {
            let next = (i + 1) as u32;
            if (next as usize) < len {
                offsets.push(next);
            }
        }
        i += 1;
    }
}

/// Parses a single line byte slice into a FlatNode.
#[inline]
pub fn parse_line(line_bytes: &[u8], line_start: u32) -> FlatNode {
    let len = line_bytes.len();
    if len == 0 {
        return FlatNode::new(0, NodeKind::Blank, line_start, 0);
    }

    // Exclude trailing '\r' and '\n' from effective length.
    let mut eff_len = len;
    while eff_len > 0 && (line_bytes[eff_len - 1] == b'\r' || line_bytes[eff_len - 1] == b'\n') {
        eff_len -= 1;
    }

    if eff_len == 0 {
        return FlatNode::new(0, NodeKind::Blank, line_start, 0);
    }

    // Count consecutive prefix '|' characters (nesting depth).
    let mut depth = 0;
    while depth < eff_len && line_bytes[depth] == b'|' {
        depth += 1;
    }

    // Skip whitespace immediately following the prefix.
    let mut content_start = depth;
    while content_start < eff_len && (line_bytes[content_start] == b' ' || line_bytes[content_start] == b'\t') {
        content_start += 1;
    }

    let span_start = line_start + content_start as u32;
    let span_len = (eff_len - content_start) as u32;

    if span_len == 0 {
        return FlatNode::new(depth as u32, NodeKind::Blank, span_start, 0);
    }

    let rest = &line_bytes[content_start..eff_len];
    let kind = classify_line(rest);

    FlatNode::new(depth as u32, kind, span_start, span_len)
}

/// Classifies syntax kind from leading line tokens (tomet / oneline compatible).
#[inline]
fn classify_line(bytes: &[u8]) -> NodeKind {
    if bytes.is_empty() {
        return NodeKind::Blank;
    }

    match bytes[0] {
        b'=' => {
            // Heading: =[ Header ] or ==[ Subheader ]
            NodeKind::Heading
        }
        b'>' => {
            // Callout or Header: >[ Header ] or >> Content
            NodeKind::Callout
        }
        b'@' => {
            // Element: @card(...) or @memo[ ... ]
            NodeKind::Element
        }
        b'-' | b'*' => {
            // List: - Item or * Item
            if bytes.len() > 1 && (bytes[1] == b' ' || bytes[1] == b'\t') {
                NodeKind::List
            } else {
                NodeKind::Text
            }
        }
        b'`' => {
            // Fenced code block: ```
            if bytes.starts_with(b"```") {
                NodeKind::Raw
            } else {
                NodeKind::Text
            }
        }
        _ => {
            // Numbered list item check (e.g. "1. ")
            if bytes[0].is_ascii_digit() {
                let mut idx = 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                if idx < bytes.len() && bytes[idx] == b'.' && idx + 1 < bytes.len() && bytes[idx + 1] == b' ' {
                    return NodeKind::List;
                }
            }
            NodeKind::Text
        }
    }
}

/// Main entry point for single-threaded CPU parser.
pub fn parse(input: &str) -> FlatTree {
    let bytes = input.as_bytes();
    let line_offsets = scan_line_offsets(bytes);
    let mut nodes = Vec::with_capacity(line_offsets.len());

    for (idx, &start) in line_offsets.iter().enumerate() {
        let end = if idx + 1 < line_offsets.len() {
            line_offsets[idx + 1] as usize
        } else {
            bytes.len()
        };
        let line_bytes = &bytes[start as usize..end];
        let node = parse_line(line_bytes, start);
        nodes.push(node);
    }

    FlatTree::new(input.to_string(), nodes, line_offsets)
}

/// Multi-threaded CPU parser powered by Rayon and AVX2 SIMD.
pub fn parse_par(input: &str) -> FlatTree {
    let bytes = input.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return FlatTree::new(input.to_string(), Vec::new(), Vec::new());
    }

    let num_threads = rayon::current_num_threads();
    if num_threads <= 1 || len < 64 * 1024 {
        return parse(input);
    }

    use rayon::prelude::*;

    let chunks = partition_chunks(bytes, num_threads);
    let chunks_data: Vec<(Vec<FlatNode>, Vec<u32>)> = chunks
        .into_par_iter()
        .map(|(start, end)| scan_and_parse_chunk(bytes, start, end))
        .collect();

    let total_lines: usize = chunks_data.iter().map(|(n, _)| n.len()).sum();
    let mut all_nodes = Vec::with_capacity(total_lines);
    let mut all_offsets = Vec::with_capacity(total_lines);

    for (nodes, offsets) in chunks_data {
        all_nodes.extend(nodes);
        all_offsets.extend(offsets);
    }

    FlatTree::new(input.to_string(), all_nodes, all_offsets)
}

fn partition_chunks(bytes: &[u8], target_chunks: usize) -> Vec<(usize, usize)> {
    let len = bytes.len();
    if len == 0 {
        return Vec::new();
    }
    if target_chunks <= 1 || len < 32 * 1024 {
        return vec![(0, len)];
    }

    let approx_chunk_size = len / target_chunks;
    if approx_chunk_size < 16 * 1024 {
        return vec![(0, len)];
    }

    let mut boundaries = Vec::with_capacity(target_chunks + 1);
    boundaries.push(0);

    for k in 1..target_chunks {
        let naive = k * approx_chunk_size;
        let mut pos = naive;
        while pos < len && bytes[pos] != b'\n' {
            pos += 1;
        }
        if pos < len {
            pos += 1; // Start of next line
        }
        if pos < len && pos > *boundaries.last().unwrap() {
            boundaries.push(pos);
        }
    }

    boundaries.push(len);
    boundaries.dedup();

    let mut chunks = Vec::with_capacity(boundaries.len() - 1);
    for i in 0..boundaries.len() - 1 {
        chunks.push((boundaries[i], boundaries[i + 1]));
    }
    chunks
}

fn scan_and_parse_chunk(bytes: &[u8], chunk_start: usize, chunk_end: usize) -> (Vec<FlatNode>, Vec<u32>) {
    let chunk_bytes = &bytes[chunk_start..chunk_end];
    let chunk_len = chunk_bytes.len();
    if chunk_len == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut offsets = Vec::with_capacity(chunk_len / 40 + 1);
    offsets.push(chunk_start as u32);

    scan_chunk_newlines(chunk_bytes, chunk_start as u32, &mut offsets);

    let mut nodes = Vec::with_capacity(offsets.len());
    for (idx, &line_start) in offsets.iter().enumerate() {
        let line_end = if idx + 1 < offsets.len() {
            offsets[idx + 1] as usize
        } else {
            chunk_end
        };
        let line_bytes = &bytes[line_start as usize..line_end];
        let node = parse_line(line_bytes, line_start);
        nodes.push(node);
    }

    (nodes, offsets)
}

#[inline]
fn scan_chunk_newlines(chunk: &[u8], base: u32, offsets: &mut Vec<u32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                scan_chunk_newlines_avx2(chunk, base, offsets);
            }
            return;
        }
    }
    scan_chunk_newlines_swar(chunk, base, offsets);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_chunk_newlines_avx2(chunk: &[u8], base: u32, offsets: &mut Vec<u32>) {
    use std::arch::x86_64::*;

    let len = chunk.len();
    let mut i = 0;
    let newline_vec = _mm256_set1_epi8(b'\n' as i8);

    while i + 32 <= len {
        let c = unsafe { _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i) };
        let cmp = _mm256_cmpeq_epi8(c, newline_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0 {
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros();
                let next = (i as u32) + bit + 1;
                if (next as usize) < len {
                    offsets.push(base + next);
                }
                m &= m - 1;
            }
        }
        i += 32;
    }

    while i < len {
        if chunk[i] == b'\n' {
            let next = (i + 1) as u32;
            if (next as usize) < len {
                offsets.push(base + next);
            }
        }
        i += 1;
    }
}

fn scan_chunk_newlines_swar(chunk: &[u8], base: u32, offsets: &mut Vec<u32>) {
    let mut i = 0;
    let len = chunk.len();

    while i + 8 <= len {
        let chunk_u64 = u64::from_ne_bytes(chunk[i..i + 8].try_into().unwrap());
        let match_mask = chunk_u64 ^ 0x0a0a0a0a_0a0a0a0a;
        let zero_byte_mask = (match_mask.wrapping_sub(0x01010101_01010101)) & (!match_mask) & 0x80808080_80808080;

        if zero_byte_mask != 0 {
            for b in 0..8 {
                if chunk[i + b] == b'\n' {
                    let next = (i + b + 1) as u32;
                    if (next as usize) < len {
                        offsets.push(base + next);
                    }
                }
            }
        }
        i += 8;
    }

    while i < len {
        if chunk[i] == b'\n' {
            let next = (i + 1) as u32;
            if (next as usize) < len {
                offsets.push(base + next);
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_line_offsets() {
        let text = "line1\nline2\r\nline3\n";
        let offsets = scan_line_offsets(text.as_bytes());
        assert_eq!(offsets, vec![0, 6, 13]);
    }

    #[test]
    fn test_parse_line_kinds() {
        let l1 = parse_line(b"| @card(title:\"test\")\n", 0);
        assert_eq!(l1.depth, 1);
        assert_eq!(l1.kind(), NodeKind::Element);

        let l2 = parse_line(b"||>[ Callout ]\n", 0);
        assert_eq!(l2.depth, 2);
        assert_eq!(l2.kind(), NodeKind::Callout);

        let l3 = parse_line(b"|||- List item\n", 0);
        assert_eq!(l3.depth, 3);
        assert_eq!(l3.kind(), NodeKind::List);

        let l4 = parse_line(b"|=[ Header ]\n", 0);
        assert_eq!(l4.depth, 1);
        assert_eq!(l4.kind(), NodeKind::Heading);
    }

    #[test]
    fn test_parse_document() {
        let doc = "|>[ Header ]\n| Is this syntax necessary?\n||>[ Callout ]\n|| Body text.\n";
        let tree = parse(doc);
        assert_eq!(tree.nodes.len(), 4);
        assert_eq!(tree.nodes[0].depth, 1);
        assert_eq!(tree.nodes[0].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[1].depth, 1);
        assert_eq!(tree.nodes[1].kind(), NodeKind::Text);
        assert_eq!(tree.nodes[2].depth, 2);
        assert_eq!(tree.nodes[2].kind(), NodeKind::Callout);
        assert_eq!(tree.nodes[3].depth, 2);
        assert_eq!(tree.nodes[3].kind(), NodeKind::Text);
    }

    #[test]
    fn test_parse_par_matches_sequential() {
        let sample = "|>[ Header ]\n| Paragraph line\n||>[ Nested Callout ]\n|| Body content\n|||- List 1\n|||- List 2\n";
        let mut text = String::with_capacity(sample.len() * 2000);
        for _ in 0..2000 {
            text.push_str(sample);
        }

        let seq_tree = parse(&text);
        let par_tree = parse_par(&text);

        assert_eq!(par_tree.line_offsets.len(), seq_tree.line_offsets.len());
        assert_eq!(par_tree.line_offsets, seq_tree.line_offsets);
        assert_eq!(par_tree.nodes.len(), seq_tree.nodes.len());
        for (i, (p_node, s_node)) in par_tree.nodes.iter().zip(seq_tree.nodes.iter()).enumerate() {
            assert_eq!(p_node.depth, s_node.depth, "depth mismatch at line {}", i);
            assert_eq!(p_node.kind(), s_node.kind(), "kind mismatch at line {}", i);
            assert_eq!(p_node.span_start, s_node.span_start, "span_start mismatch at line {}", i);
            assert_eq!(p_node.span_len, s_node.span_len, "span_len mismatch at line {}", i);
        }
    }
}
