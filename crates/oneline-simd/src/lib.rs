//! CPU parser: AVX2 newline scan + per-line classification, optionally multi-core.
//!
//! Both entry points use the same two-pass shape that the GPU pipeline uses:
//!
//! 1. **count** lines per chunk (a pure newline popcount),
//! 2. prefix-sum the counts to give every chunk its own output slice, then
//!    **fill** all chunks in parallel, each writing straight into the final
//!    `nodes` / `line_offsets` buffers.
//!
//! No per-chunk `Vec`s are allocated and nothing is concatenated afterwards, and
//! the source text is borrowed rather than copied.

mod line;
mod links;
mod scan;
mod search;

use std::mem::MaybeUninit;

use oneline_core::{FlatNode, FlatTree, InputTooLarge, check_input_len};

pub use line::parse_line;
pub use links::build_links;
pub use scan::{FLATTEN_SLACK, count_newlines, flatten_newlines, for_each_newline};
pub use search::{count_all_par, find_all, find_all_par, try_find_all_par};

/// Inputs smaller than this are not worth splitting across threads.
const PAR_THRESHOLD: usize = 256 * 1024;
/// Smallest chunk handed to a worker.
const MIN_CHUNK: usize = 128 * 1024;
/// Chunks per thread, for load balancing.
const CHUNKS_PER_THREAD: usize = 4;

/// Number of lines a chunk contributes. Chunks start at a line start, so this is
/// the number of `\n` plus one for an unterminated tail.
fn lines_in_chunk(chunk: &[u8]) -> usize {
    match chunk.last() {
        None => 0,
        Some(&last) => count_newlines(chunk) + usize::from(last != b'\n'),
    }
}

/// Scans line breaks quickly and returns starting byte offsets for each line.
///
/// # Panics
/// If `bytes` is larger than 4 GiB (offsets are `u32`).
pub fn scan_line_offsets(bytes: &[u8]) -> Vec<u32> {
    check_input_len(bytes.len()).expect("scan_line_offsets");
    let count = lines_in_chunk(bytes);
    let mut offsets = Vec::with_capacity(count);
    if count == 0 {
        return offsets;
    }
    offsets.push(0);
    let len = bytes.len();
    for_each_newline(bytes, |p| {
        // A line starts after every newline unless the text ends there.
        if p + 1 < len {
            offsets.push((p + 1) as u32);
        }
    });
    debug_assert_eq!(offsets.len(), count);
    offsets
}

/// Bytes handled per tile: newline positions are listed for a tile, then its lines are
/// classified while the tile is still in cache.
const TILE: usize = 16 * 1024;

/// Writes one `(offset, node)` per line of `bytes[start..end]` into the output slices.
fn fill_chunk(
    bytes: &[u8],
    start: usize,
    end: usize,
    nodes: &mut [MaybeUninit<FlatNode>],
    offsets: &mut [MaybeUninit<u32>],
) {
    let chunk = &bytes[start..end];
    let mut k = 0usize;
    let mut emit = |line_start: usize, line_end: usize| {
        let abs = (start + line_start) as u32;
        offsets[k].write(abs);
        nodes[k].write(parse_line(&chunk[line_start..line_end], abs));
        k += 1;
    };

    // Two stages per tile, as in simdjson: list the newline positions without branching
    // (the newline density is unpredictable, so a loop over mask bits mispredicts about
    // once per line), then walk the list, which is a predictable loop.
    let mut positions = vec![0u32; TILE + FLATTEN_SLACK];
    let mut line_start = 0usize;
    for (t, tile) in chunk.chunks(TILE).enumerate() {
        let n = flatten_newlines(tile, (t * TILE) as u32, &mut positions);
        for &p in &positions[..n] {
            let line_end = p as usize + 1;
            emit(line_start, line_end);
            line_start = line_end;
        }
    }
    if line_start < chunk.len() {
        emit(line_start, chunk.len());
    }

    // Safety net for the `set_len` in the caller: every slot must be written.
    assert_eq!(k, nodes.len(), "chunk line count disagrees with the count pass");
}

/// Splits `bytes` into at most `target` chunks whose boundaries are line starts.
fn partition_chunks(bytes: &[u8], target: usize) -> Vec<(usize, usize)> {
    let len = bytes.len();
    if len == 0 {
        return Vec::new();
    }
    let target = target.min(len / MIN_CHUNK).max(1);
    let approx = len / target;

    let mut bounds = vec![0usize];
    for k in 1..target {
        let mut pos = k * approx;
        // Move to the start of the next line.
        match bytes[pos..].iter().position(|&b| b == b'\n') {
            Some(i) => pos += i + 1,
            None => break,
        }
        if pos < len && pos > *bounds.last().unwrap() {
            bounds.push(pos);
        }
    }
    bounds.push(len);
    bounds.windows(2).map(|w| (w[0], w[1])).collect()
}

fn build<'a>(input: &'a str, chunks: &[(usize, usize)], parallel: bool) -> FlatTree<'a> {
    let bytes = input.as_bytes();

    // Pass 1: count lines per chunk.
    let counts: Vec<usize> = if parallel {
        use rayon::prelude::*;
        chunks.par_iter().map(|&(s, e)| lines_in_chunk(&bytes[s..e])).collect()
    } else {
        chunks.iter().map(|&(s, e)| lines_in_chunk(&bytes[s..e])).collect()
    };
    let total: usize = counts.iter().sum();

    let mut nodes: Vec<FlatNode> = Vec::with_capacity(total);
    let mut offsets: Vec<u32> = Vec::with_capacity(total);

    // Hand every chunk its own disjoint output window.
    let mut n_rest = nodes.spare_capacity_mut();
    let mut o_rest = offsets.spare_capacity_mut();
    let mut jobs = Vec::with_capacity(chunks.len());
    for (&(s, e), &c) in chunks.iter().zip(&counts) {
        let (n_head, n_tail) = n_rest.split_at_mut(c);
        let (o_head, o_tail) = o_rest.split_at_mut(c);
        jobs.push((s, e, n_head, o_head));
        n_rest = n_tail;
        o_rest = o_tail;
    }

    // Pass 2: fill.
    if parallel {
        use rayon::prelude::*;
        jobs.into_par_iter().for_each(|(s, e, n, o)| fill_chunk(bytes, s, e, n, o));
    } else {
        for (s, e, n, o) in jobs {
            fill_chunk(bytes, s, e, n, o);
        }
    }

    // SAFETY: `fill_chunk` asserted that it wrote exactly `counts[i]` slots of each
    // window, and the windows tile `0..total` without overlap.
    unsafe {
        nodes.set_len(total);
        offsets.set_len(total);
    }
    FlatTree::new(input, nodes, offsets)
}


/// How `parse_par` obtains output space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParStrategy {
    /// Pick by estimated output size (see [`prefers_gather`]).
    Auto,
    /// Count lines per chunk first, then fill the final arrays directly. Reads the text twice
    /// but copies nothing: best when the output is large relative to the text.
    TwoPass,
    /// Parse each chunk once into its own buffers, then copy them into the final arrays in
    /// parallel. Reads the text once: best when the text is much larger than the output
    /// and DRAM-bound.
    Gather,
}

/// Inputs below this are served from cache often enough that the extra read is cheap.
const GATHER_MIN_LEN: usize = 32 << 20;
/// Gather only if the (node + offset) output is at most this fraction of the text.
const GATHER_MAX_OUTPUT_RATIO: f64 = 0.5;

/// Estimated bytes of output (20 per line) per byte of text, from a few small windows.
fn output_ratio(bytes: &[u8]) -> f64 {
    const WINDOW: usize = 64 << 10;
    const WINDOWS: usize = 4;
    if bytes.len() <= WINDOW * WINDOWS {
        return f64::INFINITY; // small input: never gather
    }
    let span = bytes.len() - WINDOW;
    let lines: usize = (0..WINDOWS)
        .map(|i| count_newlines(&bytes[span / (WINDOWS - 1) * i..][..WINDOW]) + 1)
        .sum();
    lines as f64 * 20.0 / (WINDOW * WINDOWS) as f64
}

fn prefers_gather(bytes: &[u8]) -> bool {
    bytes.len() >= GATHER_MIN_LEN && output_ratio(bytes) <= GATHER_MAX_OUTPUT_RATIO
}

/// Parses `bytes[start..end]` (a chunk starting at a line start) into fresh vectors.
fn parse_chunk_local(bytes: &[u8], start: usize, end: usize, lines_hint: usize) -> (Vec<FlatNode>, Vec<u32>) {
    let chunk = &bytes[start..end];
    let mut nodes = Vec::with_capacity(lines_hint);
    let mut offsets = Vec::with_capacity(lines_hint);
    let mut positions = vec![0u32; TILE + FLATTEN_SLACK];
    let mut line_start = 0usize;
    for (t, tile) in chunk.chunks(TILE).enumerate() {
        let n = flatten_newlines(tile, (t * TILE) as u32, &mut positions);
        nodes.reserve(n);
        offsets.reserve(n);
        for &p in &positions[..n] {
            let line_end = p as usize + 1;
            let abs = (start + line_start) as u32;
            offsets.push(abs);
            nodes.push(parse_line(&chunk[line_start..line_end], abs));
            line_start = line_end;
        }
    }
    if line_start < chunk.len() {
        let abs = (start + line_start) as u32;
        offsets.push(abs);
        nodes.push(parse_line(&chunk[line_start..], abs));
    }
    (nodes, offsets)
}

/// One pass over the text: every chunk parses into its own buffers, then the buffers are
/// copied into the final arrays in parallel.
fn build_gather<'a>(input: &'a str, chunks: &[(usize, usize)]) -> FlatTree<'a> {
    use rayon::prelude::*;
    let bytes = input.as_bytes();
    let hint_per_byte = (output_ratio(bytes) / 20.0).min(1.0);

    let parts: Vec<(Vec<FlatNode>, Vec<u32>)> = chunks
        .par_iter()
        .map(|&(s, e)| parse_chunk_local(bytes, s, e, ((e - s) as f64 * hint_per_byte * 1.1) as usize + 16))
        .collect();
    let total: usize = parts.iter().map(|(n, _)| n.len()).sum();

    let mut nodes: Vec<FlatNode> = Vec::with_capacity(total);
    let mut offsets: Vec<u32> = Vec::with_capacity(total);
    let mut n_rest = nodes.spare_capacity_mut();
    let mut o_rest = offsets.spare_capacity_mut();
    let mut windows = Vec::with_capacity(parts.len());
    for (n, _) in &parts {
        let (n_head, n_tail) = n_rest.split_at_mut(n.len());
        let (o_head, o_tail) = o_rest.split_at_mut(n.len());
        windows.push((n_head, o_head));
        n_rest = n_tail;
        o_rest = o_tail;
    }
    parts.par_iter().zip(windows.into_par_iter()).for_each(|((n, o), (n_win, o_win))| {
        // SAFETY: the windows are disjoint, `len` slots each, and `MaybeUninit<T>` has the
        // layout of `T`; the sources are fully initialized vectors of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(n.as_ptr(), n_win.as_mut_ptr() as *mut FlatNode, n.len());
            std::ptr::copy_nonoverlapping(o.as_ptr(), o_win.as_mut_ptr() as *mut u32, o.len());
        }
    });
    // SAFETY: the windows tile `0..total` and were all written above.
    unsafe {
        nodes.set_len(total);
        offsets.set_len(total);
    }
    FlatTree::new(input, nodes, offsets)
}

/// Single-threaded parser.
pub fn try_parse(input: &str) -> Result<FlatTree<'_>, InputTooLarge> {
    check_input_len(input.len())?;
    if input.is_empty() {
        return Ok(FlatTree::new(input, Vec::new(), Vec::new()));
    }
    Ok(build(input, &[(0, input.len())], false))
}

/// Multi-threaded parser (Rayon + AVX2). Falls back to [`try_parse`] for small inputs.
pub fn try_parse_par(input: &str) -> Result<FlatTree<'_>, InputTooLarge> {
    try_parse_par_with(input, ParStrategy::Auto)
}

/// [`try_parse_par`] with an explicit output strategy (for benchmarks and tests).
pub fn try_parse_par_with(input: &str, strategy: ParStrategy) -> Result<FlatTree<'_>, InputTooLarge> {
    check_input_len(input.len())?;
    let threads = rayon::current_num_threads();
    if threads <= 1 || input.len() < PAR_THRESHOLD {
        return try_parse(input);
    }
    let chunks = partition_chunks(input.as_bytes(), threads * CHUNKS_PER_THREAD);
    if chunks.len() <= 1 {
        return Ok(build(input, &chunks, false));
    }
    let gather = match strategy {
        ParStrategy::Auto => prefers_gather(input.as_bytes()),
        ParStrategy::TwoPass => false,
        ParStrategy::Gather => true,
    };
    Ok(if gather { build_gather(input, &chunks) } else { build(input, &chunks, true) })
}

/// Main entry point for the single-threaded CPU parser.
///
/// # Panics
/// If `input` is larger than 4 GiB; use [`try_parse`] to handle that.
pub fn parse(input: &str) -> FlatTree<'_> {
    try_parse(input).expect("oneline_simd::parse")
}

/// Main entry point for the multi-threaded CPU parser.
///
/// # Panics
/// If `input` is larger than 4 GiB; use [`try_parse_par`] to handle that.
pub fn parse_par(input: &str) -> FlatTree<'_> {
    try_parse_par(input).expect("oneline_simd::parse_par")
}

#[cfg(test)]
mod tests {
    use super::*;
    use oneline_core::NodeKind;
    use oneline_core::datagen::{self, Profile};

    /// Obviously-correct reference: split on `\n`, parse each piece.
    fn reference(input: &str) -> (Vec<u32>, Vec<FlatNode>) {
        let bytes = input.as_bytes();
        let mut starts = Vec::new();
        if !bytes.is_empty() {
            starts.push(0usize);
            for (i, &b) in bytes.iter().enumerate() {
                if b == b'\n' && i + 1 < bytes.len() {
                    starts.push(i + 1);
                }
            }
        }
        let nodes = starts
            .iter()
            .enumerate()
            .map(|(k, &s)| {
                let e = starts.get(k + 1).copied().unwrap_or(bytes.len());
                parse_line(&bytes[s..e], s as u32)
            })
            .collect();
        (starts.into_iter().map(|s| s as u32).collect(), nodes)
    }

    fn assert_matches_reference(input: &str) {
        let (offs, nodes) = reference(input);
        assert_eq!(scan_line_offsets(input.as_bytes()), offs, "scan_line_offsets");
        for (name, tree) in [("parse", parse(input)), ("parse_par", parse_par(input))] {
            assert_eq!(tree.line_offsets, offs, "{name}: offsets");
            assert_eq!(tree.nodes, nodes, "{name}: nodes");
            assert_eq!(tree.source, input);
        }
    }

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
    fn edge_cases() {
        for case in [
            "",
            "\n",
            "\n\n\n",
            "x",
            "x\n",
            "x\ny",
            "\r\n",
            "a\r\nb\r\n",
            "|\n",
            "||||\n|||| \n",
            "   \n\t\n",
            "| 日本語\n|| ｗｉｄｅ\n",
            "```\n- not a list\n```\n",
            "no trailing newline",
        ] {
            assert_matches_reference(case);
        }
        // Lines longer than any chunk/SIMD width, and one giant unterminated line.
        assert_matches_reference(&format!("{}\n|{}", "a".repeat(100_000), "b".repeat(300_000)));
        assert_matches_reference(&"|".repeat(1_000_000));
    }

    #[test]
    fn all_profiles_match_reference_across_chunking() {
        // Big enough that parse_par really splits (> PAR_THRESHOLD, several chunks).
        for p in Profile::ALL {
            for seed in 0..3 {
                let text = datagen::generate(p, 2 * 1024 * 1024, seed);
                assert_matches_reference(&text);
            }
        }
    }

    #[test]
    fn every_parallel_strategy_matches_reference() {
        // Auto picks TwoPass on inputs this small, so force each strategy.
        for strategy in [ParStrategy::TwoPass, ParStrategy::Gather, ParStrategy::Auto] {
            for p in Profile::ALL {
                let text = datagen::generate(p, 3 * 1024 * 1024, 41);
                let (offs, nodes) = reference(&text);
                let tree = try_parse_par_with(&text, strategy).unwrap();
                assert_eq!(tree.line_offsets, offs, "{strategy:?} {p:?}: offsets");
                assert_eq!(tree.nodes, nodes, "{strategy:?} {p:?}: nodes");
            }
            // Awkward shapes: no trailing newline, one giant line, CRLF only, all newlines.
            for case in [
                format!("{}\n|{}", "a".repeat(500_000), "b".repeat(300_000)),
                "x\r\n".repeat(300_000),
                "\n".repeat(400_000),
                "|".repeat(1_000_000),
                format!("{}tail without newline", "line\n".repeat(200_000)),
            ] {
                let (offs, nodes) = reference(&case);
                let tree = try_parse_par_with(&case, strategy).unwrap();
                assert_eq!(tree.line_offsets, offs, "{strategy:?} awkward: offsets");
                assert_eq!(tree.nodes, nodes, "{strategy:?} awkward: nodes");
            }
        }
    }

    #[test]
    fn auto_prefers_gather_only_for_large_inputs_with_long_lines() {
        let long = datagen::generate(Profile::LongLines, 40 << 20, 1);
        let short = datagen::generate(Profile::ShortLines, 40 << 20, 1);
        assert!(prefers_gather(long.as_bytes()), "long lines: output is a tiny fraction of the text");
        assert!(!prefers_gather(short.as_bytes()), "short lines: output is as large as the text");
        assert!(!prefers_gather(datagen::generate(Profile::LongLines, 1 << 20, 1).as_bytes()), "too small");
    }

    #[test]
    fn chunk_boundaries_are_line_starts() {
        let text = datagen::generate(Profile::Mixed, 3 * 1024 * 1024, 99);
        let bytes = text.as_bytes();
        let chunks = partition_chunks(bytes, 64);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks.last().unwrap().1, bytes.len());
        for w in chunks.windows(2) {
            assert_eq!(w[0].1, w[1].0, "chunks must tile the input");
            assert_eq!(bytes[w[0].1 - 1], b'\n', "boundary must follow a newline");
        }
    }

    #[test]
    fn rejects_inputs_over_4gib_without_allocating() {
        use oneline_core::{InputTooLarge, MAX_INPUT_LEN, check_input_len};
        assert_eq!(check_input_len(MAX_INPUT_LEN), Ok(()));
        assert_eq!(
            check_input_len(MAX_INPUT_LEN + 1),
            Err(InputTooLarge { len: MAX_INPUT_LEN + 1 })
        );
    }
}
