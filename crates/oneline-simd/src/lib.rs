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
pub use search::{ManyPatterns, PatternHit, count_all_par, find_all, find_all_par, try_find_all_par};

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

/// Output storage that survives between parses.
///
/// A parse of a large input allocates hundreds of megabytes (the result vectors, and for the
/// gather strategy a second set of per-chunk vectors). Fresh memory is not free: the operating
/// system hands it out page by page on first touch, which at 1 GiB of input was ~15 ms of a
/// ~55 ms parse. Keeping the buffers lets the next parse write into memory that is already mapped.
///
/// ```
/// # use oneline_simd::{ParseBuffers, try_parse_par_reusing};
/// let mut buffers = ParseBuffers::new();
/// for text in ["| a\n|| b\n", "| c\n"] {
///     let tree = try_parse_par_reusing(text, &mut buffers).unwrap();
///     assert!(!tree.nodes.is_empty());
///     buffers.recycle(tree); // hand the vectors back for the next parse
/// }
/// ```
#[derive(Default)]
pub struct ParseBuffers {
    /// Per-chunk vectors of the gather strategy.
    parts: Vec<(Vec<FlatNode>, Vec<u32>)>,
    /// Recycled result vectors (empty until a tree is handed back).
    nodes: Vec<FlatNode>,
    offsets: Vec<u32>,
}

impl ParseBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes back the vectors of a tree that is no longer needed so the next parse can fill them
    /// instead of allocating. Their capacity is kept as it is: if the next input is much smaller the
    /// memory stays reserved, so drop the `ParseBuffers` when you are done with them.
    pub fn recycle(&mut self, tree: FlatTree<'_>) {
        self.nodes = tree.nodes;
        self.offsets = tree.line_offsets;
    }

    /// Result vectors with room for `total` entries, empty.
    fn take_output(&mut self, total: usize) -> (Vec<FlatNode>, Vec<u32>) {
        let mut nodes = std::mem::take(&mut self.nodes);
        nodes.clear();
        nodes.reserve(total);
        let mut offsets = std::mem::take(&mut self.offsets);
        offsets.clear();
        offsets.reserve(total);
        (nodes, offsets)
    }
}

fn build<'a>(input: &'a str, chunks: &[(usize, usize)], parallel: bool, buffers: &mut ParseBuffers) -> FlatTree<'a> {
    let bytes = input.as_bytes();

    // Pass 1: count lines per chunk.
    let counts: Vec<usize> = if parallel {
        use rayon::prelude::*;
        chunks.par_iter().map(|&(s, e)| lines_in_chunk(&bytes[s..e])).collect()
    } else {
        chunks.iter().map(|&(s, e)| lines_in_chunk(&bytes[s..e])).collect()
    };
    let total: usize = counts.iter().sum();

    let (mut nodes, mut offsets) = buffers.take_output(total);

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

/// Single-threaded one-pass parse: no gather copy to pay, so it wins whenever the second read of
/// the text is not free, i.e. beyond the cache.
fn prefers_one_pass_single(bytes: &[u8]) -> bool {
    bytes.len() >= GATHER_MIN_LEN
}

fn prefers_gather(bytes: &[u8]) -> bool {
    bytes.len() >= GATHER_MIN_LEN && output_ratio(bytes) <= GATHER_MAX_OUTPUT_RATIO
}

/// Parses `bytes[start..end]` (a chunk starting at a line start) into `nodes` / `offsets`, which are
/// emptied first and keep their capacity.
fn parse_chunk_into(
    bytes: &[u8],
    start: usize,
    end: usize,
    lines_hint: usize,
    nodes: &mut Vec<FlatNode>,
    offsets: &mut Vec<u32>,
) {
    nodes.clear();
    offsets.clear();
    nodes.reserve(lines_hint);
    offsets.reserve(lines_hint);
    let chunk = &bytes[start..end];
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
}

/// One pass over the text: every chunk parses into its own buffers, then the buffers are
/// copied into the final arrays in parallel.
fn build_gather<'a>(input: &'a str, chunks: &[(usize, usize)], buffers: &mut ParseBuffers) -> FlatTree<'a> {
    use rayon::prelude::*;
    let bytes = input.as_bytes();
    let hint_per_byte = (output_ratio(bytes) / 20.0).min(1.0);

    if buffers.parts.len() < chunks.len() {
        buffers.parts.resize_with(chunks.len(), Default::default);
    }
    buffers.parts[..chunks.len()]
        .par_iter_mut()
        .zip(chunks.par_iter())
        .for_each(|((nodes, offsets), &(s, e))| {
            parse_chunk_into(bytes, s, e, ((e - s) as f64 * hint_per_byte * 1.1) as usize + 16, nodes, offsets);
        });
    let total: usize = buffers.parts[..chunks.len()].iter().map(|(n, _)| n.len()).sum();

    let (mut nodes, mut offsets) = buffers.take_output(total);
    let parts = &buffers.parts[..chunks.len()];
    let mut n_rest = nodes.spare_capacity_mut();
    let mut o_rest = offsets.spare_capacity_mut();
    let mut windows = Vec::with_capacity(parts.len());
    for (n, _) in parts {
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
///
/// Large inputs are parsed in one pass straight into the result vectors (sized from a sample of the
/// text, growing if the estimate was low). Counting the lines first would read the text a second
/// time, which at gigabyte sizes is a DRAM-bound pass of its own; below `GATHER_MIN_LEN` the text
/// is cache-hot and counting is nearly free, and gives exact allocations.
pub fn try_parse(input: &str) -> Result<FlatTree<'_>, InputTooLarge> {
    try_parse_reusing(input, &mut ParseBuffers::new())
}

/// [`try_parse`] filling the vectors held by `buffers` (see [`ParseBuffers`]).
pub fn try_parse_reusing<'a>(input: &'a str, buffers: &mut ParseBuffers) -> Result<FlatTree<'a>, InputTooLarge> {
    check_input_len(input.len())?;
    if input.is_empty() {
        return Ok(FlatTree::new(input, Vec::new(), Vec::new()));
    }
    let bytes = input.as_bytes();
    if prefers_one_pass_single(bytes) {
        let hint_per_byte = (output_ratio(bytes) / 20.0).min(1.0);
        // Generous: capacity that is reserved but never touched costs address space, not memory.
        let hint = (bytes.len() as f64 * hint_per_byte * 1.25) as usize + 16;
        let (mut nodes, mut offsets) = buffers.take_output(hint);
        parse_chunk_into(bytes, 0, bytes.len(), hint, &mut nodes, &mut offsets);
        return Ok(FlatTree::new(input, nodes, offsets));
    }
    Ok(build(input, &[(0, input.len())], false, buffers))
}

/// Multi-threaded parser (Rayon + AVX2). Falls back to [`try_parse`] for small inputs.
pub fn try_parse_par(input: &str) -> Result<FlatTree<'_>, InputTooLarge> {
    try_parse_par_with(input, ParStrategy::Auto)
}

/// [`try_parse_par`] with an explicit output strategy (for benchmarks and tests).
pub fn try_parse_par_with(input: &str, strategy: ParStrategy) -> Result<FlatTree<'_>, InputTooLarge> {
    try_parse_par_with_buffers(input, strategy, &mut ParseBuffers::new())
}

/// [`try_parse_par`] filling the buffers held by `buffers` (see [`ParseBuffers`]).
pub fn try_parse_par_reusing<'a>(input: &'a str, buffers: &mut ParseBuffers) -> Result<FlatTree<'a>, InputTooLarge> {
    try_parse_par_with_buffers(input, ParStrategy::Auto, buffers)
}

/// The general entry point: explicit strategy and reusable buffers.
pub fn try_parse_par_with_buffers<'a>(
    input: &'a str,
    strategy: ParStrategy,
    buffers: &mut ParseBuffers,
) -> Result<FlatTree<'a>, InputTooLarge> {
    check_input_len(input.len())?;
    let threads = rayon::current_num_threads();
    if threads <= 1 || input.len() < PAR_THRESHOLD {
        return try_parse_reusing(input, buffers);
    }
    let chunks = partition_chunks(input.as_bytes(), threads * CHUNKS_PER_THREAD);
    if chunks.len() <= 1 {
        return Ok(build(input, &chunks, false, buffers));
    }
    let gather = match strategy {
        ParStrategy::Auto => prefers_gather(input.as_bytes()),
        ParStrategy::TwoPass => false,
        ParStrategy::Gather => true,
    };
    Ok(if gather { build_gather(input, &chunks, buffers) } else { build(input, &chunks, true, buffers) })
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

        let l2 = parse_line(b"||:rule(allow:list(card))\n", 0);
        assert_eq!(l2.depth, 2);
        assert_eq!(l2.kind(), NodeKind::Connect);

        let l3 = parse_line(b"|||- List item\n", 0);
        assert_eq!(l3.depth, 3);
        assert_eq!(l3.kind(), NodeKind::List);

        let l5 = parse_line(b"|-. Ordered\n", 0);
        assert_eq!(l5.depth, 1);
        assert_eq!(l5.kind(), NodeKind::OrderedList);
        assert_eq!(parse_line(b"-.x\n", 0).kind(), NodeKind::Text);
        assert_eq!(parse_line(b"|| // c\n", 0).kind(), NodeKind::Comment);
        assert_eq!(parse_line(b"/x\n", 0).kind(), NodeKind::Text);

        let l4 = parse_line(b"|=[ Header ]\n", 0);
        assert_eq!(l4.depth, 1);
        assert_eq!(l4.kind(), NodeKind::Heading);
    }

    #[test]
    fn test_parse_document() {
        let doc = "|=[ Header ]\n| Is this syntax necessary?\n||:rule(x)\n|| Body text.\n";
        let tree = parse(doc);
        assert_eq!(tree.nodes.len(), 4);
        assert_eq!(tree.nodes[0].depth, 1);
        assert_eq!(tree.nodes[0].kind(), NodeKind::Heading);
        assert_eq!(tree.nodes[1].depth, 1);
        assert_eq!(tree.nodes[1].kind(), NodeKind::Text);
        assert_eq!(tree.nodes[2].depth, 2);
        assert_eq!(tree.nodes[2].kind(), NodeKind::Connect);
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
    fn reused_buffers_give_the_same_trees_in_any_order() {
        // One ParseBuffers across inputs that grow, shrink, vanish and change shape: stale
        // contents or a stale length from an earlier parse must never show through.
        let mut inputs: Vec<String> = Vec::new();
        for (p, size) in [(Profile::Mixed, 3 << 20), (Profile::ShortLines, 600 << 10), (Profile::LongLines, 5 << 20), (Profile::Nested, 300 << 10)] {
            inputs.push(datagen::generate(p, size, 17));
        }
        inputs.push(String::new());
        inputs.push("| tiny\n".to_string());
        inputs.push("\n".repeat(400_000));
        inputs.push(format!("{}tail", "line\n".repeat(100_000)));
        let order = [0, 1, 2, 3, 4, 5, 6, 2, 7, 0, 5, 1];

        for strategy in [ParStrategy::TwoPass, ParStrategy::Gather, ParStrategy::Auto] {
            let mut buffers = ParseBuffers::new();
            for &i in &order {
                let text = &inputs[i];
                let (offs, nodes) = reference(text);
                let tree = try_parse_par_with_buffers(text, strategy, &mut buffers).unwrap();
                assert_eq!(tree.line_offsets, offs, "{strategy:?} input {i}: offsets");
                assert_eq!(tree.nodes, nodes, "{strategy:?} input {i}: nodes");
                buffers.recycle(tree);
            }
        }
        // The single-threaded entry point shares the buffers too.
        let mut buffers = ParseBuffers::new();
        for &i in &order {
            let text = &inputs[i];
            let (offs, nodes) = reference(text);
            let tree = try_parse_reusing(text, &mut buffers).unwrap();
            assert_eq!(tree.line_offsets, offs, "single input {i}: offsets");
            assert_eq!(tree.nodes, nodes, "single input {i}: nodes");
            buffers.recycle(tree);
        }
    }

    #[test]
    fn reused_buffers_keep_their_allocations() {
        // The point of reuse: after warm-up, a same-sized parse writes into the same memory.
        let text = datagen::generate(Profile::LongLines, 4 << 20, 5);
        let mut buffers = ParseBuffers::new();
        let tree = try_parse_par_with_buffers(&text, ParStrategy::Gather, &mut buffers).unwrap();
        let (nodes_ptr, offsets_ptr) = (tree.nodes.as_ptr(), tree.line_offsets.as_ptr());
        buffers.recycle(tree);
        let parts_ptrs: Vec<*const FlatNode> = buffers.parts.iter().map(|(n, _)| n.as_ptr()).collect();
        let again = try_parse_par_with_buffers(&text, ParStrategy::Gather, &mut buffers).unwrap();
        assert_eq!(again.nodes.as_ptr(), nodes_ptr, "result nodes were reallocated");
        assert_eq!(again.line_offsets.as_ptr(), offsets_ptr, "result offsets were reallocated");
        let after: Vec<*const FlatNode> = buffers.parts.iter().map(|(n, _)| n.as_ptr()).collect();
        assert_eq!(after, parts_ptrs, "per-chunk buffers were reallocated");
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
    fn single_core_one_pass_matches_reference_on_large_inputs() {
        // At least GATHER_MIN_LEN, so `parse` takes the one-pass path.
        for p in [Profile::LongLines, Profile::Mixed] {
            let text = datagen::generate(p, 34 << 20, 7);
            assert!(prefers_one_pass_single(text.as_bytes()));
            let (offs, nodes) = reference(&text);
            let tree = parse(&text);
            assert_eq!(tree.line_offsets, offs, "{p:?}: offsets");
            assert_eq!(tree.nodes, nodes, "{p:?}: nodes");
        }
        // A sample that badly under-estimates the line count (a block of empty lines sits between
        // the sampled windows), so the vectors must grow: the result still has to be exact.
        // The sample windows sit at 0, ~1/3, ~2/3 and the end of the text; the empty lines go between the
        // first two, so the sample sees none of them.
        let mut text = "x".repeat(2 << 20);
        text.push_str(&"\n".repeat(6 << 20));
        text.push_str(&"y".repeat(30 << 20));
        text.push_str("\ntail");
        let actual_lines = text.bytes().filter(|&b| b == b'\n').count() + 1;
        let estimated = (output_ratio(text.as_bytes()) / 20.0 * text.len() as f64) as usize;
        assert!(estimated * 10 < actual_lines, "the sample must under-estimate for this test to mean anything: {estimated} vs {actual_lines}");
        let (offs, nodes) = reference(&text);
        let tree = parse(&text);
        assert_eq!(tree.line_offsets, offs, "growth: offsets");
        assert_eq!(tree.nodes, nodes, "growth: nodes");
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
