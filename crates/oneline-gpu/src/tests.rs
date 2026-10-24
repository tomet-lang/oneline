//! GPU integration tests. They need a Vulkan/Metal/DX12 adapter; on a headless
//! machine they skip. A software adapter (e.g. lavapipe) is enough for correctness.

use oneline_core::datagen::{self, Profile};
use oneline_core::{FlatNode, NodeKind};

use crate::{GpuParser, MAX_GPU_INPUT_LEN};

fn parser() -> Option<GpuParser> {
    match GpuParser::new() {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("Skipping GPU test (no adapter or headless): {e:?}");
            None
        }
    }
}

macro_rules! gpu {
    () => {
        match parser() {
            Some(p) => p,
            None => return,
        }
    };
}

#[test]
fn test_gpu_parser_basic() {
    let parser = gpu!();
    let text = "|=[ Header ]\n| Body text\n||:rule(x)\n|| Content.\n";
    let offsets = oneline_simd::scan_line_offsets(text.as_bytes());
    let tree = parser.parse(text, &offsets).unwrap();

    assert_eq!(tree.nodes.len(), 4);
    assert_eq!(
        (tree.nodes[0].depth, tree.nodes[0].kind()),
        (1, NodeKind::Heading)
    );
    assert_eq!(
        (tree.nodes[1].depth, tree.nodes[1].kind()),
        (1, NodeKind::Text)
    );
    assert_eq!(
        (tree.nodes[2].depth, tree.nodes[2].kind()),
        (2, NodeKind::Connect)
    );
    assert_eq!(
        (tree.nodes[3].depth, tree.nodes[3].kind()),
        (2, NodeKind::Text)
    );
}

#[test]
fn test_gpu_scan_line_offsets_various() {
    let parser = gpu!();
    for case in [
        "hello",
        "hello\n",
        "line1\nline2\nline3\n",
        "line1\r\nline2\r\nline3\r\n",
        "\n\n\n",
        "|=[ Header ]\n| Paragraph\n|| Content\n",
    ] {
        let cpu = oneline_simd::scan_line_offsets(case.as_bytes());
        assert_eq!(
            parser.scan_line_offsets(case).unwrap(),
            cpu,
            "case {case:?}"
        );
    }
}

#[test]
fn test_gpu_parse_raw() {
    let parser = gpu!();
    let text = "|=[ Header ]\n| Body text\n||:rule(x)\n|| Content.\n";
    let tree = parser.parse_raw(text).unwrap();
    assert_eq!(tree.nodes.len(), 4);
    assert_eq!(tree.nodes[0].kind(), NodeKind::Heading);
    assert_eq!(tree.nodes[3].depth, 2);
    assert_eq!(tree.line_offsets, vec![0, 13, 25, 36]);
}

#[test]
fn test_gpu_query_raw() {
    let parser = gpu!();
    let text = "| @card(title:\"A\")\n|| Content\n| @card(title:\"B\")\n|| More\n|=[ Header ]\n";
    assert_eq!(
        parser
            .query_raw(text, Some(NodeKind::Element), None, 100)
            .unwrap(),
        vec![0, 2]
    );
    assert_eq!(
        parser.query_raw(text, None, Some(2), 100).unwrap(),
        vec![1, 3]
    );

    let offsets = oneline_simd::scan_line_offsets(text.as_bytes());
    assert_eq!(
        parser
            .query(text, &offsets, Some(NodeKind::Element), None, 100)
            .unwrap(),
        vec![0, 2]
    );
}

/// GPU parse must equal the CPU parse bit for bit, on every profile and on the
/// awkward edge cases.
#[test]
fn gpu_parse_matches_cpu() {
    let parser = gpu!();
    for case in [
        "x",
        "x\n",
        "\n",
        "\r\n",
        "a\r\nb",
        "||||\n|||| \n",
        "```\n- x\n```\n",
        "| 日本語\n",
        // tomet grammar boundaries: list needs a blank after `-`; `>`, `*`, `1.` and fences are plain text.
        "-\tx\n",
        "- x\n",
        "-x\n",
        "-\n",
        "- \n",
        ":rule(a)\n|| :{y:2}\n",
        "=\n=[ t ]\n",
        "@\n",
        "> a\n* b\n1. c\n",
        "|-\tx\n|:y\n",
        "// c\n",
        "//\n",
        "/x\n",
        "/\n",
        "|| // c\n",
        "http://x\n",
        "/// doc\n",
        "| | x\n",
        "|  | x\n",
        "|\t|x\n",
        "| x | y\n",
        "| |\n",
        "|| |\n",
        " | x\n",
        "|| \t | \n",
        "-. x\n",
        "-.\tx\n",
        "-.x\n",
        "-.\n",
        "-. \n",
        "-..\n",
        "- -. x\n",
    ] {
        let cpu = oneline_simd::parse(case);
        let gpu = parser.parse_raw(case).unwrap();
        assert_eq!(gpu, cpu, "case {case:?}");
    }
    for p in Profile::ALL {
        let text = datagen::generate(p, 3 << 20, 11);
        assert_eq!(
            parser.parse_raw(&text).unwrap(),
            oneline_simd::parse(&text),
            "profile {p:?}"
        );
    }
}

/// More than 256 scan blocks (16 KiB each) so every pass-2 thread owns a strip of
/// several blocks, plus a block count that is not a multiple of 256.
#[test]
fn gpu_scan_with_many_blocks_matches_cpu() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 9 << 20, 3);
    let blocks = text.len().div_ceil(16384);
    assert!(blocks > 512 && blocks % 256 != 0, "blocks = {blocks}");
    assert_eq!(
        parser.scan_line_offsets(&text).unwrap(),
        oneline_simd::scan_line_offsets(text.as_bytes())
    );
}

fn cpu_matches(nodes: &[FlatNode], kind: Option<NodeKind>, depth: Option<u32>) -> Vec<u32> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| kind.is_none_or(|k| n.kind() == k) && depth.is_none_or(|d| n.depth == d))
        .map(|(i, _)| i as u32)
        .collect()
}

#[test]
fn resident_and_fused_queries_match_cpu() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 2 << 20, 8);
    let tree = oneline_simd::parse(&text);
    let doc = parser.upload(&text).unwrap();
    assert_eq!(doc.line_count(), tree.nodes.len());
    assert_eq!(doc.text_len(), text.len());

    let kinds = [
        None,
        Some(NodeKind::Text),
        Some(NodeKind::Heading),
        Some(NodeKind::Element),
        Some(NodeKind::List),
        Some(NodeKind::OrderedList),
        Some(NodeKind::Connect),
        Some(NodeKind::Comment),
    ];
    let depths = [None, Some(0), Some(1), Some(3), Some(50)];
    let all = tree.nodes.len();
    for &kind in &kinds {
        for &depth in &depths {
            let want = cpu_matches(&tree.nodes, kind, depth);
            let resident = parser.query_document(&doc, kind, depth, all).unwrap();
            assert_eq!(
                resident.indices, want,
                "resident kind={kind:?} depth={depth:?}"
            );
            assert_eq!(resident.total_matches as usize, want.len());
            let fused = parser.query_raw(&text, kind, depth, all).unwrap();
            assert_eq!(fused, want, "fused kind={kind:?} depth={depth:?}");
        }
    }
}

#[test]
fn query_truncation_reports_true_total() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 1 << 20, 4);
    let tree = oneline_simd::parse(&text);
    let want = cpu_matches(&tree.nodes, Some(NodeKind::Text), None);
    assert!(want.len() > 100);

    let doc = parser.upload(&text).unwrap();
    let r = parser
        .query_document(&doc, Some(NodeKind::Text), None, 10)
        .unwrap();
    assert_eq!(r.total_matches as usize, want.len());
    assert_eq!(r.indices.len(), 10);
    assert!(
        r.indices.iter().all(|i| want.binary_search(i).is_ok()),
        "subset of true matches"
    );

    let none = parser
        .query_document(&doc, Some(NodeKind::Text), None, 0)
        .unwrap();
    assert_eq!(none.total_matches as usize, want.len());
    assert!(none.indices.is_empty());
}

#[test]
fn empty_inputs() {
    let parser = gpu!();
    assert!(parser.scan_line_offsets("").unwrap().is_empty());
    assert!(parser.parse_raw("").unwrap().nodes.is_empty());
    assert!(parser.query_raw("", None, None, 8).unwrap().is_empty());
    let doc = parser.upload("").unwrap();
    assert_eq!(doc.line_count(), 0);
    assert_eq!(
        parser
            .query_document(&doc, None, None, 8)
            .unwrap()
            .total_matches,
        0
    );
}

#[test]
fn oversized_input_is_an_error_not_corruption() {
    use crate::util::check_gpu_len;
    assert!(check_gpu_len(MAX_GPU_INPUT_LEN).is_ok());
    assert!(check_gpu_len(MAX_GPU_INPUT_LEN + 1).is_err());
    assert!(check_gpu_len(usize::MAX).is_err());
}

// ---- substring search -------------------------------------------------------

use crate::MAX_PATTERN_LEN;

/// Search results must equal the CPU reference exactly, through both the resident
/// and the one-shot path.
fn assert_search_matches_cpu(parser: &GpuParser, text: &str, pattern: &[u8]) {
    let want = oneline_simd::find_all(text.as_bytes(), pattern);
    let cap = want.len().max(1);
    let doc = parser.upload(text).unwrap();
    let resident = parser.search_document(&doc, pattern, cap).unwrap();
    assert_eq!(
        resident.total_matches as usize,
        want.len(),
        "resident count for {pattern:?}"
    );
    assert_eq!(resident.offsets, want, "resident offsets for {pattern:?}");
    let one_shot = parser.search_raw(text, pattern, cap).unwrap();
    assert_eq!(one_shot.offsets, want, "one-shot offsets for {pattern:?}");
}

#[test]
fn search_matches_cpu_on_generated_documents() {
    let parser = gpu!();
    for profile in Profile::ALL {
        let text = datagen::generate(profile, 3 << 20, 17);
        let bytes = text.as_bytes();
        let mut patterns: Vec<Vec<u8>> = [
            "lorem",
            "SIMD GPU",
            "naïve",
            "|",
            "||",
            "\n|",
            "階層 解析",
            "- ",
            "zzzz-absent",
        ]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
        // Slices of the text itself, from 1 byte up to the 64-byte maximum.
        for (i, len) in [1usize, 2, 3, 5, 8, 17, 33, 63, 64].into_iter().enumerate() {
            let at = 1000 + i * 99_991;
            patterns.push(bytes[at..at + len].to_vec());
        }
        for p in patterns {
            assert_search_matches_cpu(&parser, &text, &p);
        }
    }
}

#[test]
fn search_handles_alignment_overlap_and_tail() {
    let parser = gpu!();
    // The needle at every alignment relative to the 4-byte words the kernel owns.
    for shift in 0..8 {
        let text = format!("{}needle{}needle", "x".repeat(shift), "y".repeat(shift));
        assert_search_matches_cpu(&parser, &text, b"needle");
    }
    assert_search_matches_cpu(&parser, "aaaaaaaa", b"aa"); // overlapping occurrences
    assert_search_matches_cpu(&parser, "abc", b"abc"); // whole text
    assert_search_matches_cpu(&parser, "xxabc", b"abc"); // flush with the end
    assert_search_matches_cpu(&parser, "ab", b"abc"); // text shorter than pattern
    assert_search_matches_cpu(&parser, "a\nb\na\nb", b"a\nb"); // pattern spans a newline
    // The buffer is zero-padded to a word boundary; a pattern ending in NUL must not
    // "match" the padding.
    assert_search_matches_cpu(&parser, "xa", b"a\0");
    assert_search_matches_cpu(&parser, "xa\0", b"a\0");
}

#[test]
fn search_rejects_bad_patterns() {
    let parser = gpu!();
    let doc = parser.upload("abc").unwrap();
    assert!(parser.search_document(&doc, b"", 8).is_err());
    assert!(
        parser
            .search_document(&doc, &[b'a'; MAX_PATTERN_LEN + 1], 8)
            .is_err()
    );
    assert!(
        parser
            .search_document(&doc, &[b'a'; MAX_PATTERN_LEN], 8)
            .is_ok()
    );
    assert!(parser.search_raw("abc", b"", 8).is_err());
    // An error must not poison later calls (the scratch is dropped and rebuilt).
    assert_eq!(
        parser.search_document(&doc, b"b", 8).unwrap().offsets,
        vec![1]
    );
}

#[test]
fn search_truncation_reports_true_total() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 2 << 20, 2);
    let want = oneline_simd::find_all(text.as_bytes(), b"|");
    assert!(
        want.len() > 10_000,
        "need a dense pattern to stress the workgroup reservation"
    );

    let doc = parser.upload(&text).unwrap();
    let r = parser.search_document(&doc, b"|", 100).unwrap();
    assert_eq!(r.total_matches as usize, want.len());
    assert_eq!(r.offsets.len(), 100);
    assert!(
        r.offsets.iter().all(|o| want.binary_search(o).is_ok()),
        "subset of true matches"
    );

    let none = parser.search_document(&doc, b"|", 0).unwrap();
    assert_eq!(none.total_matches as usize, want.len());
    assert!(none.offsets.is_empty());
}

#[test]
fn search_on_empty_document() {
    let parser = gpu!();
    let doc = parser.upload("").unwrap();
    assert_eq!(
        parser.search_document(&doc, b"a", 8).unwrap().total_matches,
        0
    );
    assert_eq!(parser.search_raw("", b"a", 8).unwrap().total_matches, 0);
}

#[test]
fn search_anchor_in_the_middle_and_at_the_text_edges() {
    let parser = gpu!();
    // 'q' is the rarest byte, so "abcq" anchors on pattern index 3. The occurrence
    // at offset 0 puts the anchor at text position 3, the one at the very end puts
    // it on the last byte.
    let text = format!("abcq{}abcq", "abc".repeat(40_000));
    assert_search_matches_cpu(&parser, &text, b"abcq");
    // Anchor at index 0 and in the middle, same text.
    assert_search_matches_cpu(&parser, &text, b"qabc");
    assert_search_matches_cpu(&parser, &text, b"cabcqabc");
    // A pattern whose rarest byte never occurs in the text: no candidates at all.
    assert_search_matches_cpu(&parser, &text, b"abcz");
    // Anchor candidates whose implied start would be before the text (index > position).
    assert_search_matches_cpu(&parser, "qxyz", b"abcdq");
}

/// A filter-window hit near the start of the text implies a pattern start *before*
/// the text. Unsigned arithmetic would wrap that to a huge offset; the kernel must
/// reject it instead of reading out of range and (for a pattern starting with NUL,
/// which out-of-range reads "match") reporting a bogus occurrence.
#[test]
fn search_rejects_window_hits_that_imply_a_start_before_the_text() {
    let parser = gpu!();
    // NUL is common in this text, so it is not rare: the filter window for the
    // pattern below is "bcdq" (pattern index 1), whose only hit is at text offset 0.
    let text = format!("bcdq{}xyz", "\0".repeat(5000));
    let pattern = [0u8, b'b', b'c', b'd', b'q'];
    assert!(oneline_simd::find_all(text.as_bytes(), &pattern).is_empty());
    assert_search_matches_cpu(&parser, &text, &pattern);
    let doc = parser.upload(&text).unwrap();
    assert_eq!(
        parser
            .search_document(&doc, &pattern, 16)
            .unwrap()
            .total_matches,
        0
    );
}

// ---- tree construction ------------------------------------------------------

use oneline_core::TreeLinks;

/// A document with one line per `(depth, blank)`: blank lines are empty or prefix-only.
fn text_from_depths(lines: &[(u32, bool)]) -> String {
    let mut text = String::new();
    for (i, &(depth, blank)) in lines.iter().enumerate() {
        text.push_str(&"|".repeat(depth as usize));
        if blank {
            // Alternate empty and prefix-only blanks: neither belongs to the tree.
            if i % 2 == 0 {
                text.truncate(text.len() - depth as usize);
            }
        } else {
            text.push_str(" x");
        }
        text.push('\n');
    }
    text
}

fn gpu_links(parser: &GpuParser, text: &str) -> TreeLinks {
    let doc = parser.upload(text).unwrap();
    let tree = parser.build_tree(&doc).unwrap();
    assert_eq!(tree.line_count(), doc.line_count());
    parser.download_links(&tree).unwrap()
}

fn assert_tree_matches_cpu(parser: &GpuParser, text: &str, what: &str) {
    let cpu = oneline_simd::build_links(&oneline_simd::parse(text).nodes);
    let gpu = gpu_links(parser, text);
    assert_eq!(gpu.parent.len(), cpu.parent.len(), "{what}: line count");
    if let Some(i) = (0..cpu.parent.len()).find(|&i| gpu.parent[i] != cpu.parent[i]) {
        panic!(
            "{what}: parent differs at line {i}: gpu {} cpu {}",
            gpu.parent[i], cpu.parent[i]
        );
    }
    if let Some(i) = (0..cpu.subtree_end.len()).find(|&i| gpu.subtree_end[i] != cpu.subtree_end[i])
    {
        panic!(
            "{what}: subtree_end differs at line {i}: gpu {} cpu {}",
            gpu.subtree_end[i], cpu.subtree_end[i]
        );
    }
}

#[test]
fn gpu_tree_matches_cpu_on_generated_documents() {
    let parser = gpu!();
    for p in Profile::ALL {
        let text = datagen::generate(p, 3 << 20, 21);
        assert_tree_matches_cpu(&parser, &text, p.name());
    }
}

/// A thread owns a run of 32 lines and a workgroup 2048: exercise sizes around both boundaries.
#[test]
fn gpu_tree_handles_run_boundaries() {
    let parser = gpu!();
    let mut rng = datagen::Rng::new(5);
    for n in [
        1usize, 2, 31, 32, 33, 255, 256, 257, 1000, 2047, 2048, 2049, 4095, 4096, 4097, 6145,
    ] {
        let lines: Vec<(u32, bool)> = (0..n)
            .map(|_| (rng.below(5) as u32, rng.chance(15)))
            .collect();
        assert_tree_matches_cpu(
            &parser,
            &text_from_depths(&lines),
            &format!("{n} random lines"),
        );
    }
}

/// Shapes where the nearest smaller line is far away, many runs back, or never exists.
#[test]
fn gpu_tree_handles_adversarial_shapes() {
    let parser = gpu!();
    let n = 256 * 40 + 77;
    let shapes: Vec<(&str, Vec<(u32, bool)>)> = vec![
        ("flat depth 1", vec![(1, false); n]),
        ("flat depth 0", vec![(0, false); n]),
        (
            "header then flat",
            std::iter::once((0, false))
                .chain(std::iter::repeat_n((1, false), n - 1))
                .collect(),
        ),
        (
            "flat then header at the end",
            std::iter::repeat_n((1, false), n - 1)
                .chain(std::iter::once((0, false)))
                .collect(),
        ),
        (
            "strictly increasing to the limit",
            (0..n as u32).map(|i| (i % 32, false)).collect(),
        ),
        (
            "strictly decreasing sawtooth",
            (0..n as u32).map(|i| (31 - i % 32, false)).collect(),
        ),
        (
            "alternating",
            (0..n as u32).map(|i| (i % 2, false)).collect(),
        ),
        ("only blanks", vec![(3, true); n]),
        (
            "blanks between levels",
            (0..n as u32).map(|i| (i % 4, i % 3 == 0)).collect(),
        ),
        ("deepest allowed", vec![(31, false); n]),
    ];
    for (name, lines) in shapes {
        assert_tree_matches_cpu(&parser, &text_from_depths(&lines), name);
    }
}

#[test]
fn gpu_tree_refuses_what_it_cannot_represent_and_stays_usable() {
    let parser = gpu!();
    // 31 levels is the deepest the kernels track; 32 must be an error, not a wrong tree.
    let deepest = parser.upload(&format!("{} x\n", "|".repeat(31))).unwrap();
    assert!(parser.build_tree(&deepest).is_ok());
    let too_deep = parser
        .upload(&format!("| a\n{} x\n| b\n", "|".repeat(32)))
        .unwrap();
    let err = parser
        .build_tree(&too_deep)
        .err()
        .expect("depth 32 must be rejected");
    assert!(err.to_string().contains("32"), "{err}");
    // A blank line that is "deep" is not part of the tree and is fine.
    let blank_deep = parser
        .upload(&format!("| a\n{}\n| b\n", "|".repeat(40)))
        .unwrap();
    assert!(parser.build_tree(&blank_deep).is_ok());
    // And the parser is still usable afterwards.
    assert_tree_matches_cpu(&parser, "| a\n|| b\n| c\n", "after an error");
}

#[test]
fn gpu_tree_of_an_empty_document() {
    let parser = gpu!();
    let doc = parser.upload("").unwrap();
    let tree = parser.build_tree(&doc).unwrap();
    assert_eq!(tree.line_count(), 0);
    assert!(parser.download_links(&tree).unwrap().parent.is_empty());
}

/// Subtree queries return exactly the CPU's subtree range filtered by kind/depth.
#[test]
fn gpu_subtree_query_matches_cpu() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 2 << 20, 33);
    let cpu = oneline_simd::parse(&text);
    let links = oneline_simd::build_links(&cpu.nodes);
    let doc = parser.upload(&text).unwrap();
    let tree = parser.build_tree(&doc).unwrap();

    let all = cpu.nodes.len();
    let mut checked = 0;
    for root in (0..all).step_by(all / 40 + 1) {
        for (kind, depth) in [
            (None, None),
            (Some(NodeKind::Text), None),
            (Some(NodeKind::List), Some(2)),
            (None, Some(3)),
        ] {
            let want: Vec<u32> = links
                .subtree(root)
                .filter(|&i| {
                    kind.is_none_or(|k| cpu.nodes[i].kind() == k)
                        && depth.is_none_or(|d| cpu.nodes[i].depth == d)
                })
                .map(|i| i as u32)
                .collect();
            let got = parser
                .query_subtree(&doc, &tree, root, kind, depth, all)
                .unwrap();
            assert_eq!(
                got.indices, want,
                "root {root} kind {kind:?} depth {depth:?}"
            );
            assert_eq!(got.total_matches as usize, want.len());
            checked += 1;
        }
    }
    assert!(checked > 100);
    assert!(
        parser
            .query_subtree(&doc, &tree, all, None, None, 8)
            .is_err(),
        "root out of range"
    );
}

// ---- streaming upload ------------------------------------------------------

/// The upload goes through a ring of 4 x 16 MiB staging buffers. 90 MB wraps the ring more than once, so
/// a chunk written to the wrong place or a buffer overwritten while its DMA is still reading it changes
/// what a search finds. (The zero padding after the text is not tested: the shaders bound-check against
/// `text_len`, so its content cannot change a result.)
#[test]
fn streaming_upload_places_every_chunk_correctly() {
    let parser = gpu!();
    // 100-byte records with one marker at a fixed spot; a chunk boundary (16 MiB) is not a multiple of
    // 100, so records straddle chunks. The text ends mid-word (3 extra bytes) to exercise the padding.
    let record = format!("{:<46}MARKER{:<47}\n", "header", "x");
    assert_eq!(record.len(), 100);
    let records = 900_000usize;
    let mut text = record.repeat(records);
    text.push_str("end");

    let expected: Vec<u32> = (0..records).map(|k| (k * 100 + 46) as u32).collect();
    let tail_at = (records * 100) as u32;
    for round in 0..2 {
        // Two uploads in a row: the second reuses a ring whose buffers may still be in flight.
        let doc = parser.upload(&text).unwrap();
        let r = parser.search_document(&doc, b"MARKER", records).unwrap();
        assert_eq!(r.total_matches as usize, records, "round {round}");
        assert_eq!(
            r.offsets, expected,
            "round {round}: a chunk landed in the wrong place"
        );
        let end = parser.search_document(&doc, b"end", 4).unwrap();
        assert_eq!(
            end.offsets,
            vec![tail_at],
            "round {round}: the unaligned tail was lost"
        );
        assert_eq!(doc.text_len(), text.len());
    }
    // The one-shot path streams too.
    assert_eq!(
        parser
            .search_raw(&text, b"MARKER", records)
            .unwrap()
            .offsets,
        expected
    );
}

/// Sizes around the chunk size and around word alignment, including the empty-ring first use.
#[test]
fn streaming_upload_handles_sizes_around_the_chunk_size() {
    let parser = gpu!();
    let chunk = 16usize << 20;
    for len in [
        1usize,
        3,
        4,
        5,
        chunk - 1,
        chunk,
        chunk + 1,
        chunk + 3,
        2 * chunk,
        2 * chunk + 2,
        5 * chunk + 1,
    ] {
        let mut text = vec![b'.'; len];
        let first = 0usize;
        let last = len - 1;
        text[first] = b'A';
        text[last] = b'Z';
        let text = String::from_utf8(text).unwrap();
        let doc = parser.upload(&text).unwrap();
        let a = parser.search_document(&doc, b"A", 4).unwrap();
        let z = parser.search_document(&doc, b"Z", 4).unwrap();
        if len == 1 {
            // 'Z' overwrote 'A'
            assert_eq!(z.offsets, vec![0], "len 1");
        } else {
            assert_eq!(a.offsets, vec![0], "len {len}: first byte");
            assert_eq!(z.offsets, vec![last as u32], "len {len}: last byte");
        }
    }
}

// ---- dictionary search -----------------------------------------------------

use crate::{MAX_PATTERNS, ManyHit};

fn many_cpu(text: &str, patterns: &[&[u8]]) -> (Vec<u32>, Vec<ManyHit>) {
    let m = oneline_simd::ManyPatterns::new(patterns);
    let hits: Vec<ManyHit> = m
        .find_all_par(text.as_bytes())
        .into_iter()
        .map(|h| ManyHit {
            pattern: h.pattern,
            offset: h.offset,
        })
        .collect();
    let mut counts = vec![0u32; patterns.len()];
    for h in &hits {
        counts[h.pattern as usize] += 1;
    }
    (counts, hits)
}

/// The GPU's counts and hits must equal Aho-Corasick's, counting and recording.
fn assert_many_matches_cpu(parser: &GpuParser, text: &str, patterns: &[&[u8]], what: &str) {
    let (want_counts, want_hits) = many_cpu(text, patterns);
    let doc = parser.upload(text).unwrap();
    let set = parser.compile_patterns(&doc, patterns).unwrap();
    assert_eq!(set.len(), patterns.len());

    let counted = parser.search_many(&doc, &set, 0).unwrap();
    assert_eq!(counted.counts, want_counts, "{what}: counts (count-only)");
    assert_eq!(
        counted.total_hits as usize,
        want_hits.len(),
        "{what}: total"
    );
    assert!(counted.hits.is_empty());

    let full = parser
        .search_many(&doc, &set, want_hits.len().max(1))
        .unwrap();
    assert_eq!(full.counts, want_counts, "{what}: counts (recording)");
    assert_eq!(full.hits, want_hits, "{what}: hits");
}

#[test]
fn many_patterns_match_aho_corasick_on_generated_text() {
    let parser = gpu!();
    let vocab = datagen::prose_vocabulary();
    let text = datagen::generate(Profile::Prose, 3 << 20, 12);
    let bytes = text.as_bytes();
    let mut rng = datagen::Rng::new(77);
    let words: Vec<&[u8]> = vocab
        .iter()
        .filter(|w| w.len() >= 4)
        .map(|w| w.as_bytes())
        .collect();
    for n in [1usize, 2, 7, 64, 500, MAX_PATTERNS] {
        // A mix: vocabulary words (frequent and rare) and slices of the text itself.
        let mut patterns: Vec<&[u8]> = Vec::new();
        for i in 0..n {
            if i % 3 == 2 {
                let len = rng.range(4, 65) as usize;
                let at = rng.below((bytes.len() - len) as u64) as usize;
                patterns.push(&bytes[at..at + len]);
            } else {
                patterns.push(words[rng.below(words.len() as u64) as usize]);
            }
        }
        assert_many_matches_cpu(&parser, &text, &patterns, &format!("{n} patterns"));
    }
}

#[test]
fn many_patterns_handle_overlap_nesting_duplicates_and_edges() {
    let parser = gpu!();
    // Patterns that contain each other, share prefixes/suffixes, repeat, or sit at the text edges.
    let text = "abcdefabcdefabcd\0\0\0\0abcdxabcdef|||||- - -\nabcdabcd";
    let patterns: Vec<&[u8]> = vec![
        b"abcd",
        b"abcdef",
        b"bcde",
        b"abcdef",
        b"cdefabcd",
        b"abcdabcd",
        b"\0\0\0\0",
        b"||||",
        b"- - -",
        b"\n|",
        b"abcdx",
        b"xxxx-not-there",
        b"abcdabcd",
    ];
    // "\n|" is 2 bytes: not allowed, drop it for the comparison.
    let patterns: Vec<&[u8]> = patterns.into_iter().filter(|p| p.len() >= 4).collect();
    assert_many_matches_cpu(&parser, text, &patterns, "nested/overlapping");
    // Whole-text pattern, pattern at the very start and the very end, text shorter than a pattern.
    assert_many_matches_cpu(
        &parser,
        "abcdefgh",
        &[b"abcdefgh", b"abcd", b"efgh"],
        "whole text",
    );
    assert_many_matches_cpu(
        &parser,
        "abcdefgh",
        &[b"abcdefghi", b"zzzz"],
        "pattern longer than the text",
    );
    // Each pattern length from 4 to 64 occurring at every alignment.
    let mut text = String::new();
    let mut patterns_owned: Vec<Vec<u8>> = Vec::new();
    for len in 4..=64usize {
        let p: Vec<u8> = (0..len)
            .map(|i| b'a' + ((i * 7 + len) % 26) as u8)
            .collect();
        for shift in 0..4 {
            text.push_str(&"#".repeat(shift));
            text.push_str(std::str::from_utf8(&p).unwrap());
            text.push('\n');
        }
        patterns_owned.push(p);
    }
    let patterns: Vec<&[u8]> = patterns_owned.iter().map(|p| p.as_slice()).collect();
    assert_many_matches_cpu(
        &parser,
        &text,
        &patterns,
        "lengths 4..=64 at all alignments",
    );
}

/// Two different 4-byte windows with the same hash share a bucket: the verifier, not the filter,
/// must tell them apart.
#[test]
fn many_patterns_survive_hash_collisions() {
    let parser = gpu!();
    // Find two windows with the same 18-bit hash by brute force (a few hundred tries).
    let hash = |w: u32| w.wrapping_mul(0x9E37_79B1) >> 14;
    let mut seen = std::collections::HashMap::new();
    let mut pair = None;
    for i in 0u32..26 * 26 * 26 * 26 {
        // Lowercase letters only, so both windows are ASCII.
        let w = u32::from_le_bytes([
            b'a' + (i % 26) as u8,
            b'a' + (i / 26 % 26) as u8,
            b'a' + (i / 676 % 26) as u8,
            b'a' + (i / 17576) as u8,
        ]);
        if let Some(&other) = seen.get(&hash(w)) {
            if other != w {
                pair = Some((other, w));
                break;
            }
        }
        seen.insert(hash(w), w);
    }
    let (a, b) = pair.expect("a collision exists among the 456,976 lowercase windows");
    let (pa, pb) = (a.to_le_bytes().to_vec(), b.to_le_bytes().to_vec());
    assert_ne!(pa, pb);
    let text = format!(
        "zz{0}zz{1}zz{0}{1}zz",
        String::from_utf8_lossy(&pa),
        String::from_utf8_lossy(&pb)
    );
    // Only meaningful if the text is valid for both byte strings (ASCII range): ensure it.
    assert!(
        pa.iter().chain(&pb).all(|&c| c < 128),
        "collision pair must be ASCII for this text"
    );
    assert_many_matches_cpu(
        &parser,
        &text,
        &[pa.as_slice(), pb.as_slice(), b"zzzz-none"],
        "colliding windows",
    );
}

#[test]
fn many_patterns_validate_their_input_and_truncate_consistently() {
    let parser = gpu!();
    let doc = parser
        .upload(&datagen::generate(Profile::Prose, 2 << 20, 3))
        .unwrap();
    let ok: &[&[u8]] = &[b"abcd"];
    assert!(parser.compile_patterns(&doc, &[]).is_err(), "empty set");
    assert!(
        parser.compile_patterns(&doc, &[b"abc"]).is_err(),
        "3 bytes is too short"
    );
    assert!(
        parser.compile_patterns(&doc, &[&[b'a'; 65]]).is_err(),
        "65 bytes is too long"
    );
    let too_many: Vec<&[u8]> = vec![b"abcd"; MAX_PATTERNS + 1];
    assert!(
        parser.compile_patterns(&doc, &too_many).is_err(),
        "too many patterns"
    );
    assert!(parser.compile_patterns(&doc, ok).is_ok());

    // Truncation keeps the true totals.
    let vocab = datagen::prose_vocabulary();
    let patterns: Vec<&[u8]> = vec![vocab[0].as_bytes(), vocab[1].as_bytes()];
    let set = parser.compile_patterns(&doc, &patterns).unwrap();
    let all = parser.search_many(&doc, &set, 0).unwrap();
    assert!(all.total_hits > 1000);
    let few = parser.search_many(&doc, &set, 50).unwrap();
    assert_eq!(few.counts, all.counts);
    assert_eq!(few.total_hits, all.total_hits);
    assert_eq!(few.hits.len(), 50);
    // Repeated searches reuse the set's buffers and must give the same answer each time.
    for _ in 0..3 {
        assert_eq!(
            parser.search_many(&doc, &set, 0).unwrap().counts,
            all.counts
        );
    }
    // A set searched against an empty document finds nothing.
    let empty = parser.upload("").unwrap();
    assert_eq!(
        parser.search_many(&empty, &set, 8).unwrap().counts,
        vec![0, 0]
    );
}

/// A window hit near the start of the text can imply a pattern start *before* the text; the
/// unsigned subtraction would wrap. With a pattern that starts with NUL, out-of-range reads
/// "match" it, so a missing guard reports a bogus occurrence.
#[test]
fn many_patterns_reject_window_hits_that_imply_a_start_before_the_text() {
    let parser = gpu!();
    // NUL is common in this text, so the pattern's rarest window is "bcdq" (index 1), whose only hit
    // is at text offset 0.
    let text = format!("bcdq{}xyz", "\0".repeat(5000));
    let pattern = [0u8, b'b', b'c', b'd', b'q'];
    assert!(many_cpu(&text, &[&pattern, b"zzzz"]).1.is_empty());
    assert_many_matches_cpu(
        &parser,
        &text,
        &[&pattern, b"zzzz"],
        "start before the text",
    );
}

/// If candidates overflow what the compact strategy can hold, the query must be redone with the fused
/// kernel and still be exact. All-'a' text with an "aaaa" pattern makes EVERY position a candidate:
/// far more than one workgroup's tile (4000) and more than the buffer (text / 8).
#[test]
fn many_patterns_fall_back_when_candidates_overflow() {
    let parser = gpu!();
    let text = "a".repeat(1 << 20);
    // Enough patterns to select the compact strategy; one of them saturates the filter.
    let mut owned: Vec<Vec<u8>> = (0..crate::many::COMPACT_MIN_PATTERNS as u32 - 1)
        .map(|i| format!("zq{i:03}x").into_bytes())
        .collect();
    owned.push(b"aaaa".to_vec());
    let patterns: Vec<&[u8]> = owned.iter().map(|p| p.as_slice()).collect();
    assert_many_matches_cpu(&parser, &text, &patterns, "saturated filter");

    let doc = parser.upload(&text).unwrap();
    let set = parser.compile_patterns(&doc, &patterns).unwrap();
    let profile = parser.profile_search_many(&doc, &set, 0).unwrap();
    let step = |name: &str| {
        profile
            .host_steps
            .iter()
            .any(|(label, _)| label.contains(name))
    };
    assert!(
        step("(compact)"),
        "the compact strategy should have been tried first: {:?}",
        profile.host_steps
    );
    assert!(
        step("(fused)"),
        "and the fused kernel should have redone it: {:?}",
        profile.host_steps
    );

    // A sparse query on the same set stays on the compact path.
    let sparse = parser
        .upload(&datagen::generate(Profile::Prose, 2 << 20, 6))
        .unwrap();
    let sparse_set = parser.compile_patterns(&sparse, &patterns).unwrap();
    let profile = parser.profile_search_many(&sparse, &sparse_set, 0).unwrap();
    assert!(
        !profile
            .host_steps
            .iter()
            .any(|(label, _)| label.contains("(fused)")),
        "{:?}",
        profile.host_steps
    );
}

/// The candidate buffer overflowing (as opposed to one workgroup's tile) must also be reported and
/// redone: with the capacity capped at 100 entries, a sparse-looking query that finds a few thousand
/// candidates overflows only the buffer.
#[test]
fn many_patterns_fall_back_when_the_candidate_buffer_overflows() {
    let parser = gpu!();
    let vocab = datagen::prose_vocabulary();
    let text = datagen::generate(Profile::Prose, 2 << 20, 9);
    let owned: Vec<&[u8]> = vocab
        .iter()
        .filter(|w| w.len() >= 4)
        .take(crate::many::COMPACT_MIN_PATTERNS)
        .map(|w| w.as_bytes())
        .collect();
    parser
        .candidate_capacity_cap
        .store(100, std::sync::atomic::Ordering::Relaxed);
    assert_many_matches_cpu(&parser, &text, &owned, "buffer overflow");
    let doc = parser.upload(&text).unwrap();
    let set = parser.compile_patterns(&doc, &owned).unwrap();
    let profile = parser.profile_search_many(&doc, &set, 0).unwrap();
    assert!(
        profile
            .host_steps
            .iter()
            .any(|(label, _)| label.contains("(fused)")),
        "the buffer overflow should have forced the fused rerun: {:?}",
        profile.host_steps
    );
    parser
        .candidate_capacity_cap
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Rebuilding into the previous tree's buffers (and the shared working buffers) must give the same
/// answer as a fresh build, whatever stale contents the buffers hold: bigger then smaller then bigger
/// documents, the same document twice, and a failed build in between.
#[test]
fn gpu_tree_reuse_matches_fresh_builds() {
    use crate::GpuTree;
    let parser = gpu!();
    let docs: Vec<(String, String)> = vec![
        (
            "nested 3MB".into(),
            datagen::generate(Profile::Nested, 3 << 20, 1),
        ),
        (
            "mixed 600KB (smaller: reuses the buffers)".into(),
            datagen::generate(Profile::Mixed, 600 << 10, 2),
        ),
        (
            "prose 6MB (bigger: must reallocate)".into(),
            datagen::generate(Profile::Prose, 6 << 20, 3),
        ),
        (
            "nested 3MB again".into(),
            datagen::generate(Profile::Nested, 3 << 20, 1),
        ),
        ("tiny".into(), "| a\n|| b\n".into()),
        ("empty".into(), String::new()),
        (
            "nested 3MB after empty".into(),
            datagen::generate(Profile::Nested, 3 << 20, 4),
        ),
    ];
    let mut tree: Option<GpuTree> = None;
    for (name, text) in &docs {
        let doc = parser.upload(text).unwrap();
        // A document the kernels refuse, followed by one that REUSES the working buffers (so a stale error
        // flag would show): it must not poison them.
        if name == "tiny" {
            let deep = parser.upload(&format!("{} x\n", "|".repeat(40))).unwrap();
            assert!(parser.build_tree_reusing(&deep, tree.take()).is_err());
        }
        let rebuilt = parser.build_tree_reusing(&doc, tree.take()).unwrap();
        let cpu = oneline_simd::build_links(&oneline_simd::parse(text).nodes);
        assert_eq!(rebuilt.line_count(), cpu.len(), "{name}");
        assert!(
            parser.download_links(&rebuilt).unwrap() == cpu,
            "{name}: reused build differs from the CPU"
        );
        // A fresh build of the same document must agree too.
        let fresh = parser.build_tree(&doc).unwrap();
        assert!(
            parser.download_links(&fresh).unwrap() == cpu,
            "{name}: fresh build differs from the CPU"
        );
        // Subtree queries on a reused tree read only the first line_count entries.
        if cpu.len() > 10 {
            let root = cpu.len() / 3;
            let want: Vec<u32> = cpu.subtree(root).map(|i| i as u32).collect();
            let got = parser
                .query_subtree(&doc, &rebuilt, root, None, None, cpu.len())
                .unwrap();
            assert_eq!(got.indices, want, "{name}: subtree query on a reused tree");
        }
        tree = Some(rebuilt);
    }
}
