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
    let text = "|>[ Header ]\n| Body text\n||>[ Callout ]\n|| Content.\n";
    let offsets = oneline_simd::scan_line_offsets(text.as_bytes());
    let tree = parser.parse(text, &offsets).unwrap();

    assert_eq!(tree.nodes.len(), 4);
    assert_eq!((tree.nodes[0].depth, tree.nodes[0].kind()), (1, NodeKind::Callout));
    assert_eq!((tree.nodes[1].depth, tree.nodes[1].kind()), (1, NodeKind::Text));
    assert_eq!((tree.nodes[2].depth, tree.nodes[2].kind()), (2, NodeKind::Callout));
    assert_eq!((tree.nodes[3].depth, tree.nodes[3].kind()), (2, NodeKind::Text));
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
        "|>[ Header ]\n| Paragraph\n|| Content\n",
    ] {
        let cpu = oneline_simd::scan_line_offsets(case.as_bytes());
        assert_eq!(parser.scan_line_offsets(case).unwrap(), cpu, "case {case:?}");
    }
}

#[test]
fn test_gpu_parse_raw() {
    let parser = gpu!();
    let text = "|>[ Header ]\n| Body text\n||>[ Callout ]\n|| Content.\n";
    let tree = parser.parse_raw(text).unwrap();
    assert_eq!(tree.nodes.len(), 4);
    assert_eq!(tree.nodes[0].kind(), NodeKind::Callout);
    assert_eq!(tree.nodes[3].depth, 2);
    assert_eq!(tree.line_offsets, vec![0, 13, 25, 40]);
}

#[test]
fn test_gpu_query_raw() {
    let parser = gpu!();
    let text = "| @card(title:\"A\")\n|| Content\n| @card(title:\"B\")\n|| More\n|=[ Header ]\n";
    assert_eq!(parser.query_raw(text, Some(NodeKind::Element), None, 100).unwrap(), vec![0, 2]);
    assert_eq!(parser.query_raw(text, None, Some(2), 100).unwrap(), vec![1, 3]);

    let offsets = oneline_simd::scan_line_offsets(text.as_bytes());
    assert_eq!(parser.query(text, &offsets, Some(NodeKind::Element), None, 100).unwrap(), vec![0, 2]);
}

/// GPU parse must equal the CPU parse bit for bit, on every profile and on the
/// awkward edge cases.
#[test]
fn gpu_parse_matches_cpu() {
    let parser = gpu!();
    for case in ["x", "x\n", "\n", "\r\n", "a\r\nb", "||||\n|||| \n", "```\n- x\n```\n", "| 日本語\n"] {
        let cpu = oneline_simd::parse(case);
        let gpu = parser.parse_raw(case).unwrap();
        assert_eq!(gpu, cpu, "case {case:?}");
    }
    for p in Profile::ALL {
        let text = datagen::generate(p, 3 << 20, 11);
        assert_eq!(parser.parse_raw(&text).unwrap(), oneline_simd::parse(&text), "profile {p:?}");
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

    let kinds = [None, Some(NodeKind::Text), Some(NodeKind::Heading), Some(NodeKind::Element),
                 Some(NodeKind::List), Some(NodeKind::Callout), Some(NodeKind::Raw)];
    let depths = [None, Some(0), Some(1), Some(3), Some(50)];
    let all = tree.nodes.len();
    for &kind in &kinds {
        for &depth in &depths {
            let want = cpu_matches(&tree.nodes, kind, depth);
            let resident = parser.query_document(&doc, kind, depth, all).unwrap();
            assert_eq!(resident.indices, want, "resident kind={kind:?} depth={depth:?}");
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
    let r = parser.query_document(&doc, Some(NodeKind::Text), None, 10).unwrap();
    assert_eq!(r.total_matches as usize, want.len());
    assert_eq!(r.indices.len(), 10);
    assert!(r.indices.iter().all(|i| want.binary_search(i).is_ok()), "subset of true matches");

    let none = parser.query_document(&doc, Some(NodeKind::Text), None, 0).unwrap();
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
    assert_eq!(parser.query_document(&doc, None, None, 8).unwrap().total_matches, 0);
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
    assert_eq!(resident.total_matches as usize, want.len(), "resident count for {pattern:?}");
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
        let mut patterns: Vec<Vec<u8>> = ["lorem", "SIMD GPU", "naïve", "|", "||", "\n|", "階層 解析", "- ", "zzzz-absent"]
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
    assert!(parser.search_document(&doc, &[b'a'; MAX_PATTERN_LEN + 1], 8).is_err());
    assert!(parser.search_document(&doc, &[b'a'; MAX_PATTERN_LEN], 8).is_ok());
    assert!(parser.search_raw("abc", b"", 8).is_err());
    // An error must not poison later calls (the scratch is dropped and rebuilt).
    assert_eq!(parser.search_document(&doc, b"b", 8).unwrap().offsets, vec![1]);
}

#[test]
fn search_truncation_reports_true_total() {
    let parser = gpu!();
    let text = datagen::generate(Profile::Mixed, 2 << 20, 2);
    let want = oneline_simd::find_all(text.as_bytes(), b"|");
    assert!(want.len() > 10_000, "need a dense pattern to stress the workgroup reservation");

    let doc = parser.upload(&text).unwrap();
    let r = parser.search_document(&doc, b"|", 100).unwrap();
    assert_eq!(r.total_matches as usize, want.len());
    assert_eq!(r.offsets.len(), 100);
    assert!(r.offsets.iter().all(|o| want.binary_search(o).is_ok()), "subset of true matches");

    let none = parser.search_document(&doc, b"|", 0).unwrap();
    assert_eq!(none.total_matches as usize, want.len());
    assert!(none.offsets.is_empty());
}

#[test]
fn search_on_empty_document() {
    let parser = gpu!();
    let doc = parser.upload("").unwrap();
    assert_eq!(parser.search_document(&doc, b"a", 8).unwrap().total_matches, 0);
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
    assert_eq!(parser.search_document(&doc, &pattern, 16).unwrap().total_matches, 0);
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
        panic!("{what}: parent differs at line {i}: gpu {} cpu {}", gpu.parent[i], cpu.parent[i]);
    }
    if let Some(i) = (0..cpu.subtree_end.len()).find(|&i| gpu.subtree_end[i] != cpu.subtree_end[i]) {
        panic!("{what}: subtree_end differs at line {i}: gpu {} cpu {}", gpu.subtree_end[i], cpu.subtree_end[i]);
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
    for n in [1usize, 2, 31, 32, 33, 255, 256, 257, 1000, 2047, 2048, 2049, 4095, 4096, 4097, 6145] {
        let lines: Vec<(u32, bool)> = (0..n).map(|_| (rng.below(5) as u32, rng.chance(15))).collect();
        assert_tree_matches_cpu(&parser, &text_from_depths(&lines), &format!("{n} random lines"));
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
        ("header then flat", std::iter::once((0, false)).chain(std::iter::repeat_n((1, false), n - 1)).collect()),
        ("flat then header at the end", std::iter::repeat_n((1, false), n - 1).chain(std::iter::once((0, false))).collect()),
        ("strictly increasing to the limit", (0..n as u32).map(|i| (i % 32, false)).collect()),
        ("strictly decreasing sawtooth", (0..n as u32).map(|i| (31 - i % 32, false)).collect()),
        ("alternating", (0..n as u32).map(|i| (i % 2, false)).collect()),
        ("only blanks", vec![(3, true); n]),
        ("blanks between levels", (0..n as u32).map(|i| (i % 4, i % 3 == 0)).collect()),
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
    let too_deep = parser.upload(&format!("| a\n{} x\n| b\n", "|".repeat(32))).unwrap();
    let err = parser.build_tree(&too_deep).err().expect("depth 32 must be rejected");
    assert!(err.to_string().contains("32"), "{err}");
    // A blank line that is "deep" is not part of the tree and is fine.
    let blank_deep = parser.upload(&format!("| a\n{}\n| b\n", "|".repeat(40))).unwrap();
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
        for (kind, depth) in [(None, None), (Some(NodeKind::Text), None), (Some(NodeKind::List), Some(2)), (None, Some(3))] {
            let want: Vec<u32> = links
                .subtree(root)
                .filter(|&i| kind.is_none_or(|k| cpu.nodes[i].kind() == k) && depth.is_none_or(|d| cpu.nodes[i].depth == d))
                .map(|i| i as u32)
                .collect();
            let got = parser.query_subtree(&doc, &tree, root, kind, depth, all).unwrap();
            assert_eq!(got.indices, want, "root {root} kind {kind:?} depth {depth:?}");
            assert_eq!(got.total_matches as usize, want.len());
            checked += 1;
        }
    }
    assert!(checked > 100);
    assert!(parser.query_subtree(&doc, &tree, all, None, None, 8).is_err(), "root out of range");
}
