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

