use std::path::Path;

use anyhow::{Context, Result};
use oneline_core::datagen::{self, Profile};
use oneline_core::NodeKind;

use super::{report, time_median};

pub(crate) fn parse_kind(k: &str) -> Option<NodeKind> {
    match k.to_lowercase().as_str() {
        "heading" => Some(NodeKind::Heading),
        "element" => Some(NodeKind::Element),
        "list" => Some(NodeKind::List),
        "ordered-list" => Some(NodeKind::OrderedList),
        "comment" => Some(NodeKind::Comment),
        "text" => Some(NodeKind::Text),
        "connect" => Some(NodeKind::Connect),
        _ => None,
    }
}

pub fn run_query(
    file: Option<&Path>,
    kind: Option<&str>,
    depth: Option<u32>,
    bench_mb: Option<usize>,
    profile: Profile,
) -> Result<()> {
    let node_kind = match kind {
        None => None,
        Some(k) => Some(parse_kind(k).with_context(|| format!("unknown kind {k:?} (heading, element, list, ordered-list, connect, comment, text)"))?),
    };
    let parser = oneline_gpu::GpuParser::new().context("Failed to initialize GPU parser for query")?;

    if let Some(mb) = bench_mb {
        println!("==> Benchmarking queries on synthetic {mb} MB dataset (profile {})...", profile.name());
        let content = datagen::generate(profile, mb << 20, 1);
        let bytes = content.len();
        let iters = 20;

        let one_shot = parser.query_raw(&content, node_kind, depth, 1024)?;
        println!("    filter: kind={node_kind:?} depth={depth:?}, {} sample matches", one_shot.len());
        report("one-shot (upload + scan + fused filter)", bytes, time_median(iters, || parser.query_raw(&content, node_kind, depth, 1024))?);

        let doc = parser.upload(&content)?;
        let r = parser.query_document(&doc, node_kind, depth, 1024)?;
        println!("    resident: {} total matches", r.total_matches);
        report("resident doc (VRAM only)", bytes, time_median(iters, || parser.query_document(&doc, node_kind, depth, 1024))?);
    } else if let Some(file_path) = file {
        let content = std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
        let offsets = parser.scan_line_offsets(&content)?;
        let matches = parser.query_raw(&content, node_kind, depth, 100)?;

        println!("==> Found {} matching lines in {} (GPU pipeline):", matches.len(), file_path.display());
        for &idx in &matches {
            let start = offsets[idx as usize] as usize;
            let end = offsets.get(idx as usize + 1).map_or(content.len(), |&o| o as usize);
            println!("  line {:>4}: {}", idx + 1, content[start..end].trim());
        }
    } else {
        eprintln!("Error: specify either a file or --bench-mb");
    }
    Ok(())
}
