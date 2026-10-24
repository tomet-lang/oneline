use std::path::Path;

use anyhow::{Context, Result};
use oneline_core::NodeKind;
use oneline_core::datagen::{self, Profile};

use super::{print_peak_rss, report, time_median};

pub fn run_bench(
    file: Option<&Path>,
    synthetic_mb: usize,
    profile: Profile,
    iterations: usize,
) -> Result<()> {
    let (name, content) = match file {
        Some(p) => {
            let s = std::fs::read_to_string(p)
                .with_context(|| format!("Failed to read file: {}", p.display()))?;
            (p.display().to_string(), s)
        }
        None => {
            println!(
                "==> Generating synthetic dataset ({synthetic_mb} MB, profile {})...",
                profile.name()
            );
            (
                format!("synthetic {} {synthetic_mb}MB", profile.name()),
                datagen::generate(profile, synthetic_mb << 20, 1),
            )
        }
    };
    let bytes = content.len();
    let lines = oneline_simd::count_newlines(content.as_bytes());
    println!(
        "==> Benchmarking {name} ({:.2} MB, {lines} lines, {:.0} B/line, {iterations} iterations)\n",
        bytes as f64 / (1 << 20) as f64,
        bytes as f64 / lines.max(1) as f64,
    );

    println!("CPU");
    report(
        "count newlines (AVX2)",
        bytes,
        time_median(iterations, || {
            Ok(oneline_simd::count_newlines(content.as_bytes()))
        })?,
    );
    report(
        "parse, single core",
        bytes,
        time_median(iterations, || {
            Ok(oneline_simd::try_parse(&content)?.nodes.len())
        })?,
    );
    report(
        &format!("parse, {} threads (Rayon)", rayon::current_num_threads()),
        bytes,
        time_median(iterations, || {
            Ok(oneline_simd::try_parse_par(&content)?.nodes.len())
        })?,
    );
    // Same parses, but handing the result vectors back each time: no fresh pages to fault in.
    let mut buffers = oneline_simd::ParseBuffers::new();
    report(
        "parse, single core, reusing buffers",
        bytes,
        time_median(iterations, || {
            let tree = oneline_simd::try_parse_reusing(&content, &mut buffers)?;
            let n = tree.nodes.len();
            buffers.recycle(tree);
            Ok(n)
        })?,
    );
    let mut buffers = oneline_simd::ParseBuffers::new();
    report(
        &format!(
            "parse, {} threads, reusing buffers",
            rayon::current_num_threads()
        ),
        bytes,
        time_median(iterations, || {
            let tree = oneline_simd::try_parse_par_reusing(&content, &mut buffers)?;
            let n = tree.nodes.len();
            buffers.recycle(tree);
            Ok(n)
        })?,
    );
    let tree = oneline_simd::try_parse_par(&content)?;
    report(
        "tree links (parent + subtree end), parallel",
        bytes,
        time_median(iterations, || Ok(oneline_simd::build_links(&tree.nodes)))?,
    );

    println!("\nGPU (every row except the last two includes upload over PCIe)");
    let parser = match oneline_gpu::GpuParser::new() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  skipped: GPU unavailable ({e:?})");
            return Ok(());
        }
    };
    if parser.is_software() {
        println!(
            "  WARNING: '{}' is a software adapter; its \"VRAM\" is system memory, so these numbers\n  say nothing about real GPU throughput or about where residency pays off.",
            parser.adapter_name()
        );
    }
    report(
        "parse_raw: upload + scan + parse + download",
        bytes,
        time_median(iterations, || {
            parser.parse_raw(&content).map(|t| t.nodes.len())
        })?,
    );
    report(
        "query_raw (fused): upload + scan + filter",
        bytes,
        time_median(iterations, || {
            parser.query_raw(&content, Some(NodeKind::Heading), None, 1024)
        })?,
    );

    let upload = time_median(iterations.min(5), || parser.upload(&content))?;
    report(
        "upload (resident): upload + scan + parse, once",
        bytes,
        upload,
    );
    let doc = parser.upload(&content)?;
    report(
        "query on resident doc (VRAM only)",
        bytes,
        time_median(iterations, || {
            parser.query_document(&doc, Some(NodeKind::Heading), None, 1024)
        })?,
    );

    // How many queries until residency pays for itself against one-shot fused queries.
    let one_shot = time_median(iterations, || {
        parser.query_raw(&content, Some(NodeKind::Heading), None, 1024)
    })?;
    let resident = time_median(iterations, || {
        parser.query_document(&doc, Some(NodeKind::Heading), None, 1024)
    })?;
    if one_shot > resident {
        let n = upload.as_secs_f64() / (one_shot - resident).as_secs_f64();
        println!("\n  residency breaks even after ~{:.1} queries", n.max(1.0));
    } else {
        println!("\n  resident queries are not faster than one-shot on this adapter");
    }
    print_peak_rss();
    Ok(())
}
