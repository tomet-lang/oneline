//! CPU vs GPU on the same question: "which lines are headings?".
//!
//! Each row answers it end to end from raw text, or from a pre-parsed document,
//! so the rows compare like with like:
//!
//! * `cold/*`   — raw text in, matching line indices out
//! * `warm/*`   — document already parsed and resident, many queries in a row

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oneline_core::datagen::Profile;
use oneline_core::{FlatNode, NodeKind};
use rayon::prelude::*;

mod common;
use common::*;

fn cpu_filter(nodes: &[FlatNode], kind: NodeKind) -> Vec<u32> {
    nodes
        .par_iter()
        .enumerate()
        .filter(|(_, n)| n.kind() == kind)
        .map(|(i, _)| i as u32)
        .collect()
}

fn cold_and_warm(c: &mut Criterion) {
    let gpu = gpu();
    let kind = NodeKind::Heading;

    for size in [MB, 16 * MB, 128 * MB] {
        let text = dataset(Profile::Mixed, size);
        let id = size_label(size);

        let mut cold = c.benchmark_group("compare/cold");
        configure(&mut cold, size);
        cold.throughput(Throughput::Bytes(text.len() as u64));
        cold.bench_with_input(BenchmarkId::new("cpu_parse_par+filter", &id), &text, |b, t| {
            b.iter(|| cpu_filter(&oneline_simd::parse_par(t).nodes, kind))
        });
        if let Some(p) = &gpu {
            cold.bench_with_input(BenchmarkId::new("gpu_one_shot_fused", &id), &text, |b, t| {
                b.iter(|| p.query_raw(t, Some(kind), None, 1 << 20).unwrap())
            });
        }
        cold.finish();

        let mut warm = c.benchmark_group("compare/warm");
        configure(&mut warm, size);
        warm.throughput(Throughput::Bytes(text.len() as u64));
        let tree = oneline_simd::parse_par(&text);
        warm.bench_function(BenchmarkId::new("cpu_filter_parsed", &id), |b| {
            b.iter(|| cpu_filter(&tree.nodes, kind))
        });
        if let Some(p) = &gpu {
            let doc = p.upload(&text).unwrap();
            warm.bench_function(BenchmarkId::new("gpu_resident", &id), |b| {
                b.iter(|| p.query_document(&doc, Some(kind), None, 1 << 20).unwrap())
            });
        }
        warm.finish();
    }
}

criterion_group!(benches, cold_and_warm);
criterion_main!(benches);
