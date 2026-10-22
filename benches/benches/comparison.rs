//! CPU vs GPU on the same question: "which lines are headings?".
//!
//! Each row answers it end to end from raw text, or from a pre-parsed document,
//! so the rows compare like with like:
//!
//! * `cold/*`   — raw text in, matching line indices out
//! * `warm/*`   — document already parsed and resident, many queries in a row
//!
//! `search` repeats the cold/warm split for substring search.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oneline_core::datagen::{self, Profile};
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


/// Substring search on `prose`, with patterns of very different selectivity taken from
/// the generator's vocabulary. Same question both ways: every occurrence offset, or just
/// the count.
///
/// * `compare/search-cold/*` — text only in host memory: upload + search vs CPU memmem
/// * `compare/search-warm/*` — text already in RAM (CPU) / resident in VRAM (GPU)
fn search(c: &mut Criterion) {
    let gpu = gpu();
    let vocab = datagen::prose_vocabulary();
    // rare word (~1k hits / 100 MB) and the most frequent word (~835k hits / 100 MB)
    let patterns = [("rare", vocab[3500].clone()), ("frequent", vocab[0].clone())];

    for size in [16 * MB, 128 * MB] {
        let text = dataset(Profile::Prose, size);
        let doc = gpu.as_ref().map(|p| p.upload(&text).unwrap());
        let id = size_label(size);
        for (name, pattern) in &patterns {
            let p = pattern.as_bytes();

            let mut cold = c.benchmark_group(format!("compare/search-cold/{name}"));
            configure(&mut cold, size);
            cold.throughput(Throughput::Bytes(text.len() as u64));
            cold.bench_function(BenchmarkId::new("cpu_memmem_par", &id), |b| {
                b.iter(|| oneline_simd::find_all_par(text.as_bytes(), p))
            });
            if let Some(g) = &gpu {
                cold.bench_function(BenchmarkId::new("gpu_one_shot", &id), |b| {
                    b.iter(|| g.search_raw(&text, p, 1 << 20).unwrap())
                });
            }
            cold.finish();

            let mut warm = c.benchmark_group(format!("compare/search-warm/{name}"));
            configure(&mut warm, size);
            warm.throughput(Throughput::Bytes(text.len() as u64));
            warm.bench_function(BenchmarkId::new("cpu_offsets", &id), |b| {
                b.iter(|| oneline_simd::find_all_par(text.as_bytes(), p))
            });
            warm.bench_function(BenchmarkId::new("cpu_count", &id), |b| {
                b.iter(|| oneline_simd::count_all_par(text.as_bytes(), p))
            });
            if let (Some(g), Some(d)) = (&gpu, &doc) {
                warm.bench_function(BenchmarkId::new("gpu_offsets", &id), |b| {
                    b.iter(|| g.search_document(d, p, 1 << 20).unwrap())
                });
                warm.bench_function(BenchmarkId::new("gpu_count", &id), |b| {
                    b.iter(|| g.search_document(d, p, 0).unwrap())
                });
            }
            warm.finish();
        }
    }
}

criterion_group!(benches, cold_and_warm, search);
criterion_main!(benches);
