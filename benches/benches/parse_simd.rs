//! CPU engines: newline counting, single-core and Rayon parsing, tree links.
//!
//! Throughput is reported per input byte, so numbers compare across sizes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oneline_core::TreeLinks;
use oneline_core::datagen::Profile;

mod common;
use common::*;

fn parse_by_size(c: &mut Criterion) {
    let mut g = c.benchmark_group("cpu/parse/mixed");
    for size in SIZES {
        let text = dataset(Profile::Mixed, size);
        configure(&mut g, size);
        g.throughput(Throughput::Bytes(text.len() as u64));
        let id = size_label(size);
        g.bench_with_input(BenchmarkId::new("count_newlines", &id), &text, |b, t| {
            b.iter(|| oneline_simd::count_newlines(t.as_bytes()))
        });
        g.bench_with_input(BenchmarkId::new("single_core", &id), &text, |b, t| {
            b.iter_with_large_drop(|| oneline_simd::parse(t))
        });
        g.bench_with_input(BenchmarkId::new("rayon", &id), &text, |b, t| {
            b.iter_with_large_drop(|| oneline_simd::parse_par(t))
        });
    }
    g.finish();
}

/// Same size, different shapes: a repeating sample flatters branch predictors,
/// very short lines stress per-line cost, long lines stress per-byte cost.
fn parse_by_profile(c: &mut Criterion) {
    let size = 16 * MB;
    let mut g = c.benchmark_group("cpu/parse/profiles-16MB");
    configure(&mut g, size);
    for profile in Profile::ALL {
        let text = dataset(profile, size);
        g.throughput(Throughput::Bytes(text.len() as u64));
        g.bench_with_input(
            BenchmarkId::new("single_core", profile.name()),
            &text,
            |b, t| b.iter_with_large_drop(|| oneline_simd::parse(t)),
        );
        g.bench_with_input(BenchmarkId::new("rayon", profile.name()), &text, |b, t| {
            b.iter_with_large_drop(|| oneline_simd::parse_par(t))
        });
    }
    g.finish();
}

/// Parallel ANSV vs the monotonic-stack reference.
fn tree_links(c: &mut Criterion) {
    let size = 16 * MB;
    let mut g = c.benchmark_group("cpu/links-16MB");
    configure(&mut g, size);
    for profile in Profile::ALL {
        let text = dataset(profile, size);
        let tree = oneline_simd::parse_par(&text);
        g.throughput(Throughput::Bytes(text.len() as u64));
        g.bench_with_input(
            BenchmarkId::new("parallel_ansv", profile.name()),
            &tree.nodes,
            |b, n| b.iter_with_large_drop(|| oneline_simd::build_links(n)),
        );
        g.bench_with_input(
            BenchmarkId::new("sequential_stack", profile.name()),
            &tree.nodes,
            |b, n| b.iter_with_large_drop(|| TreeLinks::from_nodes_sequential(n)),
        );
    }
    g.finish();
}

criterion_group!(benches, parse_by_size, parse_by_profile, tree_links);
criterion_main!(benches);
