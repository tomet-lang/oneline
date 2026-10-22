//! GPU stages. "one-shot" rows include the PCIe upload; "resident" rows do not.
//! Skips cleanly when no adapter is present.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oneline_core::NodeKind;
use oneline_core::datagen::Profile;

mod common;
use common::*;

fn gpu_stages(c: &mut Criterion) {
    let Some(parser) = gpu() else { return };
    let mut g = c.benchmark_group("gpu/mixed");
    for size in SIZES {
        let text = dataset(Profile::Mixed, size);
        configure(&mut g, size);
        g.throughput(Throughput::Bytes(text.len() as u64));
        let id = size_label(size);

        g.bench_with_input(BenchmarkId::new("scan_line_offsets", &id), &text, |b, t| {
            b.iter_with_large_drop(|| parser.scan_line_offsets(t).unwrap())
        });
        g.bench_with_input(BenchmarkId::new("parse_raw_one_shot", &id), &text, |b, t| {
            b.iter_with_large_drop(|| parser.parse_raw(t).unwrap())
        });
        g.bench_with_input(BenchmarkId::new("query_raw_one_shot_fused", &id), &text, |b, t| {
            b.iter(|| parser.query_raw(t, Some(NodeKind::Heading), None, 1024).unwrap())
        });
        g.bench_with_input(BenchmarkId::new("upload_resident", &id), &text, |b, t| {
            b.iter_with_large_drop(|| parser.upload(t).unwrap())
        });

        let doc = parser.upload(&text).unwrap();
        g.bench_function(BenchmarkId::new("query_resident", &id), |b| {
            b.iter(|| parser.query_document(&doc, Some(NodeKind::Heading), None, 1024).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, gpu_stages);
criterion_main!(benches);
