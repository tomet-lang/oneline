//! Shared dataset and adapter helpers for the benches.
#![allow(dead_code)]

use std::time::Duration;

use criterion::BenchmarkGroup;
use criterion::measurement::WallTime;
use oneline_core::datagen::{self, Profile};
use oneline_gpu::GpuParser;

pub const KB: usize = 1 << 10;
pub const MB: usize = 1 << 20;

/// Document sizes: cache resident, L2/L3 boundary, L3 spill, DRAM bound.
pub const SIZES: [usize; 4] = [64 * KB, MB, 16 * MB, 128 * MB];

pub fn size_label(bytes: usize) -> String {
    if bytes >= MB { format!("{}MB", bytes / MB) } else { format!("{}KB", bytes / KB) }
}

/// Deterministic input (seed 1) so runs are comparable across machines and commits.
pub fn dataset(profile: Profile, bytes: usize) -> String {
    datagen::generate(profile, bytes, 1)
}

/// Large inputs make iterations slow; keep sample counts low there.
pub fn configure(group: &mut BenchmarkGroup<'_, WallTime>, bytes: usize) {
    group
        .sample_size(if bytes >= 64 * MB { 10 } else { 20 })
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(if bytes >= 64 * MB { 4 } else { 2 }));
}

/// `None` (with a note on stderr) when there is no adapter, so GPU benches skip
/// instead of failing on headless machines.
pub fn gpu() -> Option<GpuParser> {
    match GpuParser::new() {
        Ok(p) => {
            if p.is_software() {
                eprintln!(
                    "WARNING: '{}' is a software adapter; GPU timings are not representative.",
                    p.adapter_name()
                );
            }
            Some(p)
        }
        Err(e) => {
            eprintln!("skipping GPU benchmarks: {e:?}");
            None
        }
    }
}
