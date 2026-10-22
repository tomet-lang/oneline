use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use oneline_core::NodeKind;
use oneline_core::datagen::{self, Profile};
use oneline_gpu::GpuParser;

/// Median per step over `runs`; every run executes the same steps in the same order.
fn median_steps(runs: &[Vec<(&'static str, Duration)>]) -> Vec<(&'static str, Duration)> {
    let n = runs.first().map_or(0, |r| r.len());
    (0..n)
        .map(|i| {
            let mut v: Vec<Duration> = runs.iter().map(|r| r[i].1).collect();
            v.sort();
            (runs[0][i].0, v[v.len() / 2])
        })
        .collect()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn print_profile(title: &str, profiles: &[oneline_gpu::Profile]) {
    let wall = median(profiles.iter().map(|p| p.wall).collect());
    println!("\n{title}   (median of {} runs, host wall {wall:.2?})", profiles.len());

    println!("  host steps (time since the previous step; GPU work overlaps and surfaces in the step that waits)");
    let host: Vec<_> = profiles.iter().map(|p| p.host_steps.clone()).collect();
    for (label, d) in median_steps(&host) {
        println!("    {label:<52} {d:>10.2?}  {:>5.1}%", 100.0 * d.as_secs_f64() / wall.as_secs_f64());
    }

    if profiles.iter().any(|p| !p.gpu_passes.is_empty()) {
        println!("  GPU passes (timestamp queries)");
        let gpu: Vec<_> = profiles.iter().map(|p| p.gpu_passes.clone()).collect();
        let steps = median_steps(&gpu);
        for (label, d) in &steps {
            println!("    {label:<52} {d:>10.2?}");
        }
        let total: Duration = steps.iter().map(|&(_, d)| d).sum();
        println!("    {:<52} {total:>10.2?}  ({:.1}% of host wall)", "total GPU execution", 100.0 * total.as_secs_f64() / wall.as_secs_f64());
    } else {
        println!("  GPU passes: not available (adapter lacks TIMESTAMP_QUERY)");
    }
}

pub fn run_profile(
    file: Option<&Path>,
    synthetic_kb: usize,
    profile: Profile,
    kind: Option<NodeKind>,
    iterations: usize,
    max_matches: usize,
    pattern: Option<&str>,
) -> Result<()> {
    let content = match file {
        Some(p) => std::fs::read_to_string(p).with_context(|| format!("Failed to read file: {}", p.display()))?,
        None => datagen::generate(profile, synthetic_kb << 10, 1),
    };
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    println!(
        "==> {} bytes ({} profile), adapter {}, timestamps: {}{}",
        content.len(),
        profile.name(),
        parser.adapter_name(),
        if parser.timestamps_supported() { "yes" } else { "no" },
        if parser.is_software() { "  [SOFTWARE ADAPTER: numbers are not representative]" } else { "" },
    );

    // Warm-up so pipeline/driver first-use costs do not pollute the medians.
    let (doc, _) = parser.profile_upload(&content)?;
    parser.profile_query_document(&doc, kind, None, max_matches)?;
    parser.profile_query_raw(&content, kind, None, max_matches)?;

    if let Some(pattern) = pattern {
        let p = pattern.as_bytes();
        parser.profile_search_document(&doc, p, max_matches)?;
        let mut resident = Vec::new();
        let mut one_shot = Vec::new();
        for _ in 0..iterations {
            resident.push(parser.profile_search_document(&doc, p, max_matches)?);
            one_shot.push(parser.profile_search_raw(&content, p, max_matches)?);
        }
        print_profile(&format!("SEARCH {pattern:?} on resident document"), &resident);
        print_profile(&format!("SEARCH {pattern:?} one-shot: upload + search"), &one_shot);
        return Ok(());
    }

    let mut uploads = Vec::new();
    let mut resident = Vec::new();
    let mut one_shot = Vec::new();
    for _ in 0..iterations {
        uploads.push(parser.profile_upload(&content)?.1);
        resident.push(parser.profile_query_document(&doc, kind, None, max_matches)?);
        one_shot.push(parser.profile_query_raw(&content, kind, None, max_matches)?);
    }
    print_profile("UPLOAD (resident): upload + scan + parse", &uploads);
    print_profile("QUERY on resident document", &resident);
    print_profile("QUERY one-shot (fused): upload + scan + filter", &one_shot);
    Ok(())
}
