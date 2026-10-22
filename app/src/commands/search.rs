use std::path::Path;

use anyhow::{Context, Result};
use oneline_core::datagen::{self, Profile};
use oneline_gpu::GpuParser;

use super::{report, time_avg};

/// Patterns of decreasing selectivity, so the benchmark covers "few hits" (scan
/// bound) through "very many hits" (output bound). On `prose` they come from the
/// generator's vocabulary (rare word, mid word, the most frequent word, a phrase).
fn default_patterns(profile: Profile) -> Vec<String> {
    if profile == Profile::Prose {
        let v = datagen::prose_vocabulary();
        return vec![v[3500].clone(), v[200].clone(), v[0].clone(), format!("{} {}", v[0], v[1])];
    }
    ["naïve naïve naïve", "SIMD GPU", "lorem"].iter().map(|s| s.to_string()).collect()
}

pub fn run_search(
    file: Option<&Path>,
    patterns: &[String],
    bench_mb: Option<usize>,
    profile: Profile,
    max_matches: usize,
    iterations: usize,
) -> Result<()> {
    if let Some(mb) = bench_mb {
        let content = datagen::generate(profile, mb << 20, 1);
        let owned;
        let patterns: Vec<&str> = if patterns.is_empty() {
            owned = default_patterns(profile);
            owned.iter().map(String::as_str).collect()
        } else {
            patterns.iter().map(String::as_str).collect()
        };
        return bench(&content, &patterns, profile.name(), max_matches, iterations);
    }

    let path = file.context("specify a file or --bench-mb")?;
    anyhow::ensure!(!patterns.is_empty(), "specify a pattern with --pattern");
    let content = std::fs::read_to_string(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
    let offsets = oneline_simd::scan_line_offsets(content.as_bytes());
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    let doc = parser.upload(&content)?;
    for pattern in patterns {
        let r = parser.search_document(&doc, pattern.as_bytes(), max_matches.min(100))?;
        println!("==> {:?}: {} occurrences in {}", pattern, r.total_matches, path.display());
        for &off in &r.offsets {
            let line = offsets.partition_point(|&o| o <= off);
            let start = offsets[line - 1] as usize;
            let end = offsets.get(line).map_or(content.len(), |&o| o as usize);
            println!("  {:>5}:{:<4} {}", line, off as usize - start + 1, content[start..end].trim_end());
        }
    }
    Ok(())
}

fn bench(content: &str, patterns: &[&str], profile: &str, max_matches: usize, iterations: usize) -> Result<()> {
    let bytes = content.len();
    let text = content.as_bytes();
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    println!(
        "==> substring search over {:.0} MB ({profile} profile), adapter {}{}",
        bytes as f64 / (1 << 20) as f64,
        parser.adapter_name(),
        if parser.is_software() { "  [SOFTWARE ADAPTER: GPU numbers are not representative]" } else { "" },
    );
    let doc = parser.upload(content)?;
    const TOP: usize = 1024;

    for pattern in patterns {
        let p = pattern.as_bytes();
        let want = oneline_simd::find_all_par(text, p);
        println!("\npattern {pattern:?} ({} bytes): {} occurrences", p.len(), want.len());

        // Agreement first: a fast wrong answer is worthless.
        let gpu = parser.search_document(&doc, p, max_matches)?;
        anyhow::ensure!(
            gpu.total_matches as usize == want.len() && (want.len() > max_matches || gpu.offsets == want),
            "GPU and CPU disagree for {pattern:?}: gpu {} vs cpu {}",
            gpu.total_matches,
            want.len()
        );

        println!("  all offsets back (up to {max_matches})                  CPU = memchr::memmem on {} threads, text already in RAM", rayon::current_num_threads());
        report("CPU single core", bytes, time_avg(iterations, || Ok(oneline_simd::find_all(text, p).len()))?);
        report("CPU parallel", bytes, time_avg(iterations, || Ok(oneline_simd::find_all_par(text, p).len()))?);
        report("GPU one-shot (upload + search)", bytes, time_avg(iterations, || Ok(parser.search_raw(content, p, max_matches)?.total_matches))?);
        report("GPU resident", bytes, time_avg(iterations, || Ok(parser.search_document(&doc, p, max_matches)?.total_matches))?);

        println!("  first {TOP} offsets only");
        report("GPU resident", bytes, time_avg(iterations, || Ok(parser.search_document(&doc, p, TOP)?.total_matches))?);

        println!("  count only (no offsets)");
        report("CPU parallel", bytes, time_avg(iterations, || Ok(oneline_simd::count_all_par(text, p)))?);
        report("GPU resident", bytes, time_avg(iterations, || Ok(parser.search_document(&doc, p, 0)?.total_matches))?);
    }
    Ok(())
}
