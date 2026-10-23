use std::path::Path;

use anyhow::{Context, Result};
use oneline_core::datagen::{self, Profile};
use oneline_gpu::GpuParser;

use super::{print_peak_rss, report, time_median};

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


/// Real patterns from the text itself, at target selectivities, so a benchmark on a
/// real corpus does not depend on words I would pick by hand. Words (4..=16 chars) are
/// counted in ~16 MB sampled evenly over the text; for each target number of hits in the
/// *whole* text the word whose scaled count is closest is chosen. A 24-byte slice from
/// the middle of the text is added as a "typical long pattern" (usually 1 hit).
fn auto_patterns(text: &str) -> Vec<String> {
    use std::collections::HashMap;

    const WINDOW: usize = 4 << 20;
    const WINDOWS: usize = 4;
    const TARGETS: [f64; 4] = [300.0, 5_000.0, 100_000.0, 1_000_000.0];

    let len = text.len();
    let mut counts: HashMap<&str, u32> = HashMap::new();
    let mut sampled = 0usize;
    for i in 0..WINDOWS {
        let mut start = len.saturating_sub(WINDOW) / (WINDOWS - 1) * i;
        let mut end = (start + WINDOW).min(len);
        while !text.is_char_boundary(start) {
            start += 1;
        }
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        sampled += end - start;
        for w in text[start..end].split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if (4..=16).contains(&w.len()) {
                *counts.entry(w).or_default() += 1;
            }
        }
    }
    let scale = len as f64 / sampled.max(1) as f64;
    let mut words: Vec<(&str, f64)> = counts.into_iter().map(|(w, c)| (w, f64::from(c) * scale)).collect();
    words.sort_by(|a, b| a.0.cmp(b.0)); // deterministic tie-breaks

    let mut chosen: Vec<String> = Vec::new();
    for target in TARGETS {
        let best = words
            .iter()
            .filter(|(w, _)| !chosen.iter().any(|c| c == w))
            .min_by(|a, b| (a.1.ln() - target.ln()).abs().total_cmp(&(b.1.ln() - target.ln()).abs()));
        if let Some((w, _)) = best {
            chosen.push((*w).to_string());
        }
    }

    // A longer pattern: 24 bytes from the middle, cut at the end of its line.
    let mut start = len / 3;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let mut end = (start + 24).min(len);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let slice = text[start..end].split('\n').next().unwrap_or("");
    if slice.len() >= 8 {
        chosen.push(slice.to_string());
    }
    chosen
}

pub fn run_search(
    file: Option<&Path>,
    patterns: &[String],
    bench_mb: Option<usize>,
    profile: Profile,
    max_matches: usize,
    bench_file: bool,
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
    if bench_file {
        // Invalid UTF-8 (rare in text corpora) is replaced; offsets then refer to the converted text.
        let bytes = std::fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;
        let content = match String::from_utf8(bytes) {
            Ok(s) => s, // no copy: matters at multi-GB sizes
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
        let owned;
        let patterns: Vec<&str> = if patterns.is_empty() {
            owned = auto_patterns(&content);
            owned.iter().map(String::as_str).collect()
        } else {
            patterns.iter().map(String::as_str).collect()
        };
        return bench(&content, &patterns, &path.display().to_string(), max_matches, iterations);
    }
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
        // Both sides return every offset, so the GPU capacity must cover all hits.
        let max_matches = max_matches.max(want.len());
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
        report("CPU single core", bytes, time_median(iterations, || Ok(oneline_simd::find_all(text, p).len()))?);
        report("CPU parallel", bytes, time_median(iterations, || Ok(oneline_simd::find_all_par(text, p).len()))?);
        report("GPU one-shot (upload + search)", bytes, time_median(iterations, || Ok(parser.search_raw(content, p, max_matches)?.total_matches))?);
        report("GPU resident", bytes, time_median(iterations, || Ok(parser.search_document(&doc, p, max_matches)?.total_matches))?);

        println!("  first {TOP} offsets only");
        report("GPU resident", bytes, time_median(iterations, || Ok(parser.search_document(&doc, p, TOP)?.total_matches))?);

        println!("  count only (no offsets)");
        report("CPU parallel", bytes, time_median(iterations, || Ok(oneline_simd::count_all_par(text, p)))?);
        report("GPU resident", bytes, time_median(iterations, || Ok(parser.search_document(&doc, p, 0)?.total_matches))?);
    }
    print_peak_rss();
    Ok(())
}
