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
        return vec![
            v[3500].clone(),
            v[200].clone(),
            v[0].clone(),
            format!("{} {}", v[0], v[1]),
        ];
    }
    ["naïve naïve naïve", "SIMD GPU", "lorem"]
        .iter()
        .map(|s| s.to_string())
        .collect()
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
    let mut words: Vec<(&str, f64)> = counts
        .into_iter()
        .map(|(w, c)| (w, f64::from(c) * scale))
        .collect();
    words.sort_by(|a, b| a.0.cmp(b.0)); // deterministic tie-breaks

    let mut chosen: Vec<String> = Vec::new();
    for target in TARGETS {
        let best = words
            .iter()
            .filter(|(w, _)| !chosen.iter().any(|c| c == w))
            .min_by(|a, b| {
                (a.1.ln() - target.ln())
                    .abs()
                    .total_cmp(&(b.1.ln() - target.ln()).abs())
            });
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
    dictionary: Option<usize>,
    sparse: bool,
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
        if let Some(n) = dictionary {
            return bench_dictionary(&content, profile.name(), n, sparse, iterations);
        }
        return bench(&content, &patterns, profile.name(), max_matches, iterations);
    }

    let path = file.context("specify a file or --bench-mb")?;
    if bench_file {
        // Invalid UTF-8 (rare in text corpora) is replaced; offsets then refer to the converted text.
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        let content = match String::from_utf8(bytes) {
            Ok(s) => s, // no copy: matters at multi-GB sizes
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        };
        if let Some(n) = dictionary {
            return bench_dictionary(&content, &path.display().to_string(), n, sparse, iterations);
        }
        let owned;
        let patterns: Vec<&str> = if patterns.is_empty() {
            owned = auto_patterns(&content);
            owned.iter().map(String::as_str).collect()
        } else {
            patterns.iter().map(String::as_str).collect()
        };
        return bench(
            &content,
            &patterns,
            &path.display().to_string(),
            max_matches,
            iterations,
        );
    }
    anyhow::ensure!(!patterns.is_empty(), "specify a pattern with --pattern");
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    let offsets = oneline_simd::scan_line_offsets(content.as_bytes());
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    let doc = parser.upload(&content)?;
    for pattern in patterns {
        let r = parser.search_document(&doc, pattern.as_bytes(), max_matches.min(100))?;
        println!(
            "==> {:?}: {} occurrences in {}",
            pattern,
            r.total_matches,
            path.display()
        );
        for &off in &r.offsets {
            let line = offsets.partition_point(|&o| o <= off);
            let start = offsets[line - 1] as usize;
            let end = offsets.get(line).map_or(content.len(), |&o| o as usize);
            println!(
                "  {:>5}:{:<4} {}",
                line,
                off as usize - start + 1,
                content[start..end].trim_end()
            );
        }
    }
    Ok(())
}

fn bench(
    content: &str,
    patterns: &[&str],
    profile: &str,
    max_matches: usize,
    iterations: usize,
) -> Result<()> {
    let bytes = content.len();
    let text = content.as_bytes();
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    println!(
        "==> substring search over {:.0} MB ({profile} profile), adapter {}{}",
        bytes as f64 / (1 << 20) as f64,
        parser.adapter_name(),
        if parser.is_software() {
            "  [SOFTWARE ADAPTER: GPU numbers are not representative]"
        } else {
            ""
        },
    );
    let doc = parser.upload(content)?;
    const TOP: usize = 1024;

    for pattern in patterns {
        let p = pattern.as_bytes();
        let want = oneline_simd::find_all_par(text, p);
        // Both sides return every offset, so the GPU capacity must cover all hits.
        let max_matches = max_matches.max(want.len());
        println!(
            "\npattern {pattern:?} ({} bytes): {} occurrences",
            p.len(),
            want.len()
        );

        // Agreement first: a fast wrong answer is worthless.
        let gpu = parser.search_document(&doc, p, max_matches)?;
        anyhow::ensure!(
            gpu.total_matches as usize == want.len()
                && (want.len() > max_matches || gpu.offsets == want),
            "GPU and CPU disagree for {pattern:?}: gpu {} vs cpu {}",
            gpu.total_matches,
            want.len()
        );

        println!(
            "  all offsets back (up to {max_matches})                  CPU = memchr::memmem on {} threads, text already in RAM",
            rayon::current_num_threads()
        );
        report(
            "CPU single core",
            bytes,
            time_median(iterations, || Ok(oneline_simd::find_all(text, p).len()))?,
        );
        report(
            "CPU parallel",
            bytes,
            time_median(iterations, || Ok(oneline_simd::find_all_par(text, p).len()))?,
        );
        report(
            "GPU one-shot (upload + search)",
            bytes,
            time_median(iterations, || {
                Ok(parser.search_raw(content, p, max_matches)?.total_matches)
            })?,
        );
        report(
            "GPU resident",
            bytes,
            time_median(iterations, || {
                Ok(parser.search_document(&doc, p, max_matches)?.total_matches)
            })?,
        );

        println!("  first {TOP} offsets only");
        report(
            "GPU resident",
            bytes,
            time_median(iterations, || {
                Ok(parser.search_document(&doc, p, TOP)?.total_matches)
            })?,
        );

        println!("  count only (no offsets)");
        report(
            "CPU parallel",
            bytes,
            time_median(iterations, || Ok(oneline_simd::count_all_par(text, p)))?,
        );
        report(
            "GPU resident",
            bytes,
            time_median(iterations, || {
                Ok(parser.search_document(&doc, p, 0)?.total_matches)
            })?,
        );
    }
    print_peak_rss();
    Ok(())
}

/// `n` dictionary words taken from the text itself: words (4..=16 chars) whose estimated
/// number of hits in the whole text is in `[100, 20_000]` (a sparse, realistic dictionary:
/// keywords, identifiers, names), chosen in a fixed pseudo-random order. If the text has
/// fewer such words, the closest ones fill the rest.
fn pick_dictionary(text: &str, n: usize, sparse: bool) -> Vec<String> {
    use std::collections::HashMap;
    const WINDOW: usize = 4 << 20;
    const WINDOWS: usize = 4;
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
    let mut words: Vec<(&str, f64)> = counts
        .into_iter()
        .map(|(w, c)| (w, f64::from(c) * scale))
        .collect();
    // Deterministic order independent of HashMap iteration: by a hash of the word.
    let key = |w: &str| {
        w.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    };
    words.sort_by_key(|&(w, _)| key(w));
    let band = if sparse {
        20.0..=1_000.0
    } else {
        100.0..=20_000.0
    };
    let (lo, hi) = (*band.start(), *band.end());
    let (mut inside, mut outside): (Vec<_>, Vec<_>) =
        words.into_iter().partition(|&(_, est)| band.contains(&est));
    outside.sort_by(|a, b| {
        let d = |est: f64| {
            if est < lo {
                lo / est.max(1.0)
            } else {
                est / hi
            }
        };
        d(a.1).total_cmp(&d(b.1))
    });
    inside.extend(outside);
    inside
        .into_iter()
        .take(n)
        .map(|(w, _)| w.to_string())
        .collect()
}

fn bench_dictionary(
    content: &str,
    label: &str,
    n: usize,
    sparse: bool,
    iterations: usize,
) -> Result<()> {
    use oneline_simd::{ManyPatterns, count_all_par};
    let bytes = content.len();
    let text = content.as_bytes();
    let parser = GpuParser::new().context("Failed to initialize GPU parser")?;
    let words = pick_dictionary(content, n, sparse);
    anyhow::ensure!(!words.is_empty(), "no usable words found in the text");
    let patterns: Vec<&[u8]> = words.iter().map(|w| w.as_bytes()).collect();
    println!(
        "==> dictionary search: {} patterns over {:.0} MB ({label}), adapter {}{}",
        patterns.len(),
        bytes as f64 / (1 << 20) as f64,
        parser.adapter_name(),
        if parser.is_software() {
            "  [SOFTWARE ADAPTER: GPU numbers are not representative]"
        } else {
            ""
        },
    );
    let doc = parser.upload(content)?;
    let set = parser.compile_patterns(&doc, &patterns)?;
    let many = ManyPatterns::new(&patterns);

    // Agreement first.
    let want = many.count_par(text, patterns.len());
    let got = parser.search_many(&doc, &set, 0)?;
    let total: u64 = want.iter().sum();
    anyhow::ensure!(
        got.counts
            .iter()
            .zip(&want)
            .all(|(&g, &w)| u64::from(g) == w),
        "GPU and CPU counts disagree (gpu total {}, cpu total {total})",
        got.total_hits
    );
    println!(
        "  {total} occurrences in total ({:.1} per MB)\n",
        total as f64 / (bytes as f64 / (1 << 20) as f64)
    );

    println!("  count per pattern");
    if patterns.len() <= 64 {
        report(
            &format!("CPU: {} x memmem (all cores)", patterns.len()),
            bytes,
            time_median(iterations, || {
                Ok(patterns
                    .iter()
                    .map(|p| count_all_par(text, p))
                    .sum::<usize>())
            })?,
        );
    }
    report(
        "CPU: aho-corasick DFA (all cores)",
        bytes,
        time_median(iterations, || {
            Ok(many.count_par(text, patterns.len()).len())
        })?,
    );
    report(
        "GPU resident",
        bytes,
        time_median(iterations, || {
            Ok(parser.search_many(&doc, &set, 0)?.total_hits)
        })?,
    );
    if parser.timestamps_supported() {
        let mut kernel: Vec<std::time::Duration> = Vec::new();
        for _ in 0..iterations {
            kernel.push(parser.profile_search_many(&doc, &set, 0)?.gpu_total());
        }
        kernel.sort();
        report(
            "GPU kernel alone (timestamps)",
            bytes,
            kernel[kernel.len() / 2],
        );
    }

    let cap = (total as usize + 16).min(1 << 24);
    println!("  every hit (offset + pattern)");
    report(
        "CPU: aho-corasick DFA (all cores)",
        bytes,
        time_median(iterations, || Ok(many.find_all_par(text).len()))?,
    );
    report(
        "GPU resident",
        bytes,
        time_median(iterations, || {
            Ok(parser.search_many(&doc, &set, cap)?.hits.len())
        })?,
    );
    print_peak_rss();
    Ok(())
}
