use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use oneline_tomet::compat::{self, Outcome};

#[derive(Default)]
struct Tally {
    total: usize,
    same: usize,
    reference_error: usize,
    examples: Vec<(PathBuf, String, String, String)>,
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)?.collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            // Build output and dependency trees hold copies of the same documents.
            if p.is_dir() && matches!(p.file_name().and_then(|n| n.to_str()), Some("target" | "node_modules" | "dist" | ".git")) {
                continue;
            }
            collect(&p, out)?;
        }
    } else if path.extension().is_some_and(|e| e == "tmt") {
        out.push(path.to_path_buf());
    }
    Ok(())
}

fn shorten(s: &str, max: usize) -> String {
    let mut t: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        t.push('…');
    }
    t
}

/// How far oneline's reading of real tomet documents is from tomet's own parser.
pub fn run_compat(paths: &[PathBuf], show: usize, texts: &[String]) -> Result<()> {
    if !texts.is_empty() {
        for text in texts {
            println!("--- {text:?}");
            match compat::compare(text) {
                Outcome::Same => println!("  same"),
                Outcome::ReferenceError(e) => println!("  tomet rejects it: {}", shorten(&e, 200)),
                Outcome::Different { real, ours } => {
                    println!("  tomet: {}", shorten(&real, 600));
                    println!("  ours:  {}", shorten(&ours, 600));
                }
            }
        }
        return Ok(());
    }
    let mut files = Vec::new();
    for p in paths {
        collect(p, &mut files).with_context(|| format!("reading {}", p.display()))?;
    }
    anyhow::ensure!(!files.is_empty(), "no .tmt files under the given paths");

    let mut by_shape: BTreeMap<&'static str, Tally> = BTreeMap::new();
    let mut excluded: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new(); // reason -> (chunks, lines)
    let (mut chunks_total, mut lines_total) = (0usize, 0usize);
    let (mut in_chunks, mut in_lines) = (0usize, 0usize);
    let (mut same_chunks, mut same_lines) = (0usize, 0usize);
    let mut tomet_rejects = 0usize;

    for file in &files {
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        for chunk in compat::chunks(&text) {
            let n = chunk.lines().count();
            chunks_total += 1;
            lines_total += n;
            if let Some(reason) = compat::outside_subset(chunk) {
                let e = excluded.entry(reason).or_default();
                e.0 += 1;
                e.1 += n;
                continue;
            }
            in_chunks += 1;
            in_lines += n;
            let tally = by_shape.entry(compat::shape(chunk)).or_default();
            tally.total += 1;
            match compat::compare(chunk) {
                Outcome::Same => {
                    tally.same += 1;
                    same_chunks += 1;
                    same_lines += n;
                }
                Outcome::ReferenceError(_) => {
                    tally.reference_error += 1;
                    tomet_rejects += 1;
                }
                Outcome::Different { real, ours } => {
                    if tally.examples.len() < show {
                        tally.examples.push((file.clone(), chunk.to_string(), real, ours));
                    }
                }
            }
        }
    }

    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    println!("files: {}   chunks (runs of non-blank lines): {chunks_total}   lines: {lines_total}\n", files.len());
    println!("outside the subset (not counted below):");
    let mut ex: Vec<_> = excluded.iter().collect();
    ex.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
    for (reason, (c, l)) in &ex {
        println!("  {reason:<34} {c:>6} chunks {l:>7} lines");
    }
    println!(
        "\ncoverage: {in_chunks} of {chunks_total} chunks ({:.1}%), {in_lines} of {lines_total} lines ({:.1}%) are inside the subset",
        pct(in_chunks, chunks_total),
        pct(in_lines, lines_total)
    );
    println!(
        "fidelity: {same_chunks} of {in_chunks} inside chunks are identical to tomet ({:.1}%), {same_lines} of {in_lines} lines ({:.1}%)   [tomet rejects {tomet_rejects}]\n",
        pct(same_chunks, in_chunks),
        pct(same_lines, in_lines)
    );

    println!("{:<24} {:>7} {:>9} {:>7} {:>10}", "inside, first line", "chunks", "identical", "%", "tomet-err");
    let mut rows: Vec<_> = by_shape.iter().collect();
    rows.sort_by_key(|(_, t)| std::cmp::Reverse(t.total));
    for (shape, t) in &rows {
        println!("{:<24} {:>7} {:>9} {:>6.1}% {:>10}", shape, t.total, t.same, pct(t.same, t.total), t.reference_error);
    }

    if show > 0 {
        for (shape, t) in &rows {
            for (file, chunk, real, ours) in &t.examples {
                println!("\n=== [{shape}] {}", file.display());
                for l in chunk.lines().take(6) {
                    println!("    | {}", shorten(l, 100));
                }
                println!("  tomet: {}", shorten(real, 420));
                println!("  ours:  {}", shorten(ours, 420));
            }
        }
    }
    Ok(())
}
