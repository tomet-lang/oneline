mod bench;
mod parse;
mod profile;
mod query;
mod search;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use oneline_core::datagen::Profile;

#[derive(Parser)]
#[command(name = "oneline", author, version, about = "High-performance SIMD/GPU parser for oneline format")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse specified file and display tree structure
    Parse {
        /// Path to input file
        file: PathBuf,

        /// Parse engine
        #[arg(short, long, default_value = "simd")]
        engine: Engine,

        /// Convert to canonical tomet-ast Document and display
        #[arg(long)]
        to_ast: bool,
    },
    /// Benchmark parsing throughput and latency (file or synthetic dataset)
    Bench {
        /// Path to input file (defaults to a synthetic dataset if omitted)
        file: Option<PathBuf>,

        /// Size of synthetic dataset in megabytes
        #[arg(long, default_value_t = 10)]
        synthetic_mb: usize,

        /// Shape of the synthetic dataset
        #[arg(long, default_value = "mixed")]
        profile: DataProfile,

        /// Number of iterations
        #[arg(short, long, default_value_t = 20)]
        iterations: usize,
    },
    /// Substring search over the raw text on the GPU (CPU baseline in --bench-mb mode)
    Search {
        /// Path to input file (optional with --bench-mb)
        file: Option<PathBuf>,

        /// Pattern to look for (1..=64 bytes); repeat for several
        #[arg(short, long)]
        pattern: Vec<String>,

        /// Benchmark GPU vs CPU on a synthetic dataset of the given size in MB
        #[arg(long)]
        bench_mb: Option<usize>,

        /// Shape of the synthetic dataset
        #[arg(long, default_value = "mixed")]
        profile: DataProfile,

        /// Benchmark GPU vs CPU on FILE (patterns are picked from its own words unless given)
        #[arg(long)]
        bench: bool,

        /// Dictionary search benchmark: N sparse words picked from the text, GPU vs CPU
        #[arg(long)]
        dictionary: Option<usize>,

        /// With --dictionary: pick words that are rare in the text (20-1000 estimated hits) instead of 100-20000
        #[arg(long)]
        sparse: bool,

        /// Capacity for returned occurrence offsets
        #[arg(long, default_value_t = 1 << 20)]
        max_matches: usize,

        /// Number of iterations (benchmark mode)
        #[arg(short, long, default_value_t = 20)]
        iterations: usize,
    },
    /// Break down where GPU time goes (per-pass timestamps + host-side steps)
    Profile {
        /// Path to input file (defaults to a synthetic dataset if omitted)
        file: Option<PathBuf>,

        /// Size of synthetic dataset in kilobytes
        #[arg(long, default_value_t = 102400)]
        synthetic_kb: usize,

        /// Shape of the synthetic dataset
        #[arg(long, default_value = "mixed")]
        profile: DataProfile,

        /// Node kind to query (heading, element, list, connect, text)
        #[arg(short, long, default_value = "heading")]
        kind: String,

        /// Capacity for returned matching line indices
        #[arg(long, default_value_t = 1024)]
        max_matches: usize,

        /// Profile a substring search for this pattern instead of the node queries
        #[arg(long)]
        pattern: Option<String>,

        /// Number of measured runs (the median is reported)
        #[arg(short, long, default_value_t = 20)]
        iterations: usize,
    },
    /// Query document structure on the GPU without downloading the full tree
    Query {
        /// Path to input file (optional if --bench-mb is given)
        file: Option<PathBuf>,

        /// Filter by node kind (heading, element, list, connect, text)
        #[arg(short, long)]
        kind: Option<String>,

        /// Filter by nesting depth
        #[arg(short, long)]
        depth: Option<u32>,

        /// Benchmark queries on a synthetic dataset of the given size in MB
        #[arg(long)]
        bench_mb: Option<usize>,

        /// Shape of the synthetic dataset
        #[arg(long, default_value = "mixed")]
        profile: DataProfile,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Engine {
    Simd,
    SimdPar,
    Gpu,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DataProfile {
    Repeat,
    ShortLines,
    Mixed,
    LongLines,
    Prose,
    Nested,
}

impl From<DataProfile> for Profile {
    fn from(p: DataProfile) -> Self {
        match p {
            DataProfile::Repeat => Profile::Repeat,
            DataProfile::ShortLines => Profile::ShortLines,
            DataProfile::Mixed => Profile::Mixed,
            DataProfile::LongLines => Profile::LongLines,
            DataProfile::Prose => Profile::Prose,
            DataProfile::Nested => Profile::Nested,
        }
    }
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Parse { file, engine, to_ast } => parse::run_parse(&file, engine, to_ast),
        Commands::Bench { file, synthetic_mb, profile, iterations } => {
            bench::run_bench(file.as_deref(), synthetic_mb, profile.into(), iterations)
        }
        Commands::Search { file, pattern, bench_mb, profile, bench, dictionary, sparse, max_matches, iterations } => {
            search::run_search(file.as_deref(), &pattern, bench_mb, profile.into(), max_matches, bench, dictionary, sparse, iterations)
        }
        Commands::Profile { file, synthetic_kb, profile, kind, max_matches, pattern, iterations } => {
            let kind = query::parse_kind(&kind).with_context(|| format!("unknown kind {kind:?}"))?;
            profile::run_profile(file.as_deref(), synthetic_kb, profile.into(), Some(kind), iterations, max_matches, pattern.as_deref())
        }
        Commands::Query { file, kind, depth, bench_mb, profile } => {
            query::run_query(file.as_deref(), kind.as_deref(), depth, bench_mb, profile.into())
        }
    }
}

/// Median wall-clock time of `f` over `iterations` runs, after one warm-up. The median,
/// not the mean: one scheduling hiccup (another process, a driver thread) can make a
/// single run several times slower, and a mean would report it.
pub(crate) fn time_median<R>(iterations: usize, mut f: impl FnMut() -> Result<R>) -> Result<Duration> {
    f()?;
    let mut times = Vec::with_capacity(iterations.max(1));
    for _ in 0..iterations.max(1) {
        let start = Instant::now();
        std::hint::black_box(f()?);
        times.push(start.elapsed());
    }
    times.sort();
    Ok(times[times.len() / 2])
}

pub(crate) fn report(label: &str, bytes: usize, avg: Duration) {
    let gbps = (bytes as f64 / 1e9) / avg.as_secs_f64();
    println!("  {label:<46} {avg:>10.2?}  {gbps:>7.2} GB/s");
}

/// Prints the process's peak resident memory (Linux; silent elsewhere). Large-input
/// runs need this: the text, parse results and the driver's staging copies add up.
pub(crate) fn print_peak_rss() {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return };
    if let Some(kb) = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
    {
        println!("\npeak host memory: {:.2} GiB", kb / (1024.0 * 1024.0));
    }
}
