mod bench;
mod parse;
mod query;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
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
    /// Query document structure on the GPU without downloading the full tree
    Query {
        /// Path to input file (optional if --bench-mb is given)
        file: Option<PathBuf>,

        /// Filter by node kind (heading, element, list, callout, text, raw)
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
}

impl From<DataProfile> for Profile {
    fn from(p: DataProfile) -> Self {
        match p {
            DataProfile::Repeat => Profile::Repeat,
            DataProfile::ShortLines => Profile::ShortLines,
            DataProfile::Mixed => Profile::Mixed,
            DataProfile::LongLines => Profile::LongLines,
        }
    }
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Parse { file, engine, to_ast } => parse::run_parse(&file, engine, to_ast),
        Commands::Bench { file, synthetic_mb, profile, iterations } => {
            bench::run_bench(file.as_deref(), synthetic_mb, profile.into(), iterations)
        }
        Commands::Query { file, kind, depth, bench_mb, profile } => {
            query::run_query(file.as_deref(), kind.as_deref(), depth, bench_mb, profile.into())
        }
    }
}

/// Average wall-clock time of `f` over `iterations` runs, after one warm-up.
pub(crate) fn time_avg<R>(iterations: usize, mut f: impl FnMut() -> Result<R>) -> Result<Duration> {
    f()?;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(f()?);
    }
    Ok(start.elapsed() / iterations.max(1) as u32)
}

pub(crate) fn report(label: &str, bytes: usize, avg: Duration) {
    let gbps = (bytes as f64 / 1e9) / avg.as_secs_f64();
    println!("  {label:<46} {avg:>10.2?}  {gbps:>7.2} GB/s");
}
