use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(name = "oneline", author, version, about = "High-performance SIMD/GPU parser for oneline format")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse specified file and display tree structure
    Parse {
        /// Path to input file
        file: PathBuf,

        /// Parse engine (simd or gpu)
        #[arg(short, long, default_value = "simd")]
        engine: Engine,

        /// Convert to canonical tomet-ast Document and display
        #[arg(long)]
        to_ast: bool,
    },
    /// Benchmark parsing throughput and latency (file or synthetic dataset)
    Bench {
        /// Path to input file (defaults to synthetic dataset if omitted)
        file: Option<PathBuf>,

        /// Size of synthetic dataset in megabytes
        #[arg(long, default_value_t = 10)]
        synthetic_mb: usize,

        /// Number of iterations
        #[arg(short, long, default_value_t = 20)]
        iterations: usize,
    },
    /// Query document structure entirely on GPU without downloading full tree
    Query {
        /// Path to input file (optional if --bench-mb is given)
        file: Option<PathBuf>,

        /// Filter by node kind (heading, element, list, callout, text)
        #[arg(short, long)]
        kind: Option<String>,

        /// Filter by nesting depth
        #[arg(short, long)]
        depth: Option<u32>,

        /// Run GPU-resident query benchmark on synthetic dataset of given size in MB
        #[arg(long)]
        bench_mb: Option<usize>,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Simd,
    SimdPar,
    Gpu,
}

fn ensure_nixos_vulkan_paths() {
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let vulkan_driver_path = "/run/opengl-driver/lib";
    let nix_vulkan_loader = "/nix/store/lk4614327cn1h9z3m6cvnls95n13m1cj-vulkan-loader-1.4.357.0/lib";

    if !current.contains(vulkan_driver_path) || !current.contains(nix_vulkan_loader) {
        let new_path = format!("{}:{}:{}", current, vulkan_driver_path, nix_vulkan_loader);
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH", new_path);
        }
    }
}

fn main() -> Result<()> {
    ensure_nixos_vulkan_paths();
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { file, engine, to_ast } => {
            let content = std::fs::read_to_string(&file)
                .with_context(|| format!("Failed to read file: {}", file.display()))?;

            println!("==> Parsing {} ({} bytes) using {:?} engine...", file.display(), content.len(), engine);

            let tree = match engine {
                Engine::Simd => oneline_simd::parse(&content),
                Engine::SimdPar => oneline_simd::parse_par(&content),
                Engine::Gpu => {
                    let parser = oneline_gpu::GpuParser::new()
                        .context("Failed to initialize GPU parser")?;
                    parser.parse_raw(&content)?
                }
            };

            if to_ast {
                let ast = oneline_tomet::to_tomet_ast(&tree)
                    .context("Failed to convert FlatTree to tomet AST")?;
                println!("\nTomet AST Document ({} top-level blocks):", ast.blocks.len());
                for (i, block) in ast.blocks.iter().enumerate() {
                    println!("  [{}] {:#?}", i, block);
                }
            } else {
                println!("\nParsed Tree ({} lines):", tree.nodes.len());
                print!("{}", tree.dump_tree());
            }
        }
        Commands::Bench { file, synthetic_mb, iterations } => {
            let (name, content) = match file {
                Some(p) => {
                    let s = std::fs::read_to_string(&p)
                        .with_context(|| format!("Failed to read file: {}", p.display()))?;
                    (p.display().to_string(), s)
                }
                None => {
                    println!("==> Generating synthetic oneline dataset ({} MB)...", synthetic_mb);
                    let sample = "|>[ Header ]\n| Is this syntax necessary?\n||>[ Callout ]\n|| Content.\n|||- Item A\n|||- Item B\n";
                    let target_bytes = synthetic_mb * 1024 * 1024;
                    let repeats = target_bytes / sample.len() + 1;
                    let mut s = String::with_capacity(repeats * sample.len());
                    for _ in 0..repeats {
                        s.push_str(sample);
                    }
                    (format!("Synthetic {}MB", synthetic_mb), s)
                }
            };

            let bytes = content.as_bytes();
            let total_bytes = bytes.len() as f64;
            let mb_size = total_bytes / (1024.0 * 1024.0);

            println!("==> Benchmarking {} ({:.2} MB, {} iterations)\n", name, mb_size, iterations);

            // 1. CPU SIMD Benchmark (Single-core)
            {
                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = oneline_simd::parse(&content);
                }
                let elapsed = start.elapsed();
                let avg_time = elapsed / iterations as u32;
                let throughput_gbps = (total_bytes / 1_000_000_000.0) / avg_time.as_secs_f64();
                println!("  [CPU SIMD (Single-core AVX2)]");
                println!("    Average Latency: {:?}", avg_time);
                println!("    Throughput:      {:.2} GB/s", throughput_gbps);
            }

            // 2. CPU SIMD Benchmark (Multi-core Rayon)
            {
                // Warmup
                let _ = oneline_simd::parse_par(&content);

                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = oneline_simd::parse_par(&content);
                }
                let elapsed = start.elapsed();
                let avg_time = elapsed / iterations as u32;
                let throughput_gbps = (total_bytes / 1_000_000_000.0) / avg_time.as_secs_f64();
                println!("\n  [CPU SIMD (Multi-core Rayon + AVX2, {} threads)]", rayon::current_num_threads());
                println!("    Average Latency: {:?}", avg_time);
                println!("    Throughput:      {:.2} GB/s", throughput_gbps);
            }

            // 2. GPU Full End-to-End Benchmark (Prefix Scan + Line Parse)
            match oneline_gpu::GpuParser::new() {
                Ok(parser) => {
                    // Warmup
                    let _ = parser.parse_raw(&content)?;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        let _ = parser.parse_raw(&content)?;
                    }
                    let elapsed = start.elapsed();
                    let avg_time = elapsed / iterations as u32;
                    let throughput_gbps = (total_bytes / 1_000_000_000.0) / avg_time.as_secs_f64();
                    println!("\n  [wgpu GPU Full Pipeline (Prefix Scan + Line Parse)]");
                    println!("    Average Latency: {:?}", avg_time);
                    println!("    Throughput:      {:.2} GB/s", throughput_gbps);

                    // Also show pure kernel time with precomputed offsets
                    let offsets = oneline_simd::scan_line_offsets(bytes);
                    let start_k = Instant::now();
                    for _ in 0..iterations {
                        let _ = parser.parse(&content, &offsets)?;
                    }
                    let elapsed_k = start_k.elapsed();
                    let avg_k = elapsed_k / iterations as u32;
                    let throughput_k = (total_bytes / 1_000_000_000.0) / avg_k.as_secs_f64();
                    println!("\n  [wgpu GPU Kernel (Precomputed Offsets)]");
                    println!("    Average Latency: {:?}", avg_k);
                    println!("    Throughput:      {:.2} GB/s", throughput_k);
                }
                Err(e) => {
                    eprintln!("\n  [wgpu Compute (GPU)]: Skipped (GPU unavailable: {:?})", e);
                }
            }
        }
        Commands::Query { file, kind, depth, bench_mb } => {
            let node_kind = kind.as_deref().and_then(|k| match k.to_lowercase().as_str() {
                "heading" => Some(oneline_core::NodeKind::Heading),
                "element" => Some(oneline_core::NodeKind::Element),
                "list" => Some(oneline_core::NodeKind::List),
                "callout" => Some(oneline_core::NodeKind::Callout),
                "text" => Some(oneline_core::NodeKind::Text),
                "raw" => Some(oneline_core::NodeKind::Raw),
                _ => None,
            });

            let parser = oneline_gpu::GpuParser::new()
                .context("Failed to initialize GPU parser for query")?;

            if let Some(mb) = bench_mb {
                println!("==> Benchmarking 100% GPU-Resident Query on Synthetic {} MB dataset...", mb);
                let sample = "|>[ Header ]\n| Regular paragraph text\n||>[ Callout ]\n|| Body text\n|||- List item A\n|||- List item B\n| @card(title: \"test\")\n|| Card content\n";
                let target_bytes = mb * 1024 * 1024;
                let repeats = target_bytes / sample.len() + 1;
                let mut content = String::with_capacity(repeats * sample.len());
                for _ in 0..repeats {
                    content.push_str(sample);
                }
                let total_bytes = content.len() as f64;

                let _ = parser.query_raw(&content, node_kind, depth, 1024)?;

                let iters = 20;
                let start = Instant::now();
                let mut match_count = 0;
                for _ in 0..iters {
                    let matches = parser.query_raw(&content, node_kind, depth, 1024)?;
                    match_count = matches.len();
                }
                let elapsed = start.elapsed();
                let avg_time = elapsed / iters as u32;
                let throughput_gbps = (total_bytes / 1_000_000_000.0) / avg_time.as_secs_f64();

                println!("\n  [100% GPU-Resident Query (Prefix Scan + Parse + Filter in VRAM)]");
                println!("    Filter criteria: kind={:?}, depth={:?}", node_kind, depth);
                println!("    Sample matches:  {} matches found", match_count);
                println!("    Average Latency: {:?}", avg_time);
                println!("    Throughput:      {:.2} GB/s", throughput_gbps);
            } else if let Some(file_path) = file {
                let content = std::fs::read_to_string(&file_path)
                    .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
                let offsets = parser.scan_line_offsets(&content)?;
                let matches = parser.query_raw(&content, node_kind, depth, 100)?;

                println!("==> Found {} matching lines in {} (100% GPU Pipeline):", matches.len(), file_path.display());
                for &idx in &matches {
                    let start = offsets[idx as usize] as usize;
                    let end = if (idx as usize) + 1 < offsets.len() {
                        offsets[(idx as usize) + 1] as usize
                    } else {
                        content.len()
                    };
                    let line_str = content[start..end].trim();
                    println!("  line {:>4}: {}", idx + 1, line_str);
                }
            } else {
                eprintln!("Error: specify either a file or --bench-mb");
            }
        }
    }

    Ok(())
}
