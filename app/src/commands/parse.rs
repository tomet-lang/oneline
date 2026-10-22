use std::path::Path;

use anyhow::{Context, Result};

use super::Engine;

pub fn run_parse(file: &Path, engine: Engine, to_ast: bool) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .with_context(|| format!("Failed to read file: {}", file.display()))?;

    println!("==> Parsing {} ({} bytes) using {:?} engine...", file.display(), content.len(), engine);

    let tree = match engine {
        Engine::Simd => oneline_simd::try_parse(&content)?,
        Engine::SimdPar => oneline_simd::try_parse_par(&content)?,
        Engine::Gpu => {
            let parser = oneline_gpu::GpuParser::new().context("Failed to initialize GPU parser")?;
            parser.parse_raw(&content)?
        }
    };

    if to_ast {
        let ast = oneline_tomet::to_tomet_ast(&tree).context("Failed to convert FlatTree to tomet AST")?;
        println!("\nTomet AST Document ({} top-level blocks):", ast.blocks.len());
        for (i, block) in ast.blocks.iter().enumerate() {
            println!("  [{}] {:#?}", i, block);
        }
    } else {
        println!("\nParsed Tree ({} lines):", tree.nodes.len());
        print!("{}", tree.dump_tree());
    }
    Ok(())
}
