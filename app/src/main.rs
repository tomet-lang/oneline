mod commands;

use anyhow::Result;
use clap::Parser;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// NixOS keeps the Vulkan loader and driver outside the default library path.
/// Only extends `LD_LIBRARY_PATH` with directories that actually exist.
fn ensure_nixos_vulkan_paths() {
    let candidates = [
        "/run/opengl-driver/lib",
        "/nix/store/lk4614327cn1h9z3m6cvnls95n13m1cj-vulkan-loader-1.4.357.0/lib",
    ];
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let mut path = current.clone();
    for dir in candidates {
        if std::path::Path::new(dir).is_dir() && !current.contains(dir) {
            path = format!("{path}:{dir}");
        }
    }
    if path != current {
        // SAFETY: called first thing in `main`, before any other thread exists.
        unsafe { std::env::set_var("LD_LIBRARY_PATH", path) };
    }
}

fn main() -> Result<()> {
    ensure_nixos_vulkan_paths();
    commands::run(commands::Cli::parse())
}
