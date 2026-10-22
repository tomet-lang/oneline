# refactor-cli-commands

Goal: Refactor `app/src/main.rs` by extracting CLI subcommands into a modular `app/src/commands/` module hierarchy.

## Context

`app/src/main.rs` currently implements the CLI argument parsing, runtime environment setup, and the full logic for `Parse`, `Bench`, and `Query` commands in a single large match block.
As new benchmark parameters and query options are added, each command should reside in its own module.

## Target Architecture

```
app/src/
├── main.rs              # CLI entry point, clap Parser struct, ensure_nixos_vulkan_paths, main()
└── commands/
    ├── mod.rs           # Commands enum and dispatch handler
    ├── parse.rs         # Commands::Parse implementation
    ├── bench.rs         # Commands::Bench implementation (SIMD single/multi, GPU pipeline/kernel)
    └── query.rs         # Commands::Query implementation (GPU query benchmark and file filtering)
```

## Steps

- [x] Create `app/src/commands/mod.rs` defining the `Commands` enum and dispatch logic.
- [x] Create `app/src/commands/parse.rs` with `run_parse(file, engine, to_ast)`.
- [x] Create `app/src/commands/bench.rs` with `run_bench(file, synthetic_mb, iterations)`.
- [x] Create `app/src/commands/query.rs` with `run_query(file, kind, depth, bench_mb)`.
- [x] Simplify `app/src/main.rs` to minimal entry point (~40 lines).
- [x] Verify `just parse docs/idea-bar-style.ol gpu`, `just bench-mb 50 5`, and `just query-bench 50` work identically.
- [x] Verify `just test` passes.

## Status: done

`run_bench` / `run_query` also take a `--profile` (dataset shape). `bench` now reports
CPU single/Rayon/tree-links and GPU one-shot vs resident, and warns on software adapters.
