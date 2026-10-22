# Oneline development tasks.

export LD_LIBRARY_PATH := env_var_or_default("LD_LIBRARY_PATH", "") + ":/run/opengl-driver/lib:/nix/store/lk4614327cn1h9z3m6cvnls95n13m1cj-vulkan-loader-1.4.357.0/lib"

default:
    @just --list

build:
    cargo build --workspace

test:
    cargo test --workspace

check:
    cargo check --workspace

bench file="docs/idea-bar-style.ol" iter="500":
    cargo run --release --bin oneline -- bench {{ file }} --iterations {{ iter }}

bench-mb mb="50" iter="5":
    cargo run --release --bin oneline -- bench --synthetic-mb {{ mb }} --iterations {{ iter }}

parse file engine="simd" *args:
    cargo run --bin oneline -- parse {{ file }} --engine {{ engine }} {{ args }}

query file *args:
    cargo run --bin oneline -- query {{ file }} {{ args }}

query-bench mb="50" *args:
    cargo run --release --bin oneline -- query --bench-mb {{ mb }} {{ args }}
