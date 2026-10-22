# phase4-zero-copy-binary-serialization

Goal: Provide zero-copy binary disk serialization and memory-mapped (`mmap`) instant loading for `FlatTree`, eliminating re-parsing overhead for recurring queries and gigabyte-scale datasets.

## Motivation

`FlatNode` was deliberately designed as a 16-byte fixed-size POD struct (`#[repr(C)]`, `depth`, `kind`, `span_start`, `span_len`) with exact binary compatibility with GPU storage buffers.
By persisting parsed documents into a canonical binary container (`.olb` - oneline binary):
1. Loading an arbitrarily large document (even 10+ GB) takes **< 1 microsecond** via `memmap2`.
2. GPU VRAM buffers can be populated via direct DMA from mmap memory without CPU parsing.
3. Cold-start query latency drops from tens of milliseconds to sub-millisecond.

## Binary Layout Specification (`.olb`)

Tightly packed, 64-byte aligned binary format:

```
[0x00..0x10] Header (16 bytes):
  - 0x00..0x08: Magic bytes b"ONELINE\0"
  - 0x08..0x0C: Format version (u32, = 1)
  - 0x0C..0x10: Flags (u32)

[0x10..0x20] Metadata (16 bytes):
  - 0x10..0x18: text_len (u64)
  - 0x18..0x20: line_count (u64)

[0x20..0x40] Section Offsets (32 bytes):
  - 0x20..0x28: text_offset (u64)
  - 0x28..0x30: nodes_offset (u64)
  - 0x30..0x38: line_offsets_offset (u64)
  - 0x38..0x40: crc32_checksum (u64)

[text_offset..]:         Raw UTF-8 text bytes
[nodes_offset..]:        FlatNode POD array (line_count * 16 bytes)
[line_offsets_offset..]: u32 line offset array (line_count * 4 bytes)
```

## Architecture

Add serialization methods in `crates/oneline-core`:
- `FlatTree::save_to_file(&self, path: impl AsRef<Path>) -> Result<()>`
- `FlatTree::open_mmap(path: impl AsRef<Path>) -> Result<MmapFlatTree>` (zero-copy view)
- CLI support:
  - `oneline compile <file.ol> -o <file.olb>`
  - `oneline query <file.olb>` (instant mmap query without parsing)

## Steps

- [ ] Add `memmap2` to workspace dependencies.
- [ ] Implement `.olb` header encoding, validation, and serialization in `oneline-core`.
- [ ] Implement `MmapFlatTree` providing zero-copy access to `&[FlatNode]` and `&[u32]`.
- [ ] Add `compile` subcommand to `app/src/commands/compile.rs`.
- [ ] Add support for transparently loading `.olb` files in `oneline query` and `oneline parse`.
- [ ] Write unit and integration tests verifying round-trip fidelity between text and `.olb`.
- [ ] Benchmark cold load time: `.ol` text parse vs `.olb` mmap load.

## Note on the "< 1 microsecond" claim

`mmap` itself is lazy, so mapping a 10 GB file does take microseconds, but the first query then pays
the page faults for everything it touches. Benchmark cold (page cache dropped) and warm separately and
report time-to-first-result, not just time-to-map. `FlatTree` now borrows its source text, so an
`MmapFlatTree` can hand out the same type without copying.
