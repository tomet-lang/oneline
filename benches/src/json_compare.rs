//! Same document, two formats: oneline text vs. the equivalent JSON, parsed by simdjson (the C++
//! library, via `simdjson-sys`), `simd-json` (its Rust port) and oneline.
//!
//! ```text
//! cargo run --release -p oneline-benches --features simdjson --bin json_compare -- \
//!     [--mb 256] [--profile mixed] [--iterations 9]
//! cargo run --release -p oneline-benches --features simdjson --bin json_compare -- --json twitter.json
//! ```
//!
//! What is compared, and what is not:
//! * Every parser gets the input already in memory; time is the median of `--iterations` runs after
//!   one warm-up. Inputs that a parser modifies (simd-json unescapes in place) are re-copied
//!   before the clock starts.
//! * The outputs differ: simdjson builds a DOM tape, simd-json a tape of nodes, oneline a flat array
//!   of 16-byte nodes plus line offsets. oneline's grammar is line-local, which is much easier to
//!   parse than JSON; this measures what that buys, not which parser is better.
//! * simdjson validates UTF-8 as part of the parse. oneline takes `&str`, so validation happened
//!   when the text was made; it is timed separately (`simdutf8`, the fast SIMD validator; `std` is shown too) and added in the last rows.

use std::hint::black_box;
use std::time::Instant;

use oneline_core::NodeKind;
use oneline_core::datagen::{self, Profile};
use simdjson_sys::{SIMDJSON_MAXSIZE_BYTES, SIMDJSON_PADDING, dom_ffi};

struct Args {
    mb: usize,
    profile: Profile,
    iterations: usize,
    json_file: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args { mb: 256, profile: Profile::Mixed, iterations: 9, json_file: None };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match a.as_str() {
            "--mb" => args.mb = value("--mb").parse().expect("--mb: number"),
            "--iterations" => args.iterations = value("--iterations").parse().expect("--iterations: number"),
            "--json" => args.json_file = Some(value("--json")),
            "--profile" => {
                let want = value("--profile");
                args.profile = Profile::ALL
                    .into_iter()
                    .find(|p| p.name() == want)
                    .unwrap_or_else(|| panic!("unknown profile {want}; one of {:?}", Profile::ALL.map(Profile::name)));
            }
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

fn kind_name(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Blank => "blank",
        NodeKind::Text => "text",
        NodeKind::Heading => "heading",
        NodeKind::Element => "element",
        NodeKind::List => "list",
        NodeKind::Connect => "connect",
        NodeKind::Unknown => "unknown",
    }
}

fn push_escaped(out: &mut Vec<u8>, text: &str) {
    for &b in text.as_bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0..0x20 => out.extend_from_slice(format!("\\u{b:04x}").as_bytes()),
            _ => out.push(b),
        }
    }
}

/// One JSON object per line, in order: `{"depth":N,"kind":"...","text":"..."}`. The same information
/// oneline's nodes carry (the tree links are derived from depth in both cases).
fn to_json(text: &str) -> Vec<u8> {
    let tree = oneline_simd::parse_par(text);
    let mut out = Vec::with_capacity(text.len() + text.len() / 2 + tree.nodes.len() * 40 + 64);
    out.extend_from_slice(b"[\n");
    for (i, node) in tree.nodes.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(b",\n");
        }
        out.extend_from_slice(b"{\"depth\":");
        out.extend_from_slice(node.depth.to_string().as_bytes());
        out.extend_from_slice(b",\"kind\":\"");
        out.extend_from_slice(kind_name(node.kind()).as_bytes());
        out.extend_from_slice(b"\",\"text\":\"");
        let (s, n) = (node.span_start as usize, node.span_len as usize);
        push_escaped(&mut out, &text[s..s + n]);
        out.extend_from_slice(b"\"}");
    }
    out.extend_from_slice(b"\n]\n");
    out
}

/// Input with `SIMDJSON_PADDING` readable bytes after the document, so simdjson can parse in place.
fn padded(json: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(json.len() + SIMDJSON_PADDING);
    v.extend_from_slice(json);
    v.resize(json.len() + SIMDJSON_PADDING, 0);
    v
}

/// Number of objects on a simdjson tape (walks it: integers occupy two words).
fn count_objects(tape: &[u64]) -> usize {
    let (mut i, mut n) = (0, 0);
    while i < tape.len() {
        match (tape[i] >> 56) as u8 {
            b'{' => n += 1,
            b'l' | b'u' | b'd' => i += 1,
            _ => {}
        }
        i += 1;
    }
    n
}

/// Median wall time in ms. `prep` runs outside the clock before every run.
fn median_ms(iterations: usize, mut prep: impl FnMut(), mut run: impl FnMut() -> usize) -> f64 {
    prep();
    black_box(run());
    let mut times: Vec<f64> = (0..iterations.max(1))
        .map(|_| {
            prep();
            let start = Instant::now();
            black_box(run());
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

struct Row {
    label: String,
    ms: f64,
}

fn print_rows(rows: &[Row], json_bytes: usize, lines: usize, oneline_bytes: Option<usize>) {
    println!("{:<46} {:>9} {:>11} {:>11} {:>12}", "", "ms", "GB/s input", "GB/s json", "M lines/s");
    for r in rows {
        let secs = r.ms / 1e3;
        let own = if r.label.starts_with("oneline") || r.label.starts_with("utf8") {
            oneline_bytes.map_or(String::from("-"), |b| format!("{:.1}", b as f64 / secs / 1e9))
        } else {
            format!("{:.1}", json_bytes as f64 / secs / 1e9)
        };
        println!(
            "{:<46} {:>9.2} {:>11} {:>11.1} {:>12.1}",
            r.label,
            r.ms,
            own,
            json_bytes as f64 / secs / 1e9,
            lines as f64 / secs / 1e6
        );
    }
}

fn run_json_parsers(json: &[u8], iterations: usize, expected_objects: Option<usize>) -> Vec<Row> {
    let mut rows = Vec::new();

    // simdjson (C++), DOM, parsing the padded input in place.
    let buf = padded(json);
    let mut parser = dom_ffi::parser_new(SIMDJSON_MAXSIZE_BYTES);
    let ms = median_ms(iterations, || {}, || {
        // SAFETY: `buf` has SIMDJSON_PADDING initialized bytes after the `json.len()` bytes passed.
        let code = unsafe { dom_ffi::parser_parse(parser.pin_mut(), &buf[..json.len()], false) };
        assert_eq!(code, 0, "simdjson rejected the document");
        0
    });
    // SAFETY: the last parse succeeded and nothing touched the parser since.
    let objects = count_objects(unsafe { dom_ffi::parser_get_tape_view(parser.as_ref().unwrap()) }.tape);
    if let Some(want) = expected_objects {
        assert_eq!(objects, want, "simdjson tape holds a different number of objects than the document has lines");
    }
    rows.push(Row { label: "simdjson 4.6 (C++), DOM, in place".into(), ms });

    // simd-json (Rust port) rewrites its input, so every run gets a fresh copy outside the clock.
    let scratch = std::cell::RefCell::new(vec![0u8; json.len()]);
    let mut buffers = simd_json::Buffers::new(json.len());
    let ms = median_ms(
        iterations,
        || scratch.borrow_mut().copy_from_slice(json),
        || simd_json::to_tape_with_buffers(&mut scratch.borrow_mut(), &mut buffers).expect("simd-json rejected the document").0.len(),
    );
    rows.push(Row { label: "simd-json 0.18 (Rust), tape".into(), ms });
    rows
}

fn main() {
    let args = parse_args();
    println!("threads: {}", rayon::current_num_threads());

    if let Some(path) = &args.json_file {
        let json = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("==> {path}: {:.2} MB of JSON (oneline cannot parse it; JSON parsers only)\n", json.len() as f64 / 1e6);
        let rows = run_json_parsers(&json, args.iterations, None);
        print_rows(&rows, json.len(), 1, None);
        return;
    }

    let text = datagen::generate(args.profile, args.mb << 20, 1);
    let lines = oneline_simd::count_newlines(text.as_bytes()) + usize::from(!text.ends_with('\n'));
    let json = to_json(&text);
    println!(
        "==> {} {} MB: {:.1} MB of oneline text, {lines} lines  ->  {:.1} MB of JSON ({:.2}x)\n",
        args.profile.name(),
        args.mb,
        text.len() as f64 / 1e6,
        json.len() as f64 / 1e6,
        json.len() as f64 / text.len() as f64,
    );

    let mut rows = run_json_parsers(&json, args.iterations, Some(lines));

    let single = median_ms(args.iterations, || {}, || oneline_simd::parse(&text).nodes.len());
    let par = median_ms(args.iterations, || {}, || oneline_simd::parse_par(&text).nodes.len());
    rows.push(Row { label: "oneline, single core".into(), ms: single });
    rows.push(Row { label: format!("oneline, {} threads", rayon::current_num_threads()), ms: par });
    let mut buffers = oneline_simd::ParseBuffers::new();
    rows.push(Row {
        label: format!("oneline, {} threads, reusing buffers", rayon::current_num_threads()),
        ms: median_ms(args.iterations, || {}, || {
            let tree = oneline_simd::try_parse_par_reusing(&text, &mut buffers).unwrap();
            let n = tree.nodes.len();
            buffers.recycle(tree);
            n
        }),
    });
    let utf8_ms = median_ms(args.iterations, || {}, || simdutf8::basic::from_utf8(black_box(text.as_bytes())).map_or(0, str::len));
    rows.push(Row { label: "utf8 validation of the text (simdutf8), alone".into(), ms: utf8_ms });
    let std_utf8_ms = median_ms(args.iterations, || {}, || std::str::from_utf8(black_box(text.as_bytes())).map_or(0, str::len));
    rows.push(Row { label: "utf8 validation of the text (std), alone".into(), ms: std_utf8_ms });
    rows.push(Row { label: "oneline single core + utf8 validation".into(), ms: single + utf8_ms });
    rows.push(Row { label: "oneline threads + utf8 validation (1 thread)".into(), ms: par + utf8_ms });

    print_rows(&rows, json.len(), lines, Some(text.len()));
    println!(
        "\n'GB/s input' is bytes of the format each parser reads; 'GB/s json' divides the JSON size by every row's time,\nso rows are comparable on the same document; 'M lines/s' counts the document's lines."
    );
}
