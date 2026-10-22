//! Deterministic synthetic document generators for tests and benchmarks.
//!
//! The old benchmark input was one six-line sample repeated; that makes every
//! branch perfectly predictable and every line ~14 bytes. These profiles vary
//! line length, nesting, line kinds, line endings and multi-byte text, so
//! throughput numbers are not an artifact of a repeating pattern.

/// Shape of the generated document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// The legacy six-line sample repeated (best case for branch predictors).
    Repeat,
    /// Very short lines (4..24 chars): worst case for per-line overhead.
    ShortLines,
    /// Mixed kinds/depths with a heavy-tailed line length (typical prose + markup).
    Mixed,
    /// Long lines (200..1500 chars): worst case for per-byte overhead.
    LongLines,
}

impl Profile {
    pub const ALL: [Profile; 4] = [Self::Repeat, Self::ShortLines, Self::Mixed, Self::LongLines];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Repeat => "repeat",
            Self::ShortLines => "short-lines",
            Self::Mixed => "mixed",
            Self::LongLines => "long-lines",
        }
    }
}

/// xorshift64* — tiny, dependency-free, reproducible.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 11) % n
    }

    /// Uniform in `lo..hi`.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo)
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

const WORDS: &[&str] = &[
    "lorem", "ipsum", "dolor", "sit", "amet", "parse", "line", "node", "depth", "tomet", "SIMD",
    "GPU", "縦棒", "階層", "構文", "解析", "naïve", "café", "データ", "|pipe|", "a", "of", "the",
];

fn push_text(out: &mut String, rng: &mut Rng, target_len: usize) {
    let start = out.len();
    while out.len() - start < target_len {
        if out.len() > start {
            out.push(' ');
        }
        out.push_str(WORDS[rng.below(WORDS.len() as u64) as usize]);
    }
}

fn push_prefix(out: &mut String, depth: u32) {
    for _ in 0..depth {
        out.push('|');
    }
    if depth > 0 {
        out.push(' ');
    }
}

fn eol(out: &mut String, rng: &mut Rng, crlf_percent: u64) {
    if rng.chance(crlf_percent) {
        out.push('\r');
    }
    out.push('\n');
}

/// Generates roughly `target_bytes` of text (always ending on a line boundary).
pub fn generate(profile: Profile, target_bytes: usize, seed: u64) -> String {
    let mut out = String::with_capacity(target_bytes + 4096);
    let mut rng = Rng::new(seed);

    if profile == Profile::Repeat {
        let sample = "|>[ Header ]\n| Is this syntax necessary?\n||>[ Callout ]\n|| Content.\n|||- Item A\n|||- Item B\n";
        while out.len() < target_bytes {
            out.push_str(sample);
        }
        return out;
    }

    let mut depth = 1u32;
    while out.len() < target_bytes {
        // Depth follows a random walk, so subtrees open and close realistically.
        match rng.below(10) {
            0..=1 if depth > 0 => depth -= 1,
            2..=3 if depth < 7 => depth += 1,
            4 if rng.chance(2) => depth = rng.below(3) as u32,
            _ => {}
        }

        let body_len = match profile {
            Profile::ShortLines => rng.range(4, 24),
            Profile::LongLines => rng.range(200, 1500),
            // Heavy tail: 1% of lines are very long.
            _ if rng.chance(1) => rng.range(800, 4000),
            _ => rng.range(5, 120),
        } as usize;

        let crlf = if profile == Profile::Mixed { 2 } else { 0 };
        let kind_roll = rng.below(100);
        match kind_roll {
            0..=4 => {
                // Blank (sometimes with a bare prefix).
                if rng.chance(30) {
                    push_prefix(&mut out, depth);
                }
            }
            5..=9 => {
                push_prefix(&mut out, depth);
                out.push_str("=[ ");
                push_text(&mut out, &mut rng, body_len.min(60));
                out.push_str(" ]");
            }
            10..=14 => {
                push_prefix(&mut out, depth);
                out.push_str(">[ ");
                push_text(&mut out, &mut rng, body_len.min(60));
                out.push_str(" ]");
            }
            15..=22 => {
                push_prefix(&mut out, depth);
                out.push_str("@card(title: \"");
                push_text(&mut out, &mut rng, body_len.min(30));
                out.push_str("\")");
            }
            23..=37 => {
                push_prefix(&mut out, depth);
                if rng.chance(70) {
                    out.push_str("- ");
                } else {
                    out.push_str("1. ");
                }
                push_text(&mut out, &mut rng, body_len);
            }
            38..=39 => {
                push_prefix(&mut out, depth);
                out.push_str("```");
            }
            _ => {
                push_prefix(&mut out, depth);
                push_text(&mut out, &mut rng, body_len);
            }
        }
        eol(&mut out, &mut rng, crlf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_sized() {
        for p in Profile::ALL {
            let a = generate(p, 100_000, 7);
            let b = generate(p, 100_000, 7);
            assert_eq!(a, b, "{p:?} must be reproducible");
            assert!(a.len() >= 100_000 && a.ends_with('\n'));
        }
    }
}
