//! Substring search over raw text: the CPU baseline for the GPU search kernel.
//!
//! Semantics match the GPU kernel exactly: the byte offset of **every**
//! occurrence, overlapping ones included, in ascending order.
//!
//! The engine is `memchr::memmem` (the SIMD searcher behind ripgrep), run on all
//! cores over overlapping chunks. It is deliberately a strong baseline: a GPU win
//! against a naive loop would say nothing.

use memchr::memmem;
use oneline_core::{InputTooLarge, check_input_len};
use rayon::prelude::*;

/// Chunks smaller than this are not worth a task.
const MIN_CHUNK: usize = 1 << 20;

/// All occurrences of `pattern` in `text`, single-threaded.
///
/// # Panics
/// If `text` is larger than 4 GiB (offsets are `u32`).
pub fn find_all(text: &[u8], pattern: &[u8]) -> Vec<u32> {
    check_input_len(text.len()).expect("find_all");
    let mut out = Vec::new();
    push_matches(text, pattern, 0, &mut out);
    out
}

/// All occurrences of `pattern` in `text`, using all cores.
pub fn try_find_all_par(text: &[u8], pattern: &[u8]) -> Result<Vec<u32>, InputTooLarge> {
    check_input_len(text.len())?;
    if pattern.is_empty() || text.len() < pattern.len() {
        return Ok(Vec::new());
    }
    let chunk = (text.len() / (rayon::current_num_threads() * 4)).max(MIN_CHUNK);
    if text.len() <= chunk {
        return Ok(find_all(text, pattern));
    }
    let starts: Vec<usize> = (0..text.len()).step_by(chunk).collect();
    Ok(starts
        .par_iter()
        .flat_map_iter(|&start| {
            // A match may begin anywhere in `start..end` and run `pattern.len() - 1`
            // bytes past `end`; extend the window, but only count starts below `end`
            // (a start at `end` or later cannot fit in the extended window).
            let end = (start + chunk).min(text.len());
            let window_end = (end + pattern.len() - 1).min(text.len());
            let mut out = Vec::new();
            push_matches(&text[start..window_end], pattern, start, &mut out);
            out
        })
        .collect())
}

/// Like [`try_find_all_par`].
///
/// # Panics
/// If `text` is larger than 4 GiB.
pub fn find_all_par(text: &[u8], pattern: &[u8]) -> Vec<u32> {
    try_find_all_par(text, pattern).expect("find_all_par")
}


/// Number of (overlapping) occurrences of `pattern` in `text`, using all cores,
/// without materializing the offsets. The CPU counterpart of a GPU count-only search.
pub fn count_all_par(text: &[u8], pattern: &[u8]) -> usize {
    if pattern.is_empty() || text.len() < pattern.len() {
        return 0;
    }
    let chunk = (text.len() / (rayon::current_num_threads() * 4)).max(MIN_CHUNK);
    (0..text.len())
        .step_by(chunk)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&start| {
            let end = (start + chunk).min(text.len());
            let window = &text[start..(end + pattern.len() - 1).min(text.len())];
            let finder = memmem::Finder::new(pattern);
            let (mut pos, mut n) = (0, 0usize);
            while let Some(i) = finder.find(&window[pos..]) {
                n += 1;
                pos += i + 1;
            }
            n
        })
        .sum()
}

/// Appends `base + i` for every (overlapping) occurrence at `i` in `haystack`.
fn push_matches(haystack: &[u8], pattern: &[u8], base: usize, out: &mut Vec<u32>) {
    if pattern.is_empty() {
        return;
    }
    let finder = memmem::Finder::new(pattern);
    let mut pos = 0;
    // `find_iter` skips overlapping matches; restart one byte after each hit.
    while let Some(i) = finder.find(&haystack[pos..]) {
        out.push((base + pos + i) as u32);
        pos += i + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oneline_core::datagen::{self, Profile, Rng};

    fn naive(text: &[u8], pattern: &[u8]) -> Vec<u32> {
        if pattern.is_empty() || text.len() < pattern.len() {
            return Vec::new();
        }
        (0..=text.len() - pattern.len())
            .filter(|&i| &text[i..i + pattern.len()] == pattern)
            .map(|i| i as u32)
            .collect()
    }

    #[test]
    fn count_matches_find_all() {
        let text = datagen::generate(Profile::Prose, 3 << 20, 4);
        for p in ["a", "ab", "the", "xyzzy-absent", "\n|"] {
            assert_eq!(count_all_par(text.as_bytes(), p.as_bytes()), find_all(text.as_bytes(), p.as_bytes()).len(), "{p:?}");
        }
        assert_eq!(count_all_par(b"aaaa", b"aa"), 3);
    }

    #[test]
    fn overlapping_and_edges() {
        assert_eq!(find_all(b"aaaa", b"aa"), vec![0, 1, 2]);
        assert_eq!(find_all(b"abcabc", b"abc"), vec![0, 3]);
        assert_eq!(find_all(b"abc", b"abcd"), Vec::<u32>::new());
        assert_eq!(find_all(b"", b"a"), Vec::<u32>::new());
        assert_eq!(find_all(b"xabc", b"abc"), vec![1]); // match flush with the end
        assert_eq!(find_all(b"a\nb", b"a\nb"), vec![0]); // patterns may contain newlines
    }

    #[test]
    fn parallel_matches_naive_on_all_chunk_boundaries() {
        // Patterns that straddle every 1 MiB chunk boundary: dense, overlapping hits.
        let text: Vec<u8> = b"ab".iter().copied().cycle().take(5 * MIN_CHUNK + 17).collect();
        for pattern in [&b"ab"[..], b"aba", b"bab", b"ababab", b"x"] {
            assert_eq!(find_all_par(&text, pattern), naive(&text, pattern), "pattern {pattern:?}");
        }
    }

    #[test]
    fn parallel_matches_naive_on_generated_documents() {
        let mut rng = Rng::new(31);
        for profile in Profile::ALL {
            let text = datagen::generate(profile, 3 << 20, 5);
            let bytes = text.as_bytes();
            let mut patterns: Vec<Vec<u8>> =
                ["lorem", "SIMD GPU", "naïve", "|", "||", "\n|", "階層 解析", "- "]
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect();
            // Random slices of the text itself: guaranteed to occur at least once.
            for _ in 0..4 {
                let len = rng.range(1, 65) as usize;
                let at = rng.below((bytes.len() - len) as u64) as usize;
                patterns.push(bytes[at..at + len].to_vec());
            }
            for p in patterns {
                assert_eq!(find_all_par(bytes, &p), naive(bytes, &p), "{profile:?} {p:?}");
            }
        }
    }
}

// ---- many patterns ----------------------------------------------------------

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};

/// One occurrence of one of several patterns: which pattern, and where it starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternHit {
    pub pattern: u32,
    pub offset: u32,
}

/// The CPU baseline for dictionary search: `aho-corasick` (the crate behind ripgrep's
/// multi-pattern search), as a DFA, run on all cores over overlapping chunks.
pub struct ManyPatterns {
    automaton: AhoCorasick,
    longest: usize,
}

impl ManyPatterns {
    /// # Panics
    /// If `patterns` is empty or contains an empty pattern.
    pub fn new<P: AsRef<[u8]>>(patterns: &[P]) -> Self {
        assert!(!patterns.is_empty() && patterns.iter().all(|p| !p.as_ref().is_empty()), "need non-empty patterns");
        let automaton = AhoCorasick::builder()
            // Report every occurrence of every pattern, including overlapping ones, as the GPU does.
            .match_kind(MatchKind::Standard)
            .kind(Some(AhoCorasickKind::DFA))
            .build(patterns)
            .expect("aho-corasick build");
        let longest = patterns.iter().map(|p| p.as_ref().len()).max().unwrap();
        Self { automaton, longest }
    }

    /// Number of occurrences of each pattern, all cores.
    pub fn count_par(&self, text: &[u8], pattern_count: usize) -> Vec<u64> {
        let chunk = (text.len() / (rayon::current_num_threads() * 4)).max(MIN_CHUNK);
        (0..text.len())
            .step_by(chunk)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&start| {
                let end = (start + chunk).min(text.len());
                let window = &text[start..(end + self.longest - 1).min(text.len())];
                let mut counts = vec![0u64; pattern_count];
                for m in self.automaton.find_overlapping_iter(window) {
                    // A match must start inside this chunk; the extension past `end` only
                    // exists so matches that begin before `end` can finish.
                    if m.start() < end - start {
                        counts[m.pattern().as_usize()] += 1;
                    }
                }
                counts
            })
            .reduce(
                || vec![0u64; pattern_count],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    }

    /// Every occurrence, sorted by `(pattern, offset)`, all cores.
    pub fn find_all_par(&self, text: &[u8]) -> Vec<PatternHit> {
        let chunk = (text.len() / (rayon::current_num_threads() * 4)).max(MIN_CHUNK);
        let mut hits: Vec<PatternHit> = (0..text.len())
            .step_by(chunk)
            .collect::<Vec<_>>()
            .par_iter()
            .flat_map_iter(|&start| {
                let end = (start + chunk).min(text.len());
                let window = &text[start..(end + self.longest - 1).min(text.len())];
                let mut out = Vec::new();
                for m in self.automaton.find_overlapping_iter(window) {
                    if m.start() < end - start {
                        out.push(PatternHit { pattern: m.pattern().as_u32(), offset: (start + m.start()) as u32 });
                    }
                }
                out
            })
            .collect();
        hits.par_sort_unstable();
        hits
    }
}

#[cfg(test)]
mod many_tests {
    use super::*;
    use oneline_core::datagen::{self, Profile};

    #[test]
    fn many_patterns_match_single_pattern_search() {
        let text = datagen::generate(Profile::Prose, 3 << 20, 8);
        let v = datagen::prose_vocabulary();
        let patterns: Vec<Vec<u8>> = [0usize, 1, 5, 200, 3500, 4000].iter().map(|&i| v[i].as_bytes().to_vec()).chain([b"|".to_vec(), b"- ".to_vec()]).collect();
        let many = ManyPatterns::new(&patterns);
        let hits = many.find_all_par(text.as_bytes());
        let counts = many.count_par(text.as_bytes(), patterns.len());
        for (i, p) in patterns.iter().enumerate() {
            let want = find_all(text.as_bytes(), p);
            let got: Vec<u32> = hits.iter().filter(|h| h.pattern as usize == i).map(|h| h.offset).collect();
            assert_eq!(got, want, "pattern {i} ({:?})", String::from_utf8_lossy(p));
            assert_eq!(counts[i] as usize, want.len());
        }
    }
}
