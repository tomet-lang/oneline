use anyhow::{Result, anyhow};

use crate::document::GpuDocument;
use crate::parser::GpuParser;
use crate::timing::{Profile, StageTimer};
use crate::util::{bind, check_gpu_len, dispatch_dims};
use crate::{MAX_PATTERN_LEN, SearchConfig};

/// Must equal `WORDS_PER_THREAD` in `search.wgsl`.
const WORDS_PER_THREAD: u32 = 16;

/// Result of a substring search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Number of occurrences (may exceed `offsets.len()`).
    pub total_matches: u32,
    /// Byte offsets of occurrences in ascending order, overlapping occurrences
    /// included. If there are more than `max_matches`, this is an arbitrary subset
    /// of size `max_matches` (slots are claimed with atomics).
    pub offsets: Vec<u32>,
}

impl GpuParser {
    pub(crate) fn text_bind_group(&self, text: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Search Text Bind Group"),
            layout: &self.text_layout,
            entries: &[bind(0, text)],
        })
    }

    fn run_search(
        &self,
        text_group: &wgpu::BindGroup,
        byte_freq: &[u32; 256],
        text_len: u32,
        pattern: &[u8],
        max_matches: usize,
        timer: Option<&StageTimer>,
    ) -> Result<SearchResult> {
        if pattern.is_empty() || pattern.len() > MAX_PATTERN_LEN {
            return Err(anyhow!(
                "pattern must be 1..={MAX_PATTERN_LEN} bytes, got {}",
                pattern.len()
            ));
        }
        let max_u32 = u32::try_from(max_matches)
            .map_err(|_| anyhow!("max_matches {max_matches} does not fit in u32"))?;

        let mut packed = [0u8; MAX_PATTERN_LEN];
        packed[..pattern.len()].copy_from_slice(pattern);
        let cfg = SearchConfig {
            text_len,
            pattern_len: pattern.len() as u32,
            max_matches: max_u32,
            window: choose_window(pattern, byte_freq),
            pattern: bytemuck::cast(packed),
        };

        // One workgroup per 256 * WORDS_PER_THREAD words.
        let dims = dispatch_dims(text_len.div_ceil(4).div_ceil(WORDS_PER_THREAD * 256));
        self.with_scratch(max_u32, timer, |s| {
            self.queue
                .write_buffer(&s.search_config, 0, bytemuck::bytes_of(&cfg));
            let (total_matches, offsets) = self.read_counted(
                s,
                &self.search_pipeline,
                text_group,
                &s.search_group1,
                dims,
                max_u32,
                "search",
                timer,
            )?;
            Ok(SearchResult {
                total_matches,
                offsets,
            })
        })
    }

    /// Searches a resident document for `pattern` (1..=64 bytes), returning the
    /// byte offset of every occurrence. Only the count and the offsets cross PCIe.
    pub fn search_document(
        &self,
        doc: &GpuDocument,
        pattern: &[u8],
        max_matches: usize,
    ) -> Result<SearchResult> {
        match doc.text_group() {
            None => Ok(SearchResult {
                total_matches: 0,
                offsets: Vec::new(),
            }),
            Some(group) => self.run_search(
                group,
                doc.byte_freq(),
                doc.text_len() as u32,
                pattern,
                max_matches,
                None,
            ),
        }
    }

    /// One-shot search: uploads the text, then searches it. No line scan is needed.
    pub fn search_raw(
        &self,
        input: &str,
        pattern: &[u8],
        max_matches: usize,
    ) -> Result<SearchResult> {
        self.search_raw_inner(input, pattern, max_matches, None)
    }

    fn search_raw_inner(
        &self,
        input: &str,
        pattern: &[u8],
        max_matches: usize,
        timer: Option<&StageTimer>,
    ) -> Result<SearchResult> {
        check_gpu_len(input.len())?;
        if input.is_empty() {
            return Ok(SearchResult {
                total_matches: 0,
                offsets: Vec::new(),
            });
        }
        let text = self.upload_text(input, timer)?;
        let group = self.text_bind_group(&text);
        self.run_search(
            &group,
            &sample_byte_histogram(input.as_bytes()),
            input.len() as u32,
            pattern,
            max_matches,
            timer,
        )
    }

    /// `search_document`, with a breakdown of where the time went.
    pub fn profile_search_document(
        &self,
        doc: &GpuDocument,
        pattern: &[u8],
        max_matches: usize,
    ) -> Result<Profile> {
        let timer = StageTimer::new(self);
        if let Some(group) = doc.text_group() {
            self.run_search(
                group,
                doc.byte_freq(),
                doc.text_len() as u32,
                pattern,
                max_matches,
                Some(&timer),
            )?;
        }
        timer.finish(self)
    }

    /// `search_raw`, with a breakdown of where the time went.
    pub fn profile_search_raw(
        &self,
        input: &str,
        pattern: &[u8],
        max_matches: usize,
    ) -> Result<Profile> {
        let timer = StageTimer::new(self);
        self.search_raw_inner(input, pattern, max_matches, Some(&timer))?;
        timer.finish(self)
    }
}

/// Bytes of text sampled to estimate byte frequencies (spread over the document).
const SAMPLE_BYTES: usize = 1 << 20;
const SAMPLE_WINDOWS: usize = 4;

/// Byte histogram of a sample of `text`: all of it if small, otherwise a few evenly
/// spaced windows. Used to find each pattern's rarest byte; ~1 ms for a 1 MiB
/// sample, against ~20 ms to upload 100 MB.
pub(crate) fn sample_byte_histogram(text: &[u8]) -> Box<[u32; 256]> {
    let mut freq = Box::new([0u32; 256]);
    let mut count = |bytes: &[u8]| {
        for &b in bytes {
            freq[b as usize] += 1;
        }
    };
    if text.len() <= SAMPLE_BYTES {
        count(text);
    } else {
        let window = SAMPLE_BYTES / SAMPLE_WINDOWS;
        let span = text.len() - window;
        for i in 0..SAMPLE_WINDOWS {
            let start = span / (SAMPLE_WINDOWS - 1) * i;
            count(&text[start..start + window]);
        }
    }
    freq
}

/// Index of the 4-byte pattern window (the whole pattern if it is shorter than 4
/// bytes) whose bytes are rarest according to `freq`. Ties go to the earlier window.
///
/// The kernel filters candidates by comparing this window against the text. On SIMT
/// hardware a warp pays for any lane that takes the slow verify path, so the window
/// has to be selective on the actual data, not merely at the start of the pattern.
pub(crate) fn choose_window(pattern: &[u8], freq: &[u32; 256]) -> u32 {
    let n = pattern.len().min(4);
    // Sum of log-frequencies = log of the product: how likely a random position matches.
    let score = |j: usize| -> f64 {
        pattern[j..j + n]
            .iter()
            .map(|&b| f64::from(freq[b as usize] + 1).ln())
            .sum()
    };
    (0..=pattern.len() - n)
        .min_by(|&a, &b| score(a).total_cmp(&score(b)))
        .expect("pattern is non-empty") as u32
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_the_rarest_four_bytes_earliest_on_ties() {
        let mut freq = [100u32; 256];
        freq[b'q' as usize] = 3;
        // Only one possible window.
        assert_eq!(choose_window(b"abcq", &freq), 0);
        assert_eq!(choose_window(b"ab", &freq), 0);
        // "aaaaq": window 1 ("aaaq") holds the rare byte, window 0 does not.
        assert_eq!(choose_window(b"aaaaq", &freq), 1);
        // The rare byte is at the front: window 0.
        assert_eq!(choose_window(b"qaaaa", &freq), 0);
        // All equal: the earliest.
        assert_eq!(choose_window(b"abcdefgh", &[7; 256]), 0);
        // A byte absent from the sample wins even when it is last.
        freq[b'z' as usize] = 0;
        assert_eq!(choose_window(b"aaaaaaz", &freq), 3);
    }

    #[test]
    fn histogram_samples_large_texts_evenly() {
        let mut text = vec![b'a'; 8 << 20];
        let n = text.len();
        text[n - 10] = b'z';
        let freq = sample_byte_histogram(&text);
        assert_eq!(
            freq.iter().map(|&c| c as usize).sum::<usize>(),
            SAMPLE_BYTES
        );
        assert!(freq[b'a' as usize] as usize >= SAMPLE_BYTES - 1);
    }
}
