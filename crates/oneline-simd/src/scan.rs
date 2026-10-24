//! Newline scanning primitives (AVX2 with a portable fallback).

/// Counts `\n` bytes.
pub fn count_newlines(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt") {
            // SAFETY: the required CPU features were just detected.
            return unsafe { count_newlines_avx2(bytes) };
        }
    }
    bytes.iter().filter(|&&b| b == b'\n').count()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn count_newlines_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let mut i = 0;
    let mut total = 0usize;
    while i + 32 <= len {
        // SAFETY: `i + 32 <= len`.
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i) };
        let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, nl)) as u32;
        total += mask.count_ones() as usize;
        i += 32;
    }
    total + bytes[i..].iter().filter(|&&b| b == b'\n').count()
}

/// Calls `f(pos)` for the position of every `\n` in `bytes`, in ascending order.
#[inline]
pub fn for_each_newline(bytes: &[u8], mut f: impl FnMut(usize)) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            unsafe { for_each_newline_avx2(bytes, &mut f) };
            return;
        }
    }
    for_each_newline_scalar(bytes, &mut f);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn for_each_newline_avx2<F: FnMut(usize)>(bytes: &[u8], f: &mut F) {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let mut i = 0;
    while i + 32 <= len {
        // SAFETY: `i + 32 <= len`.
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i) };
        let mut mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, nl)) as u32;
        while mask != 0 {
            f(i + mask.trailing_zeros() as usize);
            mask &= mask - 1; // clear lowest set bit
        }
        i += 32;
    }
    for_each_newline_scalar(&bytes[i..], &mut |p| f(i + p));
}

fn for_each_newline_scalar<F: FnMut(usize)>(bytes: &[u8], f: &mut F) {
    for (p, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            f(p);
        }
    }
}

/// Extra slots `flatten_newlines` needs past the last position: it writes whole groups
/// unconditionally and fixes the count afterwards.
pub const FLATTEN_SLACK: usize = 64;

/// Writes `base + i` for the position `i` of every `\n` in `bytes` into `out` (ascending)
/// and returns how many there are.
///
/// Branch-free per block, in the style of simdjson's `flatten_bits`: the number of
/// newlines per 64-byte block is unpredictable (here ~24% of 32-byte blocks hold one),
/// so looping `while mask != 0` mispredicts about once per newline, ~10 ns each. Instead
/// a fixed group of 8 positions is written every time and only the count advances.
///
/// # Panics
/// If `out.len() < bytes.len() + FLATTEN_SLACK`.
pub fn flatten_newlines(bytes: &[u8], base: u32, out: &mut [u32]) -> usize {
    assert!(
        out.len() >= bytes.len() + FLATTEN_SLACK,
        "output buffer too small"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("bmi1")
        {
            // SAFETY: the CPU features were just detected and `out` was checked to have the slack.
            return unsafe { flatten_newlines_avx2(bytes, base, out) };
        }
    }
    let mut n = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            out[n] = base + i as u32;
            n += 1;
        }
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt,bmi1")]
unsafe fn flatten_newlines_avx2(bytes: &[u8], base: u32, out: &mut [u32]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let dst = out.as_mut_ptr();
    let mut n = 0usize;
    let mut i = 0usize;

    while i + 64 <= len {
        // SAFETY: `i + 64 <= len`.
        let (lo, hi) = unsafe {
            (
                _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i),
                _mm256_loadu_si256(bytes.as_ptr().add(i + 32) as *const __m256i),
            )
        };
        let m_lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, nl)) as u32 as u64;
        let m_hi = _mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, nl)) as u32 as u64;
        let mut mask = m_lo | (m_hi << 32);
        let count = mask.count_ones() as usize;
        let at = base + i as u32;

        // SAFETY: every write is at index < n + 16 (+ the loop below, bounded by `count`),
        // and `n + count <= len`, so with FLATTEN_SLACK >= 64 they stay inside `out`.
        unsafe {
            // First 4 positions, written unconditionally: the average block holds < 1 newline,
            // so a group of 8 mostly wrote garbage. Larger groups are written only when needed.
            for j in 0..4 {
                *dst.add(n + j) = at + mask.trailing_zeros();
                mask &= mask.wrapping_sub(1);
            }
            if count > 4 {
                for j in 4..8 {
                    *dst.add(n + j) = at + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                }
            }
            if count > 8 {
                for j in 8..16 {
                    *dst.add(n + j) = at + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                }
                let mut j = 16;
                while j < count {
                    *dst.add(n + j) = at + mask.trailing_zeros();
                    mask &= mask.wrapping_sub(1);
                    j += 1;
                }
            }
        }
        n += count;
        i += 64;
    }

    // Tail of fewer than 64 bytes.
    while i < len {
        if bytes[i] == b'\n' {
            // SAFETY: n < len + 1 <= out.len().
            unsafe { *dst.add(n) = base + i as u32 };
            n += 1;
        }
        i += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(bytes: &[u8]) -> Vec<usize> {
        bytes
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == b'\n')
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn matches_reference_for_all_alignments() {
        let mut seed = 12345u64;
        let mut buf = Vec::new();
        for _ in 0..300 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            buf.push(if (seed >> 33) % 5 == 0 { b'\n' } else { b'x' });
        }
        for start in 0..40 {
            for end in start..buf.len() {
                let s = &buf[start..end];
                let mut got = Vec::new();
                for_each_newline(s, |p| got.push(p));
                assert_eq!(got, reference(s));
                assert_eq!(count_newlines(s), got.len());
            }
        }
    }
}

#[cfg(test)]
mod flatten_tests {
    use super::*;

    #[test]
    fn flatten_matches_reference_for_all_alignments_and_densities() {
        let mut seed = 99u64;
        for density in [0u64, 1, 3, 8, 20, 100] {
            // `density` % of bytes are newlines (100 = all of them).
            let mut buf = Vec::new();
            for _ in 0..700 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                buf.push(if (seed >> 33) % 100 < density {
                    b'\n'
                } else {
                    b'x'
                });
            }
            for start in 0..70 {
                for end in [
                    start,
                    start + 1,
                    start + 63,
                    start + 64,
                    start + 65,
                    start + 129,
                    buf.len(),
                ] {
                    let s = &buf[start..end.min(buf.len())];
                    let want: Vec<u32> = s
                        .iter()
                        .enumerate()
                        .filter(|&(_, &b)| b == b'\n')
                        .map(|(i, _)| 1000 + i as u32)
                        .collect();
                    let mut out = vec![0u32; s.len() + FLATTEN_SLACK];
                    let n = flatten_newlines(s, 1000, &mut out);
                    assert_eq!(
                        &out[..n],
                        &want[..],
                        "density {density} range {start}..{end}"
                    );
                }
            }
        }
    }
}
