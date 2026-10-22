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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(bytes: &[u8]) -> Vec<usize> {
        bytes.iter().enumerate().filter(|&(_, &b)| b == b'\n').map(|(i, _)| i).collect()
    }

    #[test]
    fn matches_reference_for_all_alignments() {
        let mut seed = 12345u64;
        let mut buf = Vec::new();
        for _ in 0..300 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
