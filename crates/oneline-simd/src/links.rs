//! Parallel construction of [`TreeLinks`] (parent + subtree end) from the flat nodes.
//!
//! Both relations are *all nearest smaller values* (ANSV) problems over the
//! per-line `depth` array, solved in three phases that map one-to-one onto a GPU
//! implementation (block-local pass, small serial combine, block-local resolve):
//!
//! 1. **local** (parallel per block): run the usual monotonic stack inside the
//!    block. Lines whose answer lies before the block ("pending") are the strict
//!    prefix minima of the block; the stack left at the end is the block summary.
//! 2. **combine** (serial over blocks): fold each block summary into a global
//!    stack kept as a *persistent* linked list, remembering its head at every
//!    block start. Total work is bounded by the summary sizes, not by `n`.
//! 3. **resolve** (parallel per block): every pending line walks that list.
//!    Pending values are non-increasing, so one walk per block suffices and the
//!    walk length is bounded by what the combine phase popped for the block.
//!
//! Total work is `O(n)`.
//!
//! Blank lines are made invisible by giving them the key `u32::MAX`: such a key is
//! never an answer for a content line and never pops one. Their own outputs are
//! overwritten at the end.

use oneline_core::{FlatNode, NONE, NodeKind, TreeLinks};
use rayon::prelude::*;

/// Lines per block in the local/resolve phases.
const BLOCK: usize = 1 << 15;
/// Internal marker: "answer lies in an earlier block". Never a valid line index.
const PENDING: u32 = u32::MAX - 1;

/// Does candidate key `v` answer a query for key `x`?
#[inline(always)]
fn ok(v: u32, x: u32, or_equal: bool) -> bool {
    if or_equal { v <= x } else { v < x }
}

/// One entry of the persistent global stack.
struct Entry {
    key: u32,
    idx: u32,
    link: u32,
}

/// For every `i`, the nearest `j < i` with `ok(keys[j], keys[i])`, or [`NONE`].
fn nearest_previous(keys: &[u32], or_equal: bool) -> Vec<u32> {
    let n = keys.len();
    assert!(n < PENDING as usize, "too many lines for 32-bit indices");
    let mut res = vec![0u32; n];
    if n == 0 {
        return res;
    }

    // Phase 1: local stacks.
    let stacks: Vec<Vec<u32>> = res
        .par_chunks_mut(BLOCK)
        .zip(keys.par_chunks(BLOCK))
        .enumerate()
        .map(|(b, (out, ks))| {
            let base = b * BLOCK;
            let mut stack: Vec<u32> = Vec::new();
            for (j, &x) in ks.iter().enumerate() {
                while let Some(&top) = stack.last() {
                    if ok(keys[top as usize], x, or_equal) {
                        break;
                    }
                    stack.pop();
                }
                out[j] = stack.last().copied().unwrap_or(PENDING);
                stack.push((base + j) as u32);
            }
            // Block summary. Among equal keys only the latest entry can ever be an
            // answer for a later block (it is nearer and equally `ok`), so drop the
            // rest; otherwise runs of equal depths would make this summary O(block).
            let mut summary: Vec<u32> = Vec::with_capacity(stack.len().min(64));
            for (i, &s) in stack.iter().enumerate() {
                let next_same = stack
                    .get(i + 1)
                    .is_some_and(|&t| keys[t as usize] == keys[s as usize]);
                if !next_same {
                    summary.push(s);
                }
            }
            summary
        })
        .collect();

    // Phase 2: persistent global stack; `heads[b]` is its top at the start of block `b`.
    let mut pool: Vec<Entry> = Vec::new();
    let mut heads = Vec::with_capacity(stacks.len());
    let mut head = NONE;
    for stack in &stacks {
        heads.push(head);
        if let Some(&first) = stack.first() {
            // The block minimum is the stack bottom: whatever is not `ok` against
            // it is popped by this block, everything below survives.
            let min_key = keys[first as usize];
            while head != NONE && !ok(pool[head as usize].key, min_key, or_equal) {
                head = pool[head as usize].link;
            }
            for &s in stack {
                pool.push(Entry {
                    key: keys[s as usize],
                    idx: s,
                    link: head,
                });
                head = (pool.len() - 1) as u32;
            }
        }
    }

    // Phase 3: resolve pending lines.
    res.par_chunks_mut(BLOCK).enumerate().for_each(|(b, out)| {
        let mut p = heads[b];
        for (j, slot) in out.iter_mut().enumerate() {
            if *slot != PENDING {
                continue;
            }
            let x = keys[b * BLOCK + j];
            while p != NONE && !ok(pool[p as usize].key, x, or_equal) {
                p = pool[p as usize].link;
            }
            *slot = if p == NONE {
                NONE
            } else {
                pool[p as usize].idx
            };
        }
    });

    res
}

/// Builds [`TreeLinks`] in parallel. Output is identical to
/// [`TreeLinks::from_nodes_sequential`].
pub fn build_links(nodes: &[FlatNode]) -> TreeLinks {
    let n = nodes.len();

    let keys: Vec<u32> = nodes
        .par_iter()
        .map(|nd| {
            if nd.kind() == NodeKind::Blank {
                u32::MAX
            } else {
                nd.depth.min(u32::MAX - 1)
            }
        })
        .collect();

    // parent: nearest previous line with a strictly smaller depth.
    let mut parent = nearest_previous(&keys, false);

    // subtree_end: nearest next line with depth <= own depth == nearest previous
    // in the reversed array with `<=`.
    let rev_keys: Vec<u32> = keys.par_iter().rev().copied().collect();
    let rev_next = nearest_previous(&rev_keys, true);

    // One fused pass: translate the reversed answer and drop blank lines
    // (identified by their sentinel key) from the tree.
    let mut subtree_end = vec![0u32; n];
    parent
        .par_iter_mut()
        .zip(subtree_end.par_iter_mut())
        .enumerate()
        .for_each(|(i, (p, e))| {
            if keys[i] == u32::MAX {
                *p = NONE;
                *e = i as u32 + 1;
            } else {
                *e = match rev_next[n - 1 - i] {
                    NONE => n as u32,
                    r => (n - 1 - r as usize) as u32,
                };
            }
        });

    TreeLinks {
        parent,
        subtree_end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oneline_core::datagen::{self, Profile, Rng};

    fn nodes_from(spec: &[(u32, bool)]) -> Vec<FlatNode> {
        spec.iter()
            .map(|&(d, blank)| {
                FlatNode::new(
                    d,
                    if blank {
                        NodeKind::Blank
                    } else {
                        NodeKind::Text
                    },
                    0,
                    1,
                )
            })
            .collect()
    }

    fn check(nodes: &[FlatNode]) {
        assert_eq!(build_links(nodes), TreeLinks::from_nodes_sequential(nodes));
    }

    #[test]
    fn tiny_cases() {
        check(&[]);
        check(&nodes_from(&[(0, false)]));
        check(&nodes_from(&[(0, true), (3, true)]));
        check(&nodes_from(&[
            (1, false),
            (2, false),
            (3, false),
            (2, false),
            (0, true),
            (1, false),
            (0, false),
        ]));
        // Depth jumps (1 -> 3 -> 2) and equal depths.
        check(&nodes_from(&[
            (1, false),
            (3, false),
            (2, false),
            (2, false),
            (1, false),
            (1, false),
        ]));
    }

    /// Pathological shapes for the persistent-stack combine phase, across block borders.
    #[test]
    fn adversarial_shapes_cross_blocks() {
        let n = BLOCK * 5 + 123;
        let shapes: Vec<(&str, Vec<u32>)> = vec![
            ("flat", vec![1; n]),
            ("flat zero", vec![0; n]),
            ("strictly increasing", (0..n as u32).collect()),
            ("strictly decreasing", (0..n as u32).rev().collect()),
            (
                "header then flat",
                std::iter::once(0)
                    .chain(std::iter::repeat_n(1, n - 1))
                    .collect(),
            ),
            ("sawtooth", (0..n as u32).map(|i| i % 7).collect()),
            (
                "sawtooth long",
                (0..n as u32).map(|i| (i / 3) % 40000 % 11).collect(),
            ),
            ("alternating", (0..n as u32).map(|i| i % 2).collect()),
        ];
        for (name, depths) in shapes {
            let nodes: Vec<FlatNode> = depths
                .iter()
                .map(|&d| FlatNode::new(d, NodeKind::Text, 0, 1))
                .collect();
            let got = build_links(&nodes);
            let want = TreeLinks::from_nodes_sequential(&nodes);
            assert!(got == want, "shape {name:?} differs");
        }
    }

    #[test]
    fn random_depths_with_blanks() {
        let mut rng = Rng::new(2024);
        for round in 0..40 {
            let n = rng.range(1, BLOCK as u64 * 4) as usize;
            let max_depth = [1, 2, 4, 8, 64][round % 5];
            let nodes: Vec<FlatNode> = (0..n)
                .map(|_| {
                    let blank = rng.chance(10);
                    FlatNode::new(
                        rng.below(max_depth) as u32,
                        if blank {
                            NodeKind::Blank
                        } else {
                            NodeKind::Text
                        },
                        0,
                        1,
                    )
                })
                .collect();
            let got = build_links(&nodes);
            let want = TreeLinks::from_nodes_sequential(&nodes);
            assert!(
                got == want,
                "round {round} (n={n}, max_depth={max_depth}) differs"
            );
        }
    }

    #[test]
    fn real_documents() {
        for p in Profile::ALL {
            let text = datagen::generate(p, 1 << 21, 5);
            let tree = crate::parse_par(&text);
            check(&tree.nodes);
        }
    }

    #[test]
    fn subtree_is_a_contiguous_range() {
        let text = "|=[ A ]\n|| x\n||- y\n|||- z\n|=[ B ]\n|| w\n";
        let tree = crate::parse(text);
        let l = build_links(&tree.nodes);
        assert_eq!(l.subtree(0), 0..4); // "A" owns x, y, z
        assert_eq!(l.subtree(4), 4..6); // "B" owns w
        assert_eq!(l.parent_of(3), Some(2));
    }
}
