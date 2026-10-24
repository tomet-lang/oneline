use std::ops::Range;

use crate::{FlatNode, NodeKind};

/// "No such line" marker used in [`TreeLinks`].
pub const NONE: u32 = u32::MAX;

/// Tree structure recovered from the per-line `depth` array.
///
/// Because every line states its own depth, the tree is fully determined by two
/// all-nearest-smaller-values (ANSV) relations over `depth`:
///
/// * `parent[i]` = nearest *previous* content line with `depth < depth[i]`
/// * `subtree_end[i]` = nearest *next* content line with `depth <= depth[i]`
///
/// The subtree of line `i` is therefore the contiguous line range
/// `i..subtree_end[i]` (blank lines inside it included). That is what makes
/// structural queries ("everything under this heading") plain range scans.
///
/// Blank lines are not part of the tree: `parent == NONE` and
/// `subtree_end == i + 1` for them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeLinks {
    pub parent: Vec<u32>,
    pub subtree_end: Vec<u32>,
}

impl TreeLinks {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn parent_of(&self, line: usize) -> Option<usize> {
        match self.parent[line] {
            NONE => None,
            p => Some(p as usize),
        }
    }

    /// Lines belonging to the subtree rooted at `line` (including `line` itself).
    pub fn subtree(&self, line: usize) -> Range<usize> {
        line..self.subtree_end[line] as usize
    }

    /// Sequential reference implementation (monotonic stack).
    ///
    /// The parallel builder in `oneline-simd` must produce identical output.
    pub fn from_nodes_sequential(nodes: &[FlatNode]) -> Self {
        let n = nodes.len();
        let mut parent = vec![NONE; n];
        let mut subtree_end: Vec<u32> = (1..=n as u32).collect();
        let mut stack: Vec<u32> = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            if node.kind() == NodeKind::Blank {
                continue;
            }
            while let Some(&top) = stack.last() {
                if nodes[top as usize].depth >= node.depth {
                    subtree_end[top as usize] = i as u32;
                    stack.pop();
                } else {
                    break;
                }
            }
            parent[i] = stack.last().copied().unwrap_or(NONE);
            stack.push(i as u32);
        }
        for top in stack {
            subtree_end[top as usize] = n as u32;
        }

        Self {
            parent,
            subtree_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(depth: u32, kind: NodeKind) -> FlatNode {
        FlatNode::new(depth, kind, 0, 1)
    }

    #[test]
    fn sequential_reference_small() {
        // d: 1 2 3 2 (blank) 1 0
        let nodes = [
            n(1, NodeKind::Text),
            n(2, NodeKind::Text),
            n(3, NodeKind::Text),
            n(2, NodeKind::Text),
            n(0, NodeKind::Blank),
            n(1, NodeKind::Text),
            n(0, NodeKind::Text),
        ];
        let l = TreeLinks::from_nodes_sequential(&nodes);
        assert_eq!(l.parent, vec![NONE, 0, 1, 0, NONE, NONE, NONE]);
        // line 1 (d2) ends at line 3 (next d<=2); line 2 (d3) also at line 3.
        assert_eq!(l.subtree_end, vec![5, 3, 3, 5, 5, 6, 7]);
        assert_eq!(l.subtree(0), 0..5);
        assert_eq!(l.parent_of(3), Some(0));
        assert_eq!(l.parent_of(4), None);
    }
}
