# spec-decisions

Open questions about the *notation* that the implementation currently answers by accident. They are research
decisions, so they need an owner's call, not a code fix. Each one changes what "line-local, data-parallel
parsing" can claim.

## 1. Multi-line raw blocks (fenced code)

Only the ``` fence line is classified `Raw`. Lines inside a block are classified like any other line (`- x`
becomes a List, `@x` an Element), and the AST bridge turns `Raw` into a paragraph. Options:

- (a) No multi-line constructs: every line of a raw block carries its own marker. Stays purely line-local.
- (b) Fence parity via a prefix scan (XOR of fence flags). Still data-parallel, and a natural fit for the
      existing GPU scan, but a line is no longer classifiable from its own bytes alone.

## 2. Header vs callout (`>[ ... ]`)

The scanner always yields `Callout`. The idea notes say `|>[ Header ]` (depth 1) is a header and
`||>[ Callout ]` (depth 2) is a callout, i.e. the meaning depends on depth (and on the "ground level is special"
problem noted in `docs/idea-bar-style.ol`). Decide whether that is a lexical rule (then `kind` should encode it)
or a bridge-time rule (current behaviour).

## 3. Heading level

`Section` level currently comes from `depth.max(1)`; the number of `=` is ignored, so `==[ Sub ]` does not
mean level 2. Decide between depth-derived level, `=`-count level, or both (and what a mismatch means).

## 4. Prose lines under a paragraph

A deeper prose line under a paragraph is merged into it as a soft break (kept from the old builder); same-depth
consecutive prose lines become separate paragraphs. Confirm that is intended, since it is the only way a
paragraph spans lines.

## 5. Escaping

A literal `|` at the start of content (the datagen vocabulary contains `|pipe|`) cannot be distinguished from a
depth marker. Define the escape or rule out such content.

When a decision lands, add a differential/regression test that pins it for CPU and GPU, and update
`classify_line` and `shader.wgsl` together.
