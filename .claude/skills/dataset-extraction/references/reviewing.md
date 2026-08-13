# Reviewing a finished run

This is the checklist for the stage-6 verification subagent. It is not generic advice —
every item on it comes from something that actually went wrong in a real run of this
skill, and would have shipped if nobody had looked.

## Why this exists

The gates and the reviewer catch different things, and the split is not close.

Across four validation runs, **every scientific error was caught by a fresh reviewing
agent and none by a gate**: 803 molecules with the wrong bonds, a boiling point labelled
Kelvin, 196 geometries that were a different compound from their own formula, three
restrictive upstream databases, and a fabricated paper title.

That is not a gap the gates can close by getting stricter, and it is worth understanding
why before you start. **A gate compares things the agent wrote against other things the
same agent wrote.** Take the Kelvin case: the gate could confirm a unit was declared, and
that the description matched the approved draft — and both said Kelvin, because the agent
was consistently wrong. Every internal comparison agreed. Only going back to the source
and to physics could break the tie.

So your job is not to re-run the gates. They have already passed. Your job is to compare
what shipped against **things the agent did not write**: the paper, the raw download,
physical reality, and the run's own machine-generated record.

## Before you start

Read the run's `findings.md` — it is the agent's own account of what it did and what it
was unsure about. Treat the uncertainties it records as the first places to look, and
treat its confident claims as the ones most worth testing.

Then get the numbers:

```bash
cmmanage agent gate verification <slug> --strict --no-record
```

`--no-record` means you can run this as often as you like without marking anything stale.

## The checklist

### 1. Do the bonds fit the geometry?

*For any dataset carrying 3D coordinates.*

For every bond, compare its length against the sum of the two atoms' covalent radii. A
bond much longer than that is not a bond.

This caught **196 molecules** whose published geometry was a chemically different species
from the published formula — gas-phase optimisation had let ion pairs react, selenium
migrate off phosphorus, and waters of hydration dissociate. The longest "bond" was
**7.6 Å**. An element-by-element comparison could not see it, because the atoms and their
order were unchanged; only the bonding differed.

If you find these, check whether re-deriving the structure from the geometry agrees with
the published formula. If it agrees with neither, the entries are unusable rather than
one method being wrong, and they should be dropped.

### 2. Does the unit permit the values?

Take each target's stated unit and ask what values it can physically hold.

A boiling point described as "in Kelvin" with a minimum of **−88.6** shipped in this
collection for months. Kelvin has no negative values; the data was Celsius. Nothing in the
metadata was malformed — it was simply false, and it was false consistently.

Apply the same reasoning to anything on an absolute scale: masses, counts, concentrations,
absolute temperatures. Beware log-transformed quantities, which legitimately go negative.

### 3. Do the structures match the published ones?

Where the source publishes both structures and geometries, compare them: canonicalise both
sides the same way and count the disagreements.

An early version of one run *inferred* bonds from the coordinates rather than using the
published structures, and produced **803 molecules** that disagreed — 610 with wrong bond
orders or charges, 193 with outright wrong connectivity. Every one was a valid molecule.
Just not the right one.

### 4. Do the two artifacts agree?

The run writes a `.mpack` and a `.csv` from the same data. Check that row *i* of one is the
same compound as row *i* of the other.

They were a **complete permutation** of each other for every dataset in this collection —
5,432 of 5,432 rows mismatched, with identical contents — because graphs came off a
worker queue in completion order while the CSV was written in source order. Both files
existed and both had the right length, which was all anything checked.

### 5. Does the count match what the paper says?

The paper states how large its dataset is. Compare.

A difference is not automatically wrong — filtering rules legitimately change the number —
but it must be *explained*, and the explanation must match the filters the script actually
applies. In one run the approved metadata described the paper's dataset while the shipped
one differed by a row.

### 6. Do the citations resolve?

Fetch every DOI and URL in `sources`, and check that the title and first author match what
is recorded.

An agent cited its source paper under an **invented title with the wrong first author**.
It was found by luck, by a subagent checking attribution for an unrelated reason. Nothing
in the pipeline looks for confabulated provenance, and the output of this pipeline is a
citable scientific artifact.

### 7. Does the run's own record agree?

`experiment_meta.json` and `experiment_code.py` are written by the experiment framework,
not by the agent, so they can contradict it. Check `status` is `done`, `has_error` is
false, and that the script on disk is the one that produced these results.

### 8. Is the element identity intact?

Check the element symbols in the graphs against the source. The shared encoder recognises
a fixed list, and anything outside it — including hydrogen — used to be written as the
string `unknown`, so a selenium and a hydrogen were indistinguishable. Fixed in the
library, but worth confirming for any dataset containing unusual elements.

## What to report

Findings, not reassurance. For each item: what you checked, what you found, and whether it
blocks. If you cannot check something, say so plainly rather than omitting it — a gap you
name is useful, a gap you hide is not.

Say explicitly whether the dataset is **publishable**, and if not, what would have to
change.

## Adding to this list

When a run surfaces a new way of being wrong, add it here with the number attached — how
many rows, how far off, what it looked like. The specifics are what make the next reviewer
able to spot it, and every item above earned its place that way.

## If you want to propose a new gate instead

Sometimes a finding suggests a check that could be mechanical. Before proposing one,
apply this test:

> **A gate is only worth building if it compares against something the agent did not
> write.**

Otherwise you are verifying that the agent can copy a value from one field to another,
which it can. Every gate in this pipeline that has ever caught a real defect satisfies
the rule:

| Check | The independent witness |
|---|---|
| Run status and script identity | `experiment_meta.json` / `experiment_code.py`, written by the framework |
| A unit that cannot hold its values | physics — Kelvin has no negatives, and nothing can argue |
| The two artifacts agreeing | two files produced by different code paths |
| Upstream licence provenance | the publisher's own terms |

And every check that has *failed* to catch something compared the agent to itself: the
metadata draft against the metadata the same agent wrote into the script, the declared
target count against the declared descriptions.

The corollary is that some things simply cannot be gated, and the honest response is to
put them on this checklist rather than to build a check that looks reassuring and proves
nothing.
