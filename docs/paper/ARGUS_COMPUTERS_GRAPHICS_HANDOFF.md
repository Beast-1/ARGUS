# ARGUS Computers and Graphics Manuscript Handoff

## Revised manuscript

`argus_computers_graphics_revision.tex` is a submission-oriented revision of the original manuscript. It preserves the reported pilot results but narrows claims that are not supported by the current evidence.

The revision now describes the effective score as a topology/material-aware candidate-selection heuristic rather than a validated perceptual metric. It also distinguishes export completion from semantic or perceptual success, identifies the 15-prompt run as a pilot, and states that external baselines, causal ablations, and human evaluation remain outstanding.

## Required before submission

1. Freeze the repository revision, benchmark, prompts, provider/model versions, and final 15-run manifest.
2. Run the supported ablations on all 30 prompts.
3. Add at least one external text-to-3D baseline and one direct-code baseline.
4. Repeat visual judging and compare it with human ratings.
5. Add a subtraction operator for prompts that require holes, slots, or internal voids.
6. Compile the LaTeX manuscript and visually inspect the PDF, figures, references, and final page count.

## Reproducibility command

From the ARGUS repository root, regenerate the current aggregate metrics with:

```text
python -m eval.report --results eval/results/runs.jsonl --format md
```

The original manuscript remains unchanged as `argus.tex`.
