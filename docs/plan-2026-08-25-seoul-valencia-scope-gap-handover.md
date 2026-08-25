# Handover: what it actually takes to let Seoul and Valencia into the pipeline

Written for `gu-dev_heatready-seoul-valencia-rescope` to read and build on directly, not to
re-derive from a transcript. Everything below is grounded in the real, current code (cited by
file/function) as of PRs #23, #24, #25 (this repo) and #440-443 (`heat-risk-data-api`), plus the
real Valencia phase docs in `heat-risk-data-api`'s `research/downscaling-confidence/`. These are
informed starting hypotheses, not a verified design — validate before building.

## The core problem, stated plainly

The P0 work this session shipped (`score_band`'s `proposed_correction` extension, `gates.py`'s
subzone fields, `replay_downscaling.py`, `promote_from_public.py`) only ever scores and publishes
a **flat constant** — one number, or one `{scale, offset}` pair, per zone or per subzone bucket.
Neither Seoul's real result nor Valencia's real result is that shape. Building the P0 pipeline
without revisiting that ceiling as the two grounding cases came into focus is the actual gap
Nishant flagged — not a rough edge to polish, the unmet goal.

## Valencia: the shape gap

**What exists today.** `gates.py`'s `BAND_GATE_SCHEMA` (lines 13-145) has exactly four correction
shapes, and all four are flat constants over a discrete key:
- `bias_correction[target][zone]` — one number.
- `delta_scale[target][zone]` — one `{scale, offset}` pair.
- `bias_correction_subzone` / `delta_scale_subzone[target][zone][subzone_code]` — same two shapes,
  one level finer, keyed by a GHCN country-prefix subzone code (e.g. `"FR"`). Still a finite,
  discrete bucket — not a continuous variable.

`score.py`'s `score_band(proposed_correction=...)` mirrors this: `PROPOSED_CORRECTION_BIAS_KEYS =
("bias_correction_c",)` and `PROPOSED_CORRECTION_AFFINE_KEYS = ("scale", "offset")` are the only
two shapes a contributor's `proposed_correction[zone]` entry can take (score.py lines 47-48,
232-268). Both are one constant pair applied uniformly across every row in the zone.

**What Valencia actually found.** A nighttime correction that varies *continuously* with distance
from the coast — not a zone-level or subzone-level bucket value. (19.4% year-round error
reduction, 95% CI 10.4%-25.2%, per the phase docs already cited in the leadership report.) There
is no bucket size where this collapses into a flat constant without losing the result — that's the
whole point of the finding.

**What a real fix needs (starting sketch, not a spec):**
1. A new gate field, e.g. `delta_scale_covariate[target][zone]`, holding a functional form instead
   of a constant — at minimum `{covariate: "distance_to_coast_km", slope: <number>, intercept:
   <number>, valid_range: [min, max]}` (linear is Valencia's actual finding; don't generalize to
   arbitrary functions without a second real case motivating it — see the "no half-finished
   abstractions" norm this codebase already follows elsewhere).
2. `score.py`'s `score_band` needs a third `proposed_correction_kind` (e.g. `"covariate_linear"`)
   evaluated **per row** using each row's own covariate value, not one shared constant — a
   genuinely different code path from the current `bias`/`affine` branches (score.py lines
   232-268), not a config toggle on top of them.
3. `submission.py`'s `_CANDIDATE_ZONE_SCHEMA` (line 49) needs a matching shape so a contributor can
   actually submit this correction kind.
4. **The hard, unresolved part, flagged not solved:** at scoring time the covariate value is known
   per-station (real station metadata). At serving time, the model needs the covariate value for
   an arbitrary future project location, not a station — i.e. a distance-to-coast lookup/raster at
   inference time, which doesn't exist today anywhere in the serving path. This is almost certainly
   why the original P0 design deferred this shape rather than an oversight — confirm that read
   against the design doc (`docs/plan-2026-08-25-crowdsourced-model-improvement-p0.md`) before
   assuming it's a small addition.

## Seoul: the path gap

**What exists today.** Two separate things gate Seoul out, and they are not the same problem:
- Rung C (contributing new model code) was declared out of scope in the very first brief, pending
  "a real decision about how to safely run someone else's model code at all" — a sandboxing
  problem, because a public, anonymous contributor's code is untrusted by default.
- The promotion path built this session (`heat-risk-data-api`'s `scripts/promote_from_public.py`)
  assumes its input is a public ledger winner that cleared two consecutive monthly cycles — an
  anti-gaming check, because an anonymous contributor could otherwise pick a favorable slice of
  history.

**Why Seoul doesn't actually need either of those solved.** Seoul's result is a retrained model,
but it is *your own team's* result, already reproducible deterministically and credential-free
through `contract.py`'s `FrozenPredictionAdapter` — the exact same frozen-prediction shape
`score_band` and `replay_downscaling.py` already consume for scoring. It was never run as
untrusted third-party code inside the pipeline; it's a known, trusted, already-validated result
sitting outside a gate built for strangers.

**Recommended framing, not yet built:** don't reopen Rung C (arbitrary contributor code execution)
to unblock Seoul. Instead, build a second, maintainer-only promotion entry point for trusted
internal research results — takes a `FrozenPredictionAdapter`-shaped result directly, reuses
`promote_from_public.py`'s existing `rederive()` / `check_evidence_bar()` verification logic
(these don't care where the winning claim came from, only whether it re-derives and clears the
bar), but skips the public ledger and the 2-consecutive-cycle wait entirely — because there is no
anonymous-contributor gaming risk to defend against for a result your own team ran and can
re-derive on demand.

**This is also the direct answer to Nishant's "two months is untenable in an emergency" concern.**
The 2-cycle gate was built to defend the *public* path against gaming. It was never meant to gate
trusted internal research, and routing Seoul/Valencia-shaped internal results through it anyway is
the actual design error — not the cycle count itself. Fixing the cycle count would be solving the
wrong problem; splitting the path is the real fix.

## Where to look first

- `heat-ready-downscaling`: `src/heatready_downscaling/gates.py`, `score.py`, `submission.py`,
  `contract.py` (`FrozenPredictionAdapter`).
- `heat-risk-data-api`: `scripts/promote_from_public.py` (`rederive`, `check_evidence_bar`,
  `merge_patch`), `scripts/replay_downscaling.py`'s counterparts, and the real Valencia numbers in
  `research/downscaling-confidence/PHASE2_PROXY_AND_WIDER_SOURCES.md`,
  `PHASE6_REAL_LOCAL_MODEL.md`, `PHASE7_VALENCIA_IS_MOSTLY_BSH_NOT_CSA.md`.
- The original design doc this all descends from: `docs/plan-2026-08-25-crowdsourced-model-
  improvement-p0.md` (PR #23) — re-read its own stated scope boundaries before assuming a gap here
  is new information; some of this may already be named there as deferred, not missed.
