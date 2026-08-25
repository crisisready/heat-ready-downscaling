# P0 design: Rung B scoring + `replay_downscaling.py` + `promote_from_public.py`

Status: **design only, no code**. This plan is posted for review before any of the three pieces
below gets implemented, per the Tier 2 ("new pipeline mechanism, cross-repo, real blast radius")
plan-before-code convention this session is operating under. **Correction (code-review finding,
2026-08-25): that tier convention is the machine-wide gu-dev baseline
(`~/.claude-harvard/CLAUDE.md`), explicitly scoped there to homestead work — this repo's own
`CLAUDE.md` does not define any tier system.** Cited here as the actual standard this design PR
follows, not as something `CLAUDE.md` itself specifies. Nothing in this PR touches `score.py`,
adds `replay_downscaling.py`, or changes `heat-risk-data-api`.

Scope: the three pieces `task-inventory.md` §3 named as the crowdsourced model-improvement
program's "engine room": (1) `score_band` accepting a contributor-proposed correction (Rung B),
(2) `replay_downscaling.py`, the tier-mix/delta-distribution diff for the human promotion-review
step `GOVERNANCE.md` already requires, and (3) `promote_from_public.py`, the production-promotion
half in `heat-risk-data-api`. Rung C (new model code) is explicitly out of scope — closed pending
an unresolved security question about executing untrusted model code, per `GOVERNANCE.md`.

## Grounding: what Seoul and Valencia actually are

Both were read in full (`research/seoul-local-sensor-validation/`,
`research/valencia-local-model-evaluation/` in the `heat-risk-data-api` worktree) as the two real
test cases this design is built against, per Nishant's own framing in `task-inventory.md` §3/§67.
**Neither is a clean Rung B submission under CONTRIBUTING.md's current vocabulary** — that's the
central finding this design has to reckon with, not design around.

- **Seoul**: the winning object is a **genuinely retrained model** — a new QRF-equivalent fit on
  S-DoT sensor data with a real spatial (dong-level) train/test split, held-out RMSE roughly
  halved on tmax (1.95°C → 1.14°C) and cut ~70% on tmin (3.32°C → 1.13°C), zone-scoped to Cwa/Dwa.
  This is **Rung-C-shaped** (new model code/artifact), not a `bias_correction` float or a
  blend-kernel triple. It is out of scope for this design by the brief's own terms.
  - Worth flagging separately (not part of this PR's three deliverables, but relevant to how
    Rung B/C get scoped later): Seoul's actual real-world path to production doesn't require
    opening Rung C at all. `contract.QRFModelAdapter.load(model_version)` already swaps model
    artifacts by version, and `zones_passing_cv_gate`/`extra_zone_gate` already restrict a model
    to specific zones. A maintainer-trained model_version, fit from contributor-*discovered data*
    (not contributor-*submitted code*) and published the normal way, never touches the
    "how do we safely execute untrusted model code" question Rung C is actually blocked on. That's
    a distinct third path — "data contribution, maintainer-run training" — worth naming explicitly
    the next time Rung B/C get rescoped, but it's not one of this PR's three pieces and isn't
    designed further here.
- **Valencia**: the statistically strongest real result (`PHASE6_REAL_LOCAL_MODEL.md`) is a
  **continuous-covariate affine correction** — `corrected_grid = raw_grid + intercept + slope *
  coast_dist_km` — fit directly on ECA&D station-minus-grid deltas, bootstrap-CI-validated
  (tmin: 19.4% RMSE reduction, CI [10.4%, 25.2%]; 31.0% on hot days, CI [20.4%, 39.2%]).  It is
  **deliberately zone-agnostic** (`PHASE7_VALENCIA_IS_MOSTLY_BSH_NOT_CSA.md` — the 9-station tight
  cluster mixes BSh/BSk/Csa, and the fit never depended on the label) and it corrects the **raw
  grid directly**, not the model's `delta_c`. This doesn't fit `bias_correction[target][zone]`
  (discrete key, no covariate) or the blend-kernel triple either.
  - Separately, Valencia also has a *second*, more mundane candidate path already staged
    (`PHASE8_SUBZONE_PRODUCTION_PATH.md`): once real Spain GHCN rows land in `ghcn_training`, a
    `--regions SP` run through the existing `delta_scale_subzone`/`bias_correction_subzone`
    mechanism (`gates.py`, shipped, PR #13) would produce an ordinary `{scale, offset}` fit for
    the whole `SP` subzone — that path **does** fit today's Rung B vocabulary exactly, and is the
    one this design should actually unblock. The coast-distance covariate result is real and
    important evidence, but it's a different, richer correction shape than Rung B declares today.

**Conclusion driving the recommendation below**: build the engine room for the Rung B shape
CONTRIBUTING.md already publicly promised (a discrete `bias_correction` float or `delta_scale`
`{scale, offset}`, optionally subzone-scoped) — that's small, well-precedented, and matches a
real pending contribution. **Correction (code-review finding, 2026-08-25): there is no live
`Cfb.FR` fit anywhere** — `Cfb.FR` only appears as an illustrative value in test fixtures
(`tests/test_gates.py`, `tests/test_publish_band_gate.py`). The one real, shipped Cfb correction
(`README.md`) is zone-level, not subzone-scoped. The `delta_scale_subzone`/`bias_correction_subzone`
mechanism itself is real and shipped (`gates.py`), but has never yet been exercised with real
production data — Valencia's own Phase 8 finding is exactly "the mechanism is fine, only the
evidence is missing." The staged `SP` subzone path is what this design should unblock; it is not
already proven in production, and this doc should not imply otherwise. Do not
try to make Rung B also swallow Seoul's retrained model or Valencia's covariate correction; both
would require either executing contributor code (Rung C's open question) or inventing a new
correction vocabulary (a real future extension, not P0). `replay_downscaling.py` and
`promote_from_public.py`, however, **do** need to be built generally enough to review *either*
kind of change (a corrected zone/subzone parameter, or — later — a swapped model_version), because
that's what the promotion step in `GOVERNANCE.md` has to hold the line on regardless of which rung
produced the candidate.

---

## 1. `score_band` Rung B extension

**Today**: `score_band` only ever *derives* `bias_correction_c`/`delta_scale_c` from the rows it's
given — it has no parameter for a contributor's own proposed number. `run_submission.py` hard-rejects
every `rung: B` manifest today for exactly this reason (see its own comment on `cross_check`).

**Proposal**: add an optional, keyword-only `proposed_correction` argument:

```python
def score_band(
    adapter, rows, target, *, fold_salt,
    proposed_correction: dict[str, dict] | None = None,  # {zone: {"bias_correction_c": float}} OR {zone: {"scale": float, "offset": float}}
) -> dict:
```

When a zone has an entry in `proposed_correction`, `score_band` additionally scores *that specific
value* out-of-fold via the same station-grouped CV folds it already builds
(`fold_of_station`/`row_folds`) — apply the contributor's number to each fold's held-out rows
(never refit it per fold, that's the whole point: this scores the *proposed* value, not a
best-fit one) — and reports it alongside the existing mechanically-derived numbers:

```python
"proposed_correction_rmse_cv_c": ...,     # out-of-fold RMSE using the contributor's number
"proposed_correction_beats_grid": ...,    # vs rmse_grid_c
"proposed_vs_best_fit_gap_c": ...,        # for referee transparency, see correction below
```

**Correction (Codex adversarial review, 2026-08-25):** for the affine (`{scale, offset}`) shape,
`proposed_vs_best_fit_gap_c` must account for *both* parameters, not scale alone — a proposal that
matches the fitted `scale` exactly but carries a wildly wrong `offset` (e.g. +100°C) would
otherwise report a near-zero gap despite being nowhere near the fitted correction. Report it as a
combined distance over the *effect on delta_c*, not the raw parameters (which live on different
scales) — e.g. the RMS difference between `proposed.scale * d + proposed.offset` and
`fit.scale * d + fit.offset` evaluated at the zone's actual observed `d` (delta_c) range, not a
bare parameter-space distance.

`gates.build_gate`/`build_subzone_patch` stay unchanged — they still only ever publish the
mechanically-derived number, never a submission's raw claim, matching `GOVERNANCE.md`'s "a
submission's own reported numbers are never trusted directly." A Rung B submission's real
contribution is the **evaluation coverage plus the specific parameter value it identified**
(e.g. "the `SP` subzone's tmin bias is +1.1°C, here's the CV-validated number") — the referee
scores whether that value is real and whether it clears `AUTO_ENABLE_MARGIN`, exactly the same
gate Rung A already clears, just against a contributor-*supplied* number instead of a
maintainer-derived one.

Blend-kernel path: same idea, one level down — `validate_station_blend.py`'s grid search already
implements the leave-one-station-out neighbor-blend scoring; a Rung B submission proposing
`(L_km, R_km, tau)` for a broad Köppen group would need that scoring function (not
`score_band` itself) to accept the specific triple instead of grid-searching it, and report the
same "is your triple's blend RMSE actually beat-the-QRF, out-of-fold" verdict
`validate_station_blend.py`'s own `blend_gate_passes` already computes. This needs a parallel,
smaller change to `validate_station_blend.py`'s scoring path, not `score_band`.

`run_submission.py`'s `cross_check` gets simplified: the `manifest["rung"] != "A"` hard reject
becomes conditional — Rung B is accepted when `method.kind == "parameters"` and the manifest
declares exactly the value(s) being proposed (schema addition to `submission.MANIFEST_SCHEMA`,
not sketched further here).

**Real gap, found by Codex adversarial review, 2026-08-25 — the single most important fix to this
section:** the paragraphs above only wire `proposed_correction` into `run_submission.py`'s
*provisional* referee path. `scripts/score_forward_eval.py` — the **monthly official cycle that
actually determines wins and drives promotion** (`main()`'s `score_cell`/`consecutive_wins` call
sites) — calls `score.score_band` with no `proposed_correction` at all today, and decides
`metrics["status"] == "win"` from the mechanically-*fitted* `qrf_beats_grid_with_margin`, not from
the contributor's declared value. **Left as originally scoped, a Rung B candidate would earn or
lose every official cycle based on the auto-derived correction, never its own submitted
parameter** — the entire mechanism this design exists to build would be inert at the one stage
that actually matters for promotion. Fix: `score_forward_eval.py` must read the winning
candidate's `proposed_correction` back out of its archived submission manifest (`load_active_candidates`
already loads `submissions/{...}/manifest.yaml` per candidate cell — the value is already sitting
right there) and pass it through to `score_band` on every monthly re-score, using the *same*
fixed, declared value every cycle (never re-fit it from that cycle's own data, for the identical
leakage reason `score_band`'s own CV discipline already exists) — `consecutive_wins`/promotion
must be scored against "does the contributor's specific number keep generalizing," not "does some
better-fit number exist this month."

**Open question for Nishant**: should a Rung B submission's `proposed_correction` be allowed to
also include the subzone key (i.e., can a public contributor propose a subzone-scoped fix, or is
subzone scoping maintainer-only until there's more precedent)? Recommend: allow it — it's the
exact shape the staged `SP` fit needs — but require the submission to also supply the
`--regions`-scoped report `build_subzone_patch()` already expects, so nothing new has to be
invented on the publish side.

---

## 2. `replay_downscaling.py`

Grounded directly in what both cases' own ad hoc tooling already had to check by hand:

- Seoul's `analyze_three_arms.py` joined ERA5/existing-model/new-model predictions to ground
  truth and found the existing model's tmin correction is nearly indistinguishable from doing
  nothing (a confidence/applied-rate story, not just an RMSE number).
- Valencia's `PHASE3_BOUNDARY_DIAGNOSTIC.md` diagnosed an artificial jump between neighboring
  polygons that straddle a 0.1° grid-cell boundary — a spatial-discontinuity check, not something
  an aggregate RMSE diff would ever surface.
- `GOVERNANCE.md` names "a large `out_of_distribution` rate spike" as grounds to hold a promotion
  even after the numbers clear tolerance — i.e. the review step is explicitly about
  *distribution* shifts, not just the headline metric.

**Proposal**: a CLI, `scripts/replay_downscaling.py --snapshot-dir <dir> --band-key <band>
--model-version <model_version> --old <gate-json|none> --new <gate-json>`, that:

**Correction (Codex adversarial review, 2026-08-25):** the original sketch let each side be
"either a model_version or a gate-json" independently — this doesn't work.
`FrozenPredictionAdapter.from_snapshot` needs a `model_version` to pick the right predictions
partition, and `BAND_GATE_SCHEMA`/`BLEND_GATE_SCHEMA` carry no `model_version` field at all — a
gate JSON alone can never say which frozen-prediction partition it should be replayed against
(and the whole point of replaying old-vs-new is comparing two *corrections* under the *same*
`model_version`, never two different models at once, which is a different, harder question this
tool doesn't attempt). Fixed shape: `--model-version` is always required and shared by both
sides; `--old`/`--new` are each a gate JSON (`bias_correction`/`delta_scale`/`blend params`) to
apply on top of that one shared model's frozen predictions — `--old` may be omitted to mean "no
correction" (today's already-served baseline, for a first-ever promotion in a zone).

1. Loads the same band-paired snapshot rows both arms would score against
   (`snapshot.read_predictions_partitions`-style, reproducible, no live inference needed when
   both sides can be expressed as `FrozenPredictionAdapter` + gate parameters — exactly the
   byte-exact-lookup discipline `contract.py`'s own docstring already commits to).
2. Runs `adapter.predict` for the *old* published correction and the *new* candidate correction
   over the identical rows.
3. Reports, per (target, zone[, subzone]):
   - **Tier-mix diff**: the confidence histogram (`high`/`medium`/`low`) before vs. after, the
     `applied` rate before vs. after, the `cv_gate_passed` rate, and the `out_of_distribution`
     rate — the exact fields `contract.ModelAdapter.predict` already returns per row, diffed
     rather than reported once.
   - **Delta-distribution diff**: `delta_c`'s mean/std/p10/p50/p90 before vs. after, and the
     paired per-row delta (`new_delta_c - old_delta_c`) distribution — catches a correction that
     moves the *median* acceptably but blows up the *tails*.
   - **Boundary-discontinuity check**: for rows with polygon/station geometry, the mean absolute
     jump in the served value between geographically adjacent rows that fall on opposite sides of
     the correction's own scope boundary (grid-cell edge for a `delta_scale`, subzone edge for a
     `bias_correction_subzone`) — a direct generalization of Valencia's own
     `diagnose_grid_boundary_discontinuity.py` pattern, run against whichever boundary the
     candidate correction actually introduces.
4. Emits both a JSON diff (machine-checkable — e.g. `promote_from_public.py` can auto-flag "OOD
   rate rose by more than X pp" as a hard stop) and a human-readable summary for the maintainer's
   review.

**Non-goal**: this tool never writes anywhere. It's read-only against the snapshot and whatever
gate/model artifacts it's pointed at, matching every other tool in this repo's "compute offline,
publish is a separate deliberate step" discipline (`FrozenPredictionAdapter`, `qc_sdot_anomalies.py`,
`diagnose_grid_boundary_discontinuity.py` all follow the same pattern already).

**Open question for Nishant**: what's the actual pass/fail bar on the boundary-discontinuity
check — is this purely informational for the human reviewer (my default recommendation, since
Valencia's own diagnostic was informational-first), or should `promote_from_public.py` treat a
discontinuity above some threshold as an automatic hold, the same way `GOVERNANCE.md` already
calls out for `out_of_distribution` spikes?

---

## 3. `promote_from_public.py` (heat-risk-data-api)

Confirmed directly: this file does not exist yet anywhere in `heat-risk-data-api` (checked the
repo and every worktree). `GOVERNANCE.md`'s own description is the only spec today: "independently
re-derives every claimed metric against a private copy of the snapshot before publishing anything...
then a human (the maintainer) reviews the tier-mix/delta-distribution diff `replay_downscaling.py`
produces before confirming the publish."

**Proposal**, as a CLI in `heat-risk-data-api`:

1. Input: a submission that has won 2 consecutive official cycles (per `CONTRIBUTING.md`'s
   promotion bar) — its `manifest.yaml` + the correction value(s) it published as its provisional
   result (from `run_submission.py`'s referee output, not the contributor's own
   `claimed_report.json`).
2. **Re-derivation, never trust**: re-run `heatready_downscaling.score.score_band` (imported at
   the pinned package version, exactly like `run_submission.py` already does) against a **private**
   copy of the snapshot's `ghcn_training` data — not the public snapshot the contributor scored
   against — using the identical `fold_salt`. If the re-derived metric doesn't match the winning
   cycle's claim within the manifest's `tolerance`, hard stop; this never proceeds on a "close
   enough, ship it" basis. **This private copy must be pinned to the exact station-days the
   winning official cycle actually scored (its own `current_snapshot_version` from
   `score_forward_eval.py`'s `cycles.jsonl` line), a frozen snapshot-equivalent extract, never the
   live/growing `ghcn_training` table (Codex adversarial review finding, 2026-08-25, resolving
   the open question this section originally left unanswered in the wrong direction): re-deriving
   against a superset that has grown since the winning cycle ran legitimately shifts RMSE/bias by
   more than tight tolerances like `0.005°C`, which would silently and permanently block a
   genuinely valid winner on more-data-arrived-since noise, not a real discrepancy.** The
   "cheaper, no second copy to maintain" framing this section originally floated is exactly the
   failure mode to avoid — tolerance comparison is only meaningful against the identical dataset
   the claim was made against.
3. Calls `replay_downscaling.py` (old = currently-published gate, new = the candidate correction)
   and surfaces the diff.
4. **Human-in-the-loop, no exceptions**: presents the re-derivation result and the diff, and waits
   for the maintainer's explicit confirmation before doing anything else. This step writes to
   `s3://.../downscaling/{blend_,}gates/...` — data actually served to production — which is
   exactly what this session's own standing `production_write_policy` says auto-approval never
   covers, regardless of how clean the numbers or the diff look. This script's publish step must
   never run unattended and must never be triggered by a cron/CI signal alone, only by the
   maintainer's own live confirmation in that moment — the same boundary `GOVERNANCE.md` already
   states in words, this is just naming it as this session's own manager-directive language too.
5. **Publish is merge-only**, mirroring `build_subzone_patch()`/`merge_subzone_patch()`'s existing
   discipline: a Rung B promotion for one (target, zone[, subzone]) can never silently overwrite
   or drop an unrelated zone/subzone's already-published correction. A whole-gate rebuild is a
   separate, larger operation this script should refuse to do implicitly.

**Resolved above (was an open question in the original draft):** "private copy of the snapshot"
means a literal, frozen extract pinned to the winning cycle's own station-days, not the live
production `ghcn_training` table — see the correction in point 2. A live superset is not a
"stronger check," it's a different, uncontrolled dataset that breaks tolerance comparison.

---

## Summary recommendation

1. Extend `score_band` with `proposed_correction` (Section 1) — small, precedented, unblocks
   exactly the Rung B shape CONTRIBUTING.md already promised and the staged Valencia `SP` subzone
   path actually needs. Do not extend Rung B's vocabulary to cover Valencia's covariate-affine
   correction or Seoul's retrained model in this pass. **Critically, this must include wiring
   `score_forward_eval.py`'s monthly official cycle to the same declared value, not just
   `run_submission.py`'s provisional referee** (Codex finding, Section 1) — otherwise promotion
   is decided by the auto-derived number regardless of what a Rung B submission actually proposed,
   making the whole extension inert where it matters most.
2. Build `replay_downscaling.py` (Section 2) general enough to diff *any* old-vs-new correction
   pair (a parameter change today, a swapped model_version later) on tier-mix, delta-distribution,
   and boundary discontinuity — because the promotion review step has to hold regardless of which
   rung or path produced the candidate.
3. Build `promote_from_public.py` (Section 3) around re-derivation-never-trust + mandatory
   human confirmation + merge-only publish, explicitly wired to this session's
   `production_write_policy` (no automated path to a production write, ever).
4. Flag, but do not resolve in this PR: the "maintainer-trained model_version from
   contributor-discovered data" path Seoul's own workflow already demonstrates works today
   without opening Rung C — worth a named line item the next time Rung B/C get rescoped, since it
   sidesteps Rung C's actual blocker (executing contributor *code*) entirely.

No implementation in this PR. Requesting go-ahead on the above before writing any of the three
pieces.
