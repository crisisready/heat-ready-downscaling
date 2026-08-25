# Rung D: contributed data, and what "reproducible" can honestly mean for it

**Design only. No implementation.** Rung D decides how private sensor data meets a
public-reproducibility rule, which is a trust-model decision rather than a coding one.

**Revision note (2026-08-25, after round-1 review).** An earlier draft of this document reached
three conclusions that review showed were wrong, and cited one precedent that does not exist.
They are corrected below rather than quietly edited, because a design record whose errors are
invisible is worth less than one that shows where it was wrong: Seoul is **not** the easy case,
`promote_from_public.py` **cannot** be reused as-is, the shared-QC-harness argument was
**refuted by its own evidence**, and the "internal-track promotions stay out of the credit
ledger's tenure path" precedent I cited repeatedly **was never real** — it was my own unbuilt
proposal from an earlier design doc, cited later as though it were established practice.

## The tension, stated exactly

The referee's trust model is three things: a submission runs **this repository's own code**, at a
**pinned version**, against a **published snapshot**. Any stranger can re-derive any number
because GHCN-Daily is free and public.

Contributed sensor data breaks **one** of those three: the published snapshot. It does not touch
what code runs or at what version — Rung D admits *data*, and contributed code still never
executes (see "Not proposed" below). An earlier draft said it "breaks all three legs at once",
which was overstatement that made the problem look larger than it is; the three items it listed
underneath were all restatements of the snapshot leg.

That one leg is enough to matter:

1. The data is not in the published snapshot.
2. It may not be redistributable at all — municipal agreements and commercial feeds routinely
   forbid it.
3. Even when redistributable, a curated snapshot is not a place to dump every city's raw feed.

So: **what does reproducibility mean for a result nobody outside the project can recompute?**

## The move: separate reproducibility from publication

"Reproducible" bundles three things that public data delivers together:

- **(a) Independent re-derivation** — the maintainer, not the claimant, recomputes the number.
- **(b) Third-party falsifiability** — someone unaffiliated can check the claim.
- **(c) Identification** — what was used is pinned precisely enough to settle a dispute.

Contributed data can give **(a)** and **(c)**. It cannot always give **(b)**. Saying so is better
than implying otherwise.

**(a) needs new code, and an earlier draft wrongly said it did not.** The private serving
repository's `scripts/promote_from_public.py` (`crisisready/heat-risk-data-api`) does perform
independent re-derivation today, but it is bound to the snapshot layout: `main()` calls
`verify_frozen_snapshot_pin()`, which hard-exits unless a `MANIFEST.json` in
`--frozen-snapshot-dir` declares exactly the winning cycle's `snapshot_version`, and `rederive()`
then reads through `snapshot.read_band_partitions()` and `contract.FrozenPredictionAdapter`. A
private sensor store is neither a versioned snapshot nor in that layout. **Every
`no-redistribution` claim rests on a re-derivation path that does not exist yet.** Naming that is
the point: assuming a mechanism exists when it does not is the error the licensing gate (#31) was
built to correct, and this document made it about the very step its trust model depends on.

## Three tiers, keyed to the axis the licensing gate already defines

`redistribution_tier`, from #31, is the axis that determines what reproducibility is achievable.

| Tier | Enters a published snapshot? | What "reproducible" means |
|---|---|---|
| `unrestricted` | Yes | Full public reproducibility |
| `attribution-required` | Yes, notice carried forward | Same |
| `no-redistribution` | **No** | Maintainer-verifiable and auditable, not publicly recomputable |

### What `no-redistribution` buys, stated more carefully than before

The data lands in a private store, the maintainer re-derives (see the dependency above), and what
publishes is a **content-addressed attestation**.

An earlier draft claimed the attestation proved four things. It proves two. Hashes over the
*submitted* files show that the claimant did not alter the data **after** claiming, and that the
maintainer consumed the same bytes. They cannot show that the QC bar was met on the whole feed,
or that the sensor set was not trimmed to flatter the result — because **the excluded bytes are
never hashed or published.** The concrete attack: pull all 1,044 sensors, compute the result over
subsets, publish ids and hashes for the 744 that help. The maintainer re-derives the same number
from the same bytes and every hash verifies.

Closing that requires the attestation to cover the **full pre-QC pull plus the exclusion set with
its reasons**, not just the surviving rows. That is a real design requirement, not a detail, and
it is what makes the tier auditable rather than merely tamper-evident.

**Any cell resting on this tier is labelled.** Two surfaces named for that labelling do not exist
in this repository yet: there is no `registry/` and no public models page — both are roadmap
Phase 1 deliverables. `docs/leaderboard.{md,json}` is the only surface that exists today, so it
is the only one this design can commit to.

## What a Rung D submission is

A **data-source dossier**. The checklist is §0 of
`research/seoul-local-sensor-validation/REPRODUCE_FOR_A_NEW_CITY.md` in the private
`crisisready/heat-risk-data-api` repo. **It has to be restated in this public repository to be
binding** — a requirement a contributor cannot read is not a requirement — so the items are
reproduced here in full, and implementation should move them into `CONTRIBUTING.md` rather than
linking out.

Its provenance, accurately: it was **distilled from** the Chicago and NYC rejections after the
fact, and its own source describes it as "the checklist that would have caught both faster". It
has never actually gated a city. An earlier draft said it had "already earned its keep", which
credited it with work it has not yet done.

1. **Live-endpoint proof.** Hit the actual data endpoint, not the dataset's landing page.
2. **Sensor count for the variable you actually need.** Chicago advertised 286 sensors; 7
   reported temperature.
3. **Currency.** A recent real timestamp, not a frozen campaign. NYC's Hyperlocal Temperature
   Monitoring is 475 sensors and ends in 2019.
4. **Self-serve access.** Not request-and-wait with a possible fee — the NYC-area NYS Mesonet
   subset (29 stations) is gated exactly this way.
5. **A bulk path**, in addition to any documented API. Seoul's usable route was a CSV archive
   sitting beside a gated "Open API".
6. **Licence and redistribution tier**, through the #31 gate. Two frictions worth naming now:
   `ODbL-1.0`, `CC-BY-SA-4.0` and `CC-BY-SA-3.0` route to a maintainer rather than auto-passing,
   because share-alike conflicts with `DATA_LICENSE`'s CC BY 4.0 republication (older `CC-BY-SA`
   versions are not in that set at all, so they are outright rejections). And
   `reproducible_fetch` is a **required** key regardless of tier, which a `no-redistribution`
   source in a private store cannot honestly populate — so the first such dossier either fails
   admission or declares a decorative URL. That needs resolving before the tier can ship.
7. **A QC report** from the harness below.

## QC: the argument I made was wrong, and the corrected one is narrower

The earlier draft argued the harness must be **ours** because the S-DoT QC script produced a
false "0 of 60,919 sensor-days usable". Review pointed out the flaw, and it is fatal to that
argument: **that script *was* ours.** A shared harness would have shipped the identical
row-order bug to every contributor and produced the identical false rejection. "We run it, the
contributor runs it too" is two executions of the same code and catches nothing about a bug in
that code — a single point of failure dressed as a safeguard.

What the incident actually argues for is an **independent cross-check**, which the harness alone
cannot be. Three things, honestly separated:

- **A shared harness is still worth having**, but for *comparability* — every dossier's QC report
  means the same thing — not for correctness.
- **A null result requires a second method before it is accepted.** "This source is unusable" is
  the conclusion the S-DoT bug would have produced, and it is the conclusion nobody
  double-checks. Any dossier rejected on QC grounds gets an independent recount before the
  rejection stands.
- **Cadence is derived and cross-validated**, never assumed — the specific bug was inferring a
  10-minute cadence from time-then-sensor row order, and it is caught by checking one sensor's
  own timestamp sequence against the file-wide assumption.

The rest of the harness generalises what S-DoT's QC did catch: physical-plausibility bounds,
per-sensor-day completeness against the measured cadence, any cross-check field the feed offers
(black-globe below ambient flagged 20 badly-sited sensors of 1,044), and **join-key
normalisation against the boundary polygons** — the defect that matched only 101 of 406 dong
where 319 of 406 should have matched, understating coverage roughly threefold. (Distinct from the
post-QC figure of 315 of 423 dong having usable ground truth; an earlier draft conflated the
two.)

## Both open decisions, as decided by Nishant (2026-08-25)

**Reaffirmed under the corrected facts**, after round-1 review reversed this document's
sequencing conclusion. Both calls stand, and the reversal strengthens rather than weakens the
first: with Seoul landing in a labelled tier rather than sailing through as `unrestricted`, the
labelled-tier path stops being an edge case for awkward municipal partners and becomes the
ordinary route that the flagship city itself takes. The label has to be good because the best
dossier we have will carry one.

**1. `no-redistribution` data is admissible on the public leaderboard, with the visible label.**
My recommendation, adopted.

**2. Consumer networks (Netatmo-class) are admitted.**

Recorded precisely, because an earlier draft misstated the governance change: this is **not** an
override of the approved roadmap. `ROADMAP.md` already says consumer sources are "admissible
only with a validated bias treatment — the default expectation, given the record, is rejection."
The decision is that the door is open; the conditions below are that stated precondition made
concrete. Calling it an override overstated what changed, in a section whose whole job is
recording the change accurately.

## Consumer networks: the conditions that make an open door safe

Consumer stations fail systematically, not randomly: sun exposure reads warm by day, balcony and
indoor siting suppresses diurnal range, wall and vent proximity adds persistent night-time warm
bias.

1. **`source_class` declared, not inferred** — `institutional | consumer`. The QC bar, the
   weighting and the labelling differ, and inferring it from the endpoint is the kind of silent
   guess this program keeps getting caught by.
2. **A siting screen, not just plausibility.** Bounds catch −40 °C in July; they do not catch a
   sensor in direct sun. Cohort-relative daytime warm bias, suppressed diurnal range, persistent
   night-time offset, correlation breakdown against neighbours.
3. **Density is a precondition**, because every screen in (2) is cohort-relative. A sparse
   consumer feed cannot be screened this way and is not admissible as a consumer source. This is
   the condition most likely to be narrower than "open" was meant to be, and the honest cost of
   relaxing it is admitting data whose bias cannot be characterised.
4. **The bias treatment validates against independent institutional reference data.** The
   methodological crux: a correction fitted and judged on consumer data alone looks excellent and
   means nothing, because the reference carries the same bias. A city with no institutional
   reference can have its consumer network described, not validated.
5. **Weighting is explicit**, declared in the dossier and recorded in the attestation.

### Condition 5's evidence, corrected

An earlier draft justified (5) with the Valencia widening result and got the mechanism wrong.
That degradation was caused by **geographic and microclimate spread** — stations up to 115 km
away and 1,515 m elevation, genuinely different mountain interiors — not by count imbalance; 31
against 9 is barely 3×. And 5,000 consumer sensors *inside one city* is close to the opposite
case: same microclimate, more samples. The draft also omitted that restricting the wide set to
hot days **recovered** significance for both targets (tmax 18.2% CI [8.5, 26.6]; tmin 9.3% CI
[2.8, 13.7]).

So Valencia does not establish (5). The honest justification is narrower: a pooled fit in which
influence tracks sensor count lets the least-controlled instruments dominate a cell's estimate,
and (2)–(4) exist precisely because consumer instruments are the least controlled. That is an
argument from bias, not from dilution, and it should be tested rather than asserted — a
with-and-without-weighting arm on the first dense consumer city is the measurement that would
settle it.

### An unresolved conflict with standing policy

`ROADMAP.md`'s data-sourcing policy requires that sources carrying device-owner locations —
naming consumer crowd networks as the known case — be admissible "only after
anonymization/aggregation that removes it, stated in the QC report."

**Conditions 2 and 3 require precise per-sensor geolocation at sub-neighbourhood granularity,
which is exactly the information that policy requires stripped.** As written, a Netatmo-class
dossier cannot satisfy both gates. This is a real conflict, not a wording problem, and the
earlier draft did not notice it.

The resolution I would propose, for review rather than as a decision: screening happens
**maintainer-side on coordinates that are never published**, and the dossier and attestation
carry only aggregates — per-cohort bias statistics, counts, and the exclusion set by opaque
sensor id. That keeps the screens computable while nothing location-bearing is republished. It
also means a consumer dossier cannot be independently re-screened by a third party, which is a
further narrowing of leg (b) and should be labelled as such.

## Labelling, and a pattern named rather than extended

This design now needs two provenance labels: `no-redistribution` and consumer-derived. Two is
where a pattern should be named instead of extended a third time.

Proposal: **one evidence-provenance label per cell** — publicly reproducible, maintainer-attested,
consumer-derived, or a combination — rather than accumulating one-off badges. A reader deciding
whether to trust a number needs to know which kind it is without reading the manifest.

An earlier draft justified this by analogy to "the same reasoning that keeps internal-track
promotions out of the credit ledger's tenure path." **That precedent does not exist.** There is
no internal track in `GOVERNANCE.md`, `ledger/README.md` or `ledger.py`'s `CREDIT_LINE_SCHEMA`;
it was a proposal in my own earlier design document, cited in later documents as though it were
established practice. The argument stands on its own without a fabricated precedent, and the
fabrication is recorded here because it appeared in more than one place.

## Sequencing: the earlier conclusion was wrong

The earlier draft concluded that S-DoT was `unrestricted`-tier, so "Seoul does not need the hard
case" and "the tiers can land in order."

**Both are wrong**, and the correction has to be made carefully, because the failure here was
asserting a licence position with nothing behind it — replacing it with a differently-unsupported
one would repeat the error in the opposite direction.

### S-DoT's licensing position, separated by what is actually known

**Established.** Nothing in either repository records S-DoT's licence. Not the QC findings, not
the comparative analysis, not the corpus builder, not the reproduce-for-a-new-city recipe. The
research thread used the data without ever pinning its terms, which is unremarkable for research
and disqualifying for admission.

**Established.** Under #31's rules, `unrestricted` is a narrow tier:
`SPDX_ATTRIBUTION_REQUIRED` covers every allowlisted licence except `CC0-1.0` and `PDDL-1.0`, so
declaring `unrestricted` for anything attribution-bearing is a hard rejection, not a judgement
call.

**Inference, and labelled as one.** Datasets published through `data.seoul.go.kr` are typically
issued under the Korea Open Government Licence, whose common Type 1 is attribution-required. KOGL
has no SPDX identifier, so it is not on `SPDX_ALLOWLIST` and cannot auto-pass regardless of type.
That points at `attribution-required`, or at the `proprietary-licensed` needs-review path with
KOGL named as the licensor. **This is a portal-level expectation, not a verified fact about the
S-DoT dataset specifically, and no admission decision may rest on it.**

**Therefore a prerequisite, not an assumption.** Establishing S-DoT's actual licence and type is
step one of the dependency order below — a real task with a real possible outcome that the
expectation above is wrong. What is safe to say now is only the negative: **`unrestricted` is not
available**, so Seoul's dossier goes through a labelled tier whichever of the remaining paths it
takes.

`attribution-required` is the tier this document says cannot ship until something generates the
snapshot's attribution notices from the field, since #31 validates `redistribution_tier` and
`attribution_required` but nothing reads them. **So the first dossier is blocked on exactly the
work the earlier draft deferred**, and the real dependency order is:

1. Establish S-DoT's actual licence and type from the source, not from portal convention.
   Nothing in either repo records it, and the KOGL expectation above is an inference.
2. Build the attribution-notice mechanism that consumes `redistribution_tier`.
3. Build a re-derivation path for data outside the snapshot layout (the `promote_from_public.py`
   gap above).
4. Then the tiers, in whatever order their evidence allows.

## Not proposed

No change to what the referee executes. `GOVERNANCE.md` says "No contributor Python executes
anywhere **in v1**" — quoted with its scope qualifier, which an earlier draft dropped; the v1
scoping is load-bearing, since Rung C is a planned rung whose opening is explicitly conditioned
on an execution-safety decision. A Rung D dossier is a declaration plus a QC report, not a
program.

Also out of scope: feeding contributed data into the **global** training corpus, which the
roadmap stages as its own separately-gated path.
