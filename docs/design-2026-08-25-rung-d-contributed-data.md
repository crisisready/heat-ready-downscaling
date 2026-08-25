# Rung D: contributed data, and what "reproducible" can honestly mean for it

**Design only. No implementation in this PR, by instruction.** Rung D decides how private
sensor data meets a public-reproducibility rule, which is a trust-model decision rather than a
coding one, so it wants Nishant's review before anything is built.

## The tension, stated exactly

The referee's whole trust model is three sentences in `GOVERNANCE.md`: a submission declares
what to run **from this repository's own code**, at a pinned version, against a **published
snapshot**. No contributor Python executes anywhere. Any stranger can re-derive any number.

That works because GHCN-Daily is free and public. Contributed sensor data breaks all three
legs at once:

1. It is not in the published snapshot.
2. It may not be redistributable at all — municipal agreements and commercial feeds routinely
   forbid it.
3. Even when redistributable, a curated snapshot is not a place to dump every city's raw feed.

So the question Rung D has to answer is not "how do we accept data" but: **what does
reproducibility mean for a result nobody outside the project can recompute?**

## The move: separate reproducibility from publication

What the program actually needs from "reproducible" is three distinct things, which public data
happens to deliver together:

- **(a) Independent re-derivation** — the maintainer, not the claimant, recomputes the number.
  `promote_from_public.py` already does this and does not care where the data came from.
- **(b) Third-party falsifiability** — someone unaffiliated can check the claim.
- **(c) Identification** — what exactly was used is pinned precisely enough that a dispute is
  resolvable.

Public data gives all three, which is why it stays the default and the preferred path.
Contributed data can always give **(a)** and **(c)**. It cannot always give **(b)**. The
honest design says so out loud rather than implying otherwise.

## Three tiers, keyed to the axis the licensing gate already defines

The `redistribution_tier` field that landed with the licensing gate (#31) turns out to be exactly
the axis that determines what reproducibility is achievable. That is not a coincidence worth
hiding — it means Rung D needs no new vocabulary for the hard part, and it is why the licensing
gate had to be built first.

| Tier | Can it enter a published snapshot? | What "reproducible" means |
|---|---|---|
| `unrestricted` | Yes | Full public reproducibility. Any stranger recomputes the number. |
| `attribution-required` | Yes, with attribution carried forward | Same as above. `DATA_LICENSE` already does this for LandScan/ORNL. |
| `no-redistribution` | **No** | Maintainer-verifiable and publicly **auditable**, but not publicly recomputable. |

### What `no-redistribution` actually buys

The data lands in a private store. The maintainer re-derives the claim against it. What gets
published is a **content-addressed attestation**: the exact station/sensor ids and date range
used, a per-file `sha256`, row counts, and the full QC report — plus the derived numbers.

A third party cannot recompute the result. They **can** verify that the claimant did not alter
the data after claiming, that the QC bar was met, that the maintainer's re-derivation consumed
the same bytes, and that the sensor set was not quietly trimmed to flatter the outcome.

**One dependency this exposes, and it is not hypothetical.** `#31` validates
`redistribution_tier` and `attribution_required` but nothing yet *consumes* them — no code
generates the snapshot's attribution notices from the tier. An attestation for
`attribution-required` data has to carry that notice forward, so Rung D cannot ship its
`attribution-required` path until something reads the field. Named here rather than discovered
during implementation, because assuming a mechanism exists when it does not is the specific
error `#31` was written to correct and then made three more times inside itself.

That is tamper-evidence plus independent verification. It is genuinely weaker than public
reproducibility, and the difference is not cosmetic, so:

**Any cell whose evidence rests on `no-redistribution` data is labelled as such — in the
registry, on the public models page, and on the leaderboard.** Without that label the program
silently merges "anyone can check this" and "trust us, we checked" under one badge. This is the
same reasoning that keeps internal-track promotions out of the credit ledger's tenure path:
the record's value comes from the categories staying distinct.

## What a Rung D submission is

A **data-source dossier**, not a model and not a parameter. The acceptance checklist is
`REPRODUCE_FOR_A_NEW_CITY.md`'s §0, promoted from a research habit to a submission
requirement — and it is worth promoting precisely because it has already earned its keep:
it is what rejected Chicago and NYC after each looked fine on paper.

1. **Live-endpoint proof.** Hit the actual data endpoint, not the dataset's landing page.
2. **Sensor count for the variable you actually need.** Chicago advertised 286 sensors; 7
   reported temperature.
3. **Currency.** A recent real timestamp, not a frozen campaign. NYC's Hyperlocal Temperature
   Monitoring is 475 sensors and ends in 2019.
4. **Self-serve access.** Not request-and-wait with a possible fee (NYS Mesonet's 29 real
   stations are gated exactly this way).
5. **A bulk path**, in addition to any documented API. Seoul's usable route was a CSV archive
   sitting beside a gated "Open API".
6. **Licence and redistribution tier**, through the gate that now exists (`#31`). Worth
   anticipating for municipal feeds specifically: `ODbL-1.0` and the `CC-BY-SA` family now route
   to a maintainer rather than auto-passing, because share-alike terms conflict with
   `DATA_LICENSE`'s CC BY 4.0 republication. Open government data under ODbL is common, so a real
   fraction of Rung D dossiers will land in that review path by design rather than by accident.
7. **A QC report** from the shared harness below.

## The QC harness, and why it must be ours rather than the contributor's

`dataqc.py`, generalising the S-DoT QC steps:

- **Cadence verified, never assumed** — derived from one sensor's own timestamp sequence.
- Physical-plausibility bounds for the city's real climate and season.
- Per-sensor-day completeness against the **measured** cadence, not a guessed one.
- Any cross-check field the feed offers (S-DoT's black-globe vs ambient caught 20 badly sited
  sensors).
- **Join-key normalisation against the boundary polygons.**

The argument for a shared harness rather than "run your own QC and report it" is not
convenience, it is a real incident. The S-DoT QC script itself got the cadence wrong on its
first pass — it read the CSV's time-then-sensor row order as one sensor at 10-minute
intervals — and produced a headline **"0 of 60,919 sensor-days usable"**. That is a false
*rejection* of a data source that turned out to be excellent. A contributor running bespoke QC
would have reported it and walked away, and we would have believed them. The same thread's
whitespace-sensitive dong join separately understated Seoul's real coverage by about 3×
(101 dong instead of 315).

Both errors were in the direction of wrongly *discarding* good data, which is the failure mode
a contributor has no incentive to catch and every incentive to accept. We run the harness; the
contributor runs it too; both reports attach to the submission.

## What this unblocks, and what it does not

Rung D is a hard dependency for Seoul: nothing today can independently re-derive
`local/seoul-sdot-v1`, and the program's own "a submission's own reported numbers are never
trusted" rule cannot be honoured for it until this exists. S-DoT is the natural first dossier
and is `unrestricted`-tier as far as the record shows, which means **Seoul does not need the
hard case** — it can go through the fully-reproducible path.

That matters for sequencing: the `no-redistribution` machinery is the part that needs the most
review and the least urgency. The tiers can land in order.

## What I am NOT proposing

No implementation, per the constraint on this piece. Also deliberately absent: any change to
what the referee executes. Rung D admits *data*, and contributed code still never runs —
`GOVERNANCE.md`'s "no contributor Python executes anywhere" stands untouched, and a Rung D
dossier is a declaration plus a QC report, not a program.

## Two decisions that are Nishant's, not mine

1. **Is `no-redistribution` data admissible at all for a cell that appears on the public
   leaderboard?** My recommendation is yes, with the label above, because the alternative
   excludes most municipal partners — and a labelled weaker guarantee is more useful than a
   missing city. But it is a genuine trust-model choice about what the leaderboard means, and
   the conservative answer (private cells only, never on the public board) is defensible.
2. **Consumer-network sources** (Netatmo-class). The approved roadmap says admissible only
   with a validated bias treatment, and that the default expectation given the record is
   rejection. Worth confirming that stands rather than inheriting it silently, since it is the
   difference between "a few municipal partners" and "a very large amount of noisy data".

## Explicitly out of scope here

Feeding contributed data into the **global** training corpus. The roadmap already stages that
as its own separately-gated path with a stricter bar, and folding it in here would bundle a
much larger decision — a global retrain's behaviour — into a data-acceptance design.
