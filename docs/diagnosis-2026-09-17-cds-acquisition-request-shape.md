# Diagnosis: why the Af/Am ERA5-Land corpus pull crawled for 48 hours

**Date:** 2026-09-17
**Scope:** `scripts/build_training_set.py`'s CDS ERA5-Land fetch path (and the private-repo
`era5.py` it calls). Written in response to a direct instruction to start from the assumption
that the fault is ours, not CDS congestion.

> "We're doing something wrong. Start with that assumption and then figure out how we fix our
> CDS calls." — and — "you have three CDS keys. most orgs only have 1. it makes no sense to say
> that these orgs are usually just not trained."

Both premises hold up. The three keys are three genuinely distinct CDS accounts, and the
dominant cost is our own request shape, not the global queue.

---

## 1. The measured symptom

From the live bastion lane log (`run_country_US_lane1.log`, 2026-09-17):

```
01:53:06  [US_Af_c0] ERA5 fetch starting (source=cds)
...
03:50:31  era5: Downloaded+merged ERA5 across 14 calendar-month segment(s) to /tmp/tmpjjtldq70.nc
03:50:32  [US_Af_c0] ERA5 fetch finished in 7046.1s
03:50:33  ghcn: Upserted 338 ghcn_training row(s)
```

**117 minutes of ERA5 fetch to produce one station-year (338 rows).** And that was the *lucky*
case — see §2.3.

Corpus state after ~48 h of pulling (prod SQL, 2026-09-17T13:2x): US/Af 2,708 rows / 8 stations;
VM/Am 40 rows / 1 station; VQ/Af 356 rows / 1 station; ID, KE, SL 0 rows. Ten stations.

## 2. Root cause: request *count*, not request *size*

Three multipliers stack. The first two are ours.

### 2.1 Geographic fan-out — the clearest defect

`_chunk_stations_by_extent` buckets stations into `_ERA5_MAX_CHUNK_EXTENT_DEG = 0.5` degree
cells, and each bucket gets its own complete ERA5-Land pull. `station_ids_country_US_lane1.json`
holds 5 stations; the log shows them processed as `US_Af_c0`, `US_Af_c1`, … each
`building training rows for 1 station(s)`. **Five stations became five independent pulls.**

That constant exists, per its own comment, to stay under "CDS's own per-request cost limit."
But `era5._split_by_calendar_month`'s docstring already records the live 2026-08-03 finding
that refutes this:

> items = variables x levels x timesteps, **which has no area/grid-point term at all: chunking
> the bounding box or station count, tried first, could never have fixed this**

[verified: ECMWF CDS documentation gives ERA5-Land hourly a **12,000 field** per-request limit,
and a MARS "field" is one variable x level x timestep — subsetting the area does not change how
many fields a request names.]

So on 2026-08-03 two mechanisms were added for one symptom: calendar-month splitting, which
genuinely fixes the field-count overrun, and bbox/station chunking, which by that same finding
*cannot* affect it. The second was kept anyway. It does not reduce cost; it multiplies request
count by the number of 0.5-degree cells the station set occupies.

Size confirms volume is not the *current* constraint: `US_Af_c0`'s bbox is one station padded
0.5 degree (~100 ERA5-Land cells), and the observed segment payloads are **~1.0 MB each**. That
does **not** mean bytes are free at any bbox size — see 5.1 for the real ceiling.

**One recorded observation in this codebase contradicts the claim above, and it has to be
addressed rather than ignored.** `_ERA5_MAX_CHUNK_EXTENT_DEG`'s own comment states:

> confirmed live 2026-08-03 that `_group_stations_by_zone` alone is NOT sufficient: 2 same-zone
> … stations (raw spread ~1.4deg lat x 0.33deg lon) still tripped CDS's "cost limits exceeded"
> rejection once padded to a 2.37deg x 1.33deg bbox — padding, not station spread, was doing
> most of that damage.

Either the field-only cost model is incomplete, or that rejection was mis-attributed. The
mis-attribution reading is the more likely one: the calendar-month split landed *the same day*
against *the same symptom*, and the request it fixed had 34 real days cartesian-expanded into 93
day-slots = 13,392 fields at 6 variables — over the limit on the date axis alone, with the bbox
irrelevant. But "more likely" is not measured, so this is treated as an open question and is the
first thing the probe in 5.6 settles.

### 2.2 Calendar-month fan-out

`_ERA5_CHUNK_DAYS = 400` puts a training year in one outer chunk; the +/-2-day padding widens
2023 to 2022-12-30..2024-01-02, which `_split_by_calendar_month` cuts into **14 segments**
(Dec 2022, Jan–Dec 2023, Jan 2024) — exactly what the log reports. So 14 serialized CDS
requests per station-chunk per year.

This is over-conservative against the real budget — but by less than a first pass suggests,
and the correction matters. CDS's ERA5-Land hourly per-request limit is **~12,000 fields**
(the codebase calls it `~12,000`; ECMWF documents 12,000, and the only *observed* data points
are 13,392 rejected and 4,464 accepted, so treat it as approximate). A field is one
variable x level x timestep.

`_build_era5_request` builds **three** independent lists — `year`, `month`, `day` — so the
binding quantity is

```
fields = n_vars x 24 x (n_years x n_months x n_days_in_union)
```

**The `n_years` factor is easy to miss and it is decisive.** The padded 2023 window is
2022-12-30..2024-01-02, so the first and last segments of every training year straddle a year
boundary. A segment that does so multiplies its day-slots by 2:

| segment | years x months x days | slots | 6 var | 5 var | 3 var |
|---|---|---|---|---|---|
| 1 calendar month (current) | 1x1x31 | 31 | 4,464 ok (37% of budget) | 3,720 ok | 2,232 ok |
| Jan–Feb 2023 | 1x2x31 | 62 | 8,928 ok | 7,440 ok | 4,464 ok |
| **Dec 2022 + Jan 2023** | **2x2x31** | **124** | **17,856 REJECTED** | **14,880 REJECTED** | 8,928 ok |
| Jan–Mar 2023 | 1x3x31 | 93 | 13,392 REJECTED | 11,160 ok | 6,696 ok |
| Jan–May 2023 | 1x5x31 | 155 | 22,320 REJECTED | 18,600 REJECTED | 11,160 ok |

So segments may be widened but **must never cross a calendar-year boundary** — naively pairing
the 14 months into 7 would put `(Dec 2022, Jan 2023)` at 17,856 fields and earn exactly the
`cost limits exceeded` rejection `_split_by_calendar_month` was ported forward to fix.

Year-aligned minimum request counts per bbox-year, with the two padding months necessarily
standing alone:

| variables | max months/segment | segments per padded year |
|---|---|---|
| 6 (today) | 2 | 6 + 2 = **8** |
| 5 (drop unused ssrd, see 3.3) | 3 | 4 + 2 = **6** |

A naive "total fields / limit" floor (`6 x 24 x 369 = 53,136` → 5 requests) is **wrong**: it
ignores the cartesian quantization above. At 6 variables no request can exceed 2 months, so the
achievable floor is 8, not 5.

### 2.3 Queue latency — the one genuinely CDS-side term

Timestamped segment downloads in the lane1 pull: 10 segments returned inside 43 seconds, then
the remaining 4 took **31, 35, 35 and 16 minutes**. The fast 10 were CDS *server-side* cache
hits — their URLs are `cci2-prod-cache-*/2026-09-16/…`, i.e. results CDS still had materialised
from the previous day's run. That is luck, not design.

So the real cost of an uncached segment is roughly **half an hour of wall clock to move ~1 MB**
(n=4; indicative, not a precise mean). 14 x ~33 min ≈ 7.7 h per station-year with a cold cache.

The global queue is genuinely congested — CDS reports `running: 460, queued: 7003` against
"the maximum number of requests that access the CDS-MARS archive is 460". **That 7003 is CDS's
system-wide backlog, not ours**; our own per-account depth is 4–6. PR #46 had already reached
this same conclusion. Global congestion sets the ~33 min per-request price; our request count
is what turns that price into 48 hours.

## 3. Three further defects found on our side

### 3.1 The segment cache is inert in every running lane

PR #42 added `--era5-cache-dir` and PR #46 gave each calendar-month segment its own cache entry.
**None of the 7 live lanes pass `--era5-cache-dir`** [verified: process args via SSM `ps`, plus
`argparse` `default=None`]. So every `_remaining` / `_retry` relaunch re-pays every CDS
round-trip. The lanes are named for having been relaunched repeatedly.

### 3.2 CDS jobs are never dismissed, and the lock's scan order load-imbalances the accounts

Two separate things here. One is certain; one was initially overclaimed and is downgraded.

**Certain: nothing ever dismisses a CDS job.** There is no `delete`/`dismiss` call anywhere in
`era5.py`. Every account currently holds 4–6 jobs in `accepted` state, never started, up to
**83 minutes** old, while `_era5_download_lock` should permit only one in-flight request per
account. That is real hygiene debt and we should dismiss jobs on exit regardless of anything
else.

**Downgraded: that these specific jobs are queue-poisoning orphans is NOT established.** The
original version of this section argued from a FIFO violation — on every account, jobs created
*after* the oldest still-`accepted` job had already completed (6, 13 and 15 of them) — that the
old ones must be abandoned. That inference does not hold, and section 2.3 of this same document
is what breaks it: CDS serves an already-materialised result in seconds (the
`cci2-prod-cache-*` hits), and 3.1 establishes that no lane passes `--era5-cache-dir`, so a
large fraction of submissions are *identical resubmissions* that come back as cache/dedup hits
immediately while a genuinely novel request queues behind the global backlog. Later-finishes-
first is then expected, with no orphaning. Two further alternatives also fit: CDS's QoS is
cost-weighted rather than FIFO, and these accounts are shared with production's Lambda pipeline
(`manager.cds_request_lock`), so an `accepted` job need not belong to a training lane at all.

**A simpler, better-supported explanation for the rejection asymmetry.** One account — the one
`_era5_download_lock`'s free-slot-first scan reaches first — reports `queued: 5, running: 0`
against `"The maximum number of per-user requests that access the CDS-MARS data is 1"`, and
**86 of its last 200 jobs were rejected (43%)**, against 0/200 on each of the other two. The
lock scans `configured` in index order and takes the first free slot, so whenever concurrency is
1 (the normal case for a single lane) account 0 is chosen *every time* and accounts 1 and 2 are
never touched at all. That alone explains the asymmetry without any orphan theory, and it points
at a different and more valuable fix: **rotate or randomise the scan start order** so the three
accounts we actually have get used evenly.

The honest conclusion: part of issue #648's "transient congestion" framing is likely
self-inflicted, but via load-imbalance rather than via proven orphaning, and dismissing jobs is
hygiene we should do anyway.

### 3.3 One of the six requested variables is never used

`_TRAINING_ERA5_VARIABLES` requests `surface_solar_radiation_downwards` (ssrd). But
`downscaling.FEATURE_ORDER` has no solar term, and `build_rows_for_country` builds rows carrying
only `grid_tmax_c`, `grid_tmin_c`, `grid_specific_humidity_kgkg` and `nighttime_wind_ms`. **ssrd
is downloaded on every training request and discarded** — 1/6 of every request's field cost.

Dropping it raises the per-request budget from 83 to `12000 / (5 x 24) = 100` day-slots, which
makes 3 full calendar months (93 slots) fit — taking 14 segments/year to **6** (four triples
plus the two year-straddling padding months standing alone; see 2.2 for why they cannot be
merged).

The waste is slightly larger than 1/6 of the field cost: requesting ssrd also flips
`heat_calcs.aggregate_hourly_to_daily` onto its `has_thermal` branch, so every training run
pays a full UTCI/WBGT pass over every hourly row for output no row ever carries. Note that
`tests/test_build_training_set.py` currently *asserts* ssrd is requested, so removing it is a
test change too.

## 4. Can a no-queue mirror replace CDS for these inputs?

The code needs ERA5-**Land** (0.1 degree) hourly `t2m`, `d2m`, `sp`, `u10`, `v10` at station
points. Assessed against that requirement, not against "some ERA5":

| Candidate | ERA5-Land 0.1 deg? | All needed vars? | Usable today? |
|---|---|---|---|
| Google ARCO-ERA5 (`gs://gcp-public-data-arco-era5`) | **No** — ERA5 0.25 deg only | n/a | anonymous, but wrong resolution |
| AWS `era5-pds` / Microsoft Planetary Computer | **No** — same ERA5 0.25 deg source | n/a | wrong resolution |
| Open-Meteo archive, `models=era5_land` (paid key held, already wired) | Yes | **No** — `t2m`/`d2m` only | yes, but incomplete |
| DestinE Earth Data Hub ERA5-Land Zarr | Yes | **Yes** — all present | **no credential**; HTTP 401 |
| Google Earth Engine `ECMWF/ERA5_LAND/HOURLY` | Yes | Yes | no GCP/GEE credential held |

Two findings matter here.

**The 0.25-degree mirrors are disqualified on resolution, not convenience.** The model's target
is the ERA5-Land grid value it downscales, and production serves ERA5-Land. Training on 0.25
degree ERA5 while serving 0.1 degree ERA5-Land would be a training/serving mismatch of exactly
the kind issue #1 exists for.

**Open-Meteo's gap is real and is not our bug.** Re-verified live today at a Jakarta Af point,
48-hour window, both legacy (`dewpoint_2m`, `windspeed_10m`) and current (`dew_point_2m`,
`wind_speed_10m`) spellings:

| request | t2m | d2m | wind | sp | radiation |
|---|---|---|---|---|---|
| `models=era5_land`, legacy names | 48/48 | 48/48 | **0/48** | **0/48** | **0/48** |
| `models=era5_land`, current names | 48/48 | 48/48 | **0/48** | **0/48** | **0/48** |
| no `models` (best_match) | 48/48 | 48/48 | 48/48 | 48/48 | 48/48 |
| `models=era5` / `era5_seamless` | 48/48 | 48/48 | 48/48 | 48/48 | 48/48 |

A "we asked with the wrong variable names" hypothesis was tested and **disproved**. The existing
docstring was accurate: Open-Meteo simply does not serve `u10`/`v10`/`sp` under a pinned
`era5_land` model. Falling back to `best_match` would silently substitute a non-ERA5-Land blend.

Open-Meteo is, however, **~0.5 s per station-request** against CDS's ~33 min.

**Conclusion for exit criterion 1:** no no-queue mirror of genuine ERA5-Land is usable with
credentials we hold today. DestinE is the only complete candidate and needs a key (its chunk
geometry and throughput are therefore unverified — it 401s even on `.zmetadata`).

## 5. Implications for the fix

The levers, in descending size, all independent of anything CDS decides.

### 5.1 Stop splitting by 0.5-degree cell — but bound the bbox by VOLUME, not by CDS cost

Cluster stations into a bounded number of compact bboxes instead of one per occupied 0.5-degree
cell. The field-limit argument says this costs nothing in CDS's cost model.

**It does, however, cost bytes and memory, and there is a real ceiling that an earlier draft of
this document wrongly waved away as "bytes are ~free at this scale".** Four separate O(area)
costs:

- `era5.extract_era5_means` reads `ds[var].values[:, lat_idx, lon_idx]` — `.values` materialises
  the **whole** variable array before the fancy-index, so peak memory is
  `n_times x n_lat x n_lon x 8 bytes` **per variable**, regardless of how few stations are
  wanted. At ~1 degree that is a few MB; at 10 degrees ~0.7 GB/variable; at 25 degrees several
  GB/variable across 5-6 variables. This script's own module docstring already cites the
  shared-cgroup OOM this class of job has hit before.
- This repo's own measured number: *"a full-CONUS, one-month pull is ~3.5 GB regardless of how
  many of the 78,566 US stations are actually selected"* (2026-07-18). A 25-degree bbox is
  therefore ~1.5 GB **per month segment**, ~20 GB per bbox-year.
- `era5._merge_era5_segments` does an in-memory `xr.concat` across every segment — which is
  exactly why `_ERA5_CHUNK_DAYS`' own comment says its 400-day bound exists to keep that merge
  "from growing unboundedly".
- Every segment plus the merged output is a `/tmp` tempfile, so peak disk is ~2x a year's volume
  on one filesystem.

So the correct framing is: bbox chunking is the wrong instrument for a *cost-limit* problem, but
the right instrument for a *volume* problem. Replace a 0.5-degree grid bucket with clustering to
a **volume budget — a few degrees, not 25**. The Af/Am targets are naturally compact (S. Florida,
Hawaii, PR/VI), so this costs little in practice.

### 5.2 Drop `ssrd` and widen segments to what the budget actually allows

Never across a calendar-year boundary (2.2). 14 segments/bbox-year → **6**.

### 5.3 Pass `--era5-cache-dir`

So relaunches stop re-paying. This is actionable on the *currently running* lanes independently
of everything else in this document.

### 5.4 Rotate the lock's account-scan order, and dismiss jobs on exit

The free-slot-first scan always picks account 0 at concurrency 1, which is the most direct
explanation for one account carrying a 43% rejection rate while the other two carry 0% (3.2).
Rotating or randomising the scan start actually uses the three accounts we have. Dismissing
jobs on exit is separate hygiene worth doing regardless.

### 5.5 REJECTED: per-variable mixing of Open-Meteo and CDS

An earlier draft proposed taking `t2m`/`d2m` from Open-Meteo and asking CDS only for
`sp`/`u10`/`v10`, on the grounds that a 3-variable request allows 166 day-slots. **This does not
work and is withdrawn.** Three independent reasons:

1. `era5.extract_era5_means` calls the *raising* `_find_var` for t2m, d2m and sp; only
   u10/v10/ssrd use `_find_var_optional`. A CDS NetCDF lacking t2m/d2m raises `KeyError`.
   Worse, the land-mask rescue derives its `(lat_idx, lon_idx)` from t2m, and sp/u10/v10 are
   read against those indices. **The minimum CDS variable set is 5, not 3**, so the 166-slot
   budget never existed.
2. There is no join layer. The two paths are mutually exclusive whole-pipeline alternatives
   behind `--era5-source`, each returning the same `(daily, humidity, wind)` triple.
   `_daily_mean_specific_humidity` reads `d2m` and `sp` **from the same hourly row**. Mixing
   would require synthesising merged hourly rows across two producers whose datetime strings do
   not even match (`extract_era5_means` emits `"2023-06-15T00:00:00.000000000"`,
   `open_meteo._hourly_to_rows` emits `"2023-06-15T00:00"`) — a naive join yields zero matches
   silently.
3. It would put an Open-Meteo-vs-CDS offset **into the regression target**. `grid_tmax_c` /
   `grid_tmin_c` feed `delta_tmax_c = station_tmax_c - grid_tmax`, the actual label, plus the
   `grid_daily_value_c` and `grid_diurnal_range_c` features. Production serves CDS-derived grid
   values. The ~0.1-0.2 C OM-vs-CDS agreement is fine as *noise* and not fine as a *systematic
   bias in the label* — it is 5-20% of a typical few-degree delta. This is the same
   training/serving mismatch section 4 uses to disqualify the 0.25-degree mirrors, and it would
   be inconsistent to accept it here.

The supportable options remain: all-CDS with a corrected request shape, or all-Open-Meteo via
the existing `--era5-source openmeteo` flag with its disclosed NULL humidity/wind.

### 5.6 What must be measured before the fix leans on it

Two claims are staged behind measurement rather than asserted:

1. **Does enlarging the bbox cost extra queue time?** The field-limit model says no, but
   `_ERA5_MAX_CHUNK_EXTENT_DEG`'s own recorded observation says a 2.37 x 1.33 degree padded bbox
   was rejected for cost (2.1). A timed same-account comparison at 1 / 10 / 25 degrees settles
   which reading is right. (Download timing only — it deliberately does not run
   `extract_era5_means`, which 5.1 predicts would OOM at 25 degrees.)
2. **Is the ~12,000 field limit exact?** The proposed widenings leave thin margins against an
   approximate bound (93 slots x 5 var = 11,160 is 7% headroom). The only observed points are
   13,392 rejected and 4,464 accepted; nothing in between has been tested. So widen in stages —
   verify 2 months at 6 variables first, then 3 months at 5 — rather than jumping straight to
   the computed maximum.

---

## Appendix: corrections to the first version of this document

The first committed version of this diagnosis was reviewed adversarially and had two genuine
errors, both corrected above. Recorded here rather than silently overwritten:

- **The day-slot formula omitted `n_years`.** It treated the binding quantity as
  `n_months x n_days`, so it claimed 2-month segments were universally safe and that request
  counts could fall 14 → 7 → 5 → 3. Because the padded window straddles two year boundaries,
  consecutive pairing would have produced 17,856-field requests and reproduced the exact
  `cost limits exceeded` failure of 2026-08-03. Corrected counts are 8 (6 var) and 6 (5 var),
  and the "floor is 5 requests" figure was wrong in principle, not just in value.
- **The Open-Meteo/CDS per-variable hybrid was not implementable**, for the three reasons in
  5.5 — most seriously that it would have put a source offset into the regression label.

Two further overstatements were walked back: "bytes are ~free at this scale" (5.1 now states the
real O(area) ceiling), and the claim that stuck `accepted` jobs were *proven* to be queue-
poisoning orphans (3.2 now gives the CDS-result-caching explanation that defeats the FIFO
argument, and promotes the lock's scan order as the better-supported cause of the rejection
asymmetry).
