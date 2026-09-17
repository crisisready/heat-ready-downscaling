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

Size confirms volume is not the constraint: `US_Af_c0`'s bbox is one station padded 0.5 degree
(~100 ERA5-Land cells), and the observed segment payloads are **~1.0 MB each**.

### 2.2 Calendar-month fan-out

`_ERA5_CHUNK_DAYS = 400` puts a training year in one outer chunk; the +/-2-day padding widens
2023 to 2022-12-30..2024-01-02, which `_split_by_calendar_month` cuts into **14 segments**
(Dec 2022, Jan–Dec 2023, Jan 2024) — exactly what the log reports. So 14 serialized CDS
requests per station-chunk per year.

This is over-conservative against the real budget. With 6 variables, the 12,000-field limit
allows `12000 / (6 x 24) = 83` day-slots per request, and the cartesian year/month/day schema
means the binding quantity is `n_months x n_days_in_union`:

| segment | day-slots | fields (6 var) | under 12,000? |
|---|---|---|---|
| 1 calendar month (current) | 31 | 4,464 | yes — uses 37% of budget |
| 2 calendar months | 62 | 8,928 | yes |
| 3 calendar months | 93 | 13,392 | **no** — this is the documented 2026-08-03 failure |

So 2-month segments are safe and halve the count (14 → 7). The floor, given 6 x 24 x 369 =
53,136 fields total, is 5 requests per bbox-year.

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

### 3.2 Abandoned CDS jobs are never deleted, and they poison the account

Each account currently holds 4–6 jobs in `accepted` state, never started, up to **83 minutes**
old — but `_era5_download_lock` should permit only one in-flight request per account.

Proof they are orphans rather than jobs patiently waiting their turn: on every account, jobs
created *after* the oldest still-`accepted` job have **already completed successfully** (6, 13
and 15 of them respectively). A genuine queue does not finish later arrivals first. Nothing in
`download_era5` ever deletes or dismisses a job, so every abandoned submission keeps consuming
the per-user queue allowance.

The consequence is not theoretical. One of the three accounts — the one that
`_era5_download_lock`'s free-slot-first scan reaches first, so it absorbs the most submissions —
reports `queued: 5, running: 0` against `"The maximum number of per-user requests that access
the CDS-MARS data is 1"`, and **86 of its last 200 jobs were rejected (43%)**. The other two
accounts, reached less often, show 0 rejections out of 200. Issue #648's rejected-retry handler
treats those rejections as transient congestion and backs off; they are substantially
self-inflicted.

### 3.3 One of the six requested variables is never used

`_TRAINING_ERA5_VARIABLES` requests `surface_solar_radiation_downwards` (ssrd). But
`downscaling.FEATURE_ORDER` has no solar term, and `build_rows_for_country` builds rows carrying
only `grid_tmax_c`, `grid_tmin_c`, `grid_specific_humidity_kgkg` and `nighttime_wind_ms`. **ssrd
is downloaded on every training request and discarded** — 1/6 of every request's field cost.

Dropping it raises the per-request budget from 83 to `12000 / (5 x 24) = 100` day-slots, which
makes 3 full calendar months (93 slots) fit. That alone takes 14 segments/year to 5.

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

The levers, in descending size, all independent of anything CDS decides:

1. **Stop splitting by 0.5-degree cell.** Cluster stations into a bounded number of compact
   bboxes instead. CDS's field cost does not change; only bytes do, and bytes are ~free at this
   scale. For a geographically clustered station set this is the tens-fold term.
2. **Drop `ssrd`** (unused) and widen segments to what the field budget actually allows:
   14 requests/bbox-year → 5.
3. **Pass `--era5-cache-dir`** so relaunches stop re-paying.
4. **Delete CDS jobs after use, and reap orphans** — this is what is costing one account a 43%
   rejection rate.
5. **Take `t2m` and `d2m` from Open-Meteo** (0.5 s, already wired, live-validated) and ask CDS
   only for `sp`/`u10`/`v10`. At 3 variables the budget is `12000 / (3 x 24) = 166` day-slots,
   so 5 calendar months fit per request: 14 requests/bbox-year → 3.

One claim above is worth a direct measurement before the fix leans on it: that **enlarging the
bbox leaves queue time flat**. The field-limit argument and ECMWF's documentation both say it
should, and payload bytes are negligible either way, but "documented cost model" and "observed
queue behaviour under a 7,003-deep global backlog" are not the same thing. A timed
same-account comparison at 1 / 10 / 25 degrees settles it cheaply.
