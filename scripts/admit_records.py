"""
Run station records through the admission rule (src/heatready_downscaling/admission.py).

Wires the rule to `scripts/gsod_homogeneity.screen` (called unchanged): for each candidate record the
screen runs over the whole network with that record standing in for its station, and its result for
that station becomes the record's `BreakResult`. The rule itself never sees a model output.

Inputs are public: GSOD-built network rows (build_gsod_bsh_rows.py output), optional extra-station
rows in the same format (e.g. a SYNOP-derived station), and an IEM METAR export for one airport.
Output: <out-dir>/verdicts.jsonl (one line per record: id, sha256, verdict, deciding rule, evidence)
and <out-dir>/admission_<location>.json (the whole result).

Usage (the Ahmedabad airport pair, with Gandhinagar as the third record):
    python3 scripts/admit_records.py --location ahmedabad --serving-year 2026 \\
        --rows rows_none.json --extra-rows gnr_rows.json \\
        --airport-station IN005010600 --airport-gsod-id gsod-42647 \\
        --metar-csv metar_VAAH_2016_2025.csv --metar-id metar-VAAH \\
        --third-station ING426540 --third-id synop-42654 --out-dir out/
"""
import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import gsod_homogeneity as gh
from heatready_downscaling import admission as ad

# The METAR daily maximum rule of the Ahmedabad level investigation (heat-risk-data-api,
# research/ahmedabad-bsh-refinement/level/gandhinagar/run_metar_break.py): a day counts when at least six
# distinct hours between 11 and 17 UTC reported, and its value is the highest reported temperature.
METAR_AFTERNOON_HOURS = range(11, 18)
METAR_MIN_AFTERNOON_HOURS = 6


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rows(path):
    with open(path, "rb") as f:
        payload = json.loads(f.read())
    if payload.get("complete") is not True:
        raise SystemExit(f"{path} is not marked complete")
    return payload["rows"]


def metar_daily_max(csv_path):
    """{ISO date: daily max tmpc} from an IEM export (station, valid, tmpc), by the rule above."""
    hours, best = {}, {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["tmpc"] in ("M", ""):
                continue
            date, hour = row["valid"][:10], int(row["valid"][11:13])
            t = float(row["tmpc"])
            best[date] = max(best.get(date, -99.0), t)
            if hour in METAR_AFTERNOON_HOURS:
                hours.setdefault(date, set()).add(hour)
    return {d: v for d, v in best.items() if len(hours.get(d, ())) >= METAR_MIN_AFTERNOON_HOURS}


def _series(rows, station_id):
    return {r["date"]: r["station_tmax_c"] for r in rows if r["station_id"] == station_id
            and r.get("station_tmax_c") is not None}


def _break_result(network_rows, station_id, window):
    """The screen sees every row up to the end of the anchor window and none after it. The rule text says
    "window plus 3 preceding years", but that is 5 years and the screen needs at least 6 per station-season
    (gsod_homogeneity.MIN_YEARS), so a clip that tight leaves every record untestable; the earlier rows are kept."""
    tests, first_kept = gh.screen([r for r in network_rows if int(r["date"][:4]) <= window[1]])
    return ad.break_result_from_screen(tests, first_kept, station_id)


def _substituted(rows, station_id, daily):
    """The network with one station's daily tmax replaced by another record's (days it lacks, or
    that have no ERA5-Land value, drop out), exactly as the investigation's METAR test did."""
    out = []
    for r in rows:
        if r["station_id"] != station_id:
            out.append(r)
            continue
        v = daily.get(r["date"])
        if v is None or r.get("grid_tmax_c") is None:
            continue
        out.append({**r, "station_tmax_c": v, "delta_tmax_c": v - r["grid_tmax_c"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--location", required=True)
    ap.add_argument("--serving-year", type=int, required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--extra-rows", action="append", default=[])
    ap.add_argument("--airport-station", required=True)
    ap.add_argument("--airport-gsod-id", required=True)
    ap.add_argument("--metar-csv", required=True)
    ap.add_argument("--metar-id", required=True)
    ap.add_argument("--metar-station", default="VAAH")
    ap.add_argument("--third-station")
    ap.add_argument("--third-id")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    window = ad.anchor_window(args.serving_year)
    base = _load_rows(args.rows)
    for extra in args.extra_rows:
        base += _load_rows(extra)
    coords = {}
    for r in base:
        coords.setdefault(r["station_id"], (r["lat"], r["lon"]))
    amd = coords[args.airport_station]
    metar_days = metar_daily_max(args.metar_csv)

    gsod_days = _series(base, args.airport_station)
    gsod = ad.Record(args.airport_gsod_id, ad.sha256_of_days(gsod_days), args.airport_station, *amd, "gsod",
                     "daily", gsod_days, _break_result(base, args.airport_station, window))
    metar = ad.Record(args.metar_id, ad.sha256_of_days(metar_days), args.metar_station, *amd, "metar", "hourly",
                      metar_days,
                      _break_result(_substituted(base, args.airport_station, metar_days), args.airport_station, window))
    resolvers = []
    if args.third_station:
        t_days = _series(base, args.third_station)
        resolvers.append(ad.Record(args.third_id, ad.sha256_of_days(t_days), args.third_station,
                                   *coords[args.third_station], "synop", "synoptic", t_days,
                                   _break_result(base, args.third_station, window)))

    result = ad.admit(args.location, [gsod, metar], args.serving_year, resolvers=resolvers)
    os.makedirs(args.out_dir, exist_ok=True)
    log = os.path.join(args.out_dir, "verdicts.jsonl")
    ad.log_result(result, log)
    out = ad.result_to_dict(result)
    out["inputs"] = {"rows": {"path": os.path.basename(args.rows), "sha256": _sha256_file(args.rows)},
                     "extra_rows": [{"path": os.path.basename(p), "sha256": _sha256_file(p)} for p in args.extra_rows],
                     "metar_csv": {"path": os.path.basename(args.metar_csv), "sha256": _sha256_file(args.metar_csv)},
                     "last_day": {r.record_id: max(r.days) for r in [gsod, metar] + resolvers}}
    out["resolvers"] = [{"record_id": t.record_id, "sha256": t.sha256, "homogeneity": t.break_result.__dict__,
                         "coverage": {s: round(c, 3) for s, c in
                                      ad.season_coverage(t, ad.anchor_window(args.serving_year)).items()}}
                        for t in resolvers]
    with open(os.path.join(args.out_dir, f"admission_{args.location}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for v in result.verdicts:
        print(f"{v.record_id}: {v.verdict} ({v.rule}) role={v.role} coverage={v.evidence.get('coverage')}")
    print("chosen:", result.chosen_record_id, "window:", result.window)


if __name__ == "__main__":
    main()
