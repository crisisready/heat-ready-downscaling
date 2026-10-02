"""
Build the scorecard's truth station list (scorecard/v1/truth_stations.csv) and its dedupe record
(scorecard/v1/dedupe_removed.csv) from station-level inputs only, no database:

  --spans        ghcn_training station spans CSV (station_id, lat, lon, d_min, d_max, n_rows)
  --gsod-stations  station JSON(s) written by build_gsod_bsh_rows.py ({"stations": [...]}); repeatable
  --ghcnd-stations NOAA ghcnd-stations.txt, for station names (airport tags)
  --holdout      scorecard/v1/holdout_stations.txt: the stations Nishant approved moving out of training

Truth set = (a) ghcn_training stations with no row before the fast tier's scored year (unseen in
space and time already) plus (b) the holdout stations. Deduplicated: any two truth stations within
1 km, or sharing a WMO id (GHCN station list, GSOD USAF), keep the one with more scored-year rows;
a truth station that duplicates a training station is dropped from truth (it is not unseen). Zones: heatready_downscaling.koppen (the DB's own labeller).
Airport tag from the station name; urban/rural is filled later by `scorecard.py build-manifest`
from the rows' GHSL urban fraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

AIRPORT_RE = re.compile(r"\b(AIRPORT|ARPT|AIRP|INTL|INTERNATIONAL|AP|AERO|AEROPUERTO|AEROPORT|AEROPORTO|AFB|AB|RAF|"
                        r"AIRFIELD|FLD|MUNI)\b")
DEDUPE_KM = 1.0


def hav_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def airport_tag(name: str | None) -> str:
    if not name:
        return "unknown"
    return "yes" if AIRPORT_RE.search(name.upper()) else "no"


def build(spans_path, gsod_paths, ghcnd_path, holdout_path, first_scored_year: int):
    from heatready_downscaling.koppen import koppen_climate_zone
    from heatready_downscaling.scorecard import zone_group

    names, wmo = {}, {}
    with open(ghcnd_path) as f:
        for line in f:
            names[line[:11].strip()] = line[41:71].strip()
            if line[80:85].strip():
                wmo[line[:11].strip()] = line[80:85].strip()
    with open(holdout_path) as f:
        holdout = [ln.split("#", 1)[0].strip() for ln in f]
    holdout = [h for h in holdout if h]

    stations: dict[str, dict] = {}
    with open(spans_path, newline="") as f:
        for r in csv.DictReader(f):
            stations[r["station_id"]] = {"station_id": r["station_id"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                                         "first_year": int(r["d_min"][:4]), "scored_rows": int(r["n_rows"])
                                         if int(r["d_min"][:4]) >= first_scored_year else 0,
                                         "source": "ghcn_training", "name": names.get(r["station_id"], "")}
    for path in gsod_paths:
        with open(path) as f:
            for s in json.load(f)["stations"]:
                sid = s["station_id"]
                if s.get("usaf", "").isdigit() and s["usaf"] != "999999":
                    wmo.setdefault(sid, s["usaf"][:5])
                rec = stations.get(sid)
                first = min(int(y) for y in s["coverage_by_year"])
                if rec is None:
                    stations[sid] = {"station_id": sid, "lat": s["lat"], "lon": s["lon"], "first_year": first,
                                     "scored_rows": int(s["coverage_by_year"].get(str(first_scored_year), 0)),
                                     "source": "gsod_extra", "name": s.get("name", "")}
                else:  # same id in ghcn_training and a GSOD part: one physical station, earliest year wins
                    rec["first_year"] = min(rec["first_year"], first)
                    rec["scored_rows"] = max(rec["scored_rows"], int(s["coverage_by_year"].get(str(first_scored_year), 0)))
                    rec["source"] = "ghcn_training+gsod_extra"
                    rec["name"] = rec["name"] or s.get("name", "")
    missing = sorted(set(holdout) - set(stations))
    if missing:
        raise SystemExit(f"holdout stations not in any input: {missing}")

    truth = {sid for sid, s in stations.items() if s["first_year"] >= first_scored_year} | set(holdout)
    removed = []
    ids = sorted(stations)
    pts = {sid: (stations[sid]["lat"], stations[sid]["lon"]) for sid in ids}
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if abs(pts[a][0] - pts[b][0]) > 0.02:
                continue
            d = hav_km(pts[a], pts[b])
            if d <= DEDUPE_KM:
                pairs.append((a, b, d, "1 km"))
    by_wmo: dict[str, list[str]] = {}
    for sid in ids:
        if sid in wmo:
            by_wmo.setdefault(wmo[sid], []).append(sid)
    near = {(a, b) for a, b, _, _ in pairs}
    for group in by_wmo.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if (a, b) not in near:
                    pairs.append((a, b, hav_km(pts[a], pts[b]), "same WMO id"))
    for a, b, d, why in pairs:
        ta, tb = a in truth, b in truth
        if ta and tb:
            lose, keep = sorted((a, b), key=lambda s: (stations[s]["scored_rows"], s))
            removed.append({"station_id": lose, "reason": f"duplicate of another truth station ({why})", "kept": keep,
                            "distance_km": round(d, 3)})
        elif ta or tb:
            lose, keep = (a, b) if ta else (b, a)
            if lose in holdout:
                raise SystemExit(f"holdout station {lose} duplicates training station {keep} ({why})")
            removed.append({"station_id": lose, "reason": f"duplicate of a training station ({why})", "kept": keep,
                            "distance_km": round(d, 3)})
    drop = {r["station_id"] for r in removed}
    if drop & set(holdout):  # a moved station must be scored, or it was moved for nothing
        raise SystemExit(f"dedupe would drop holdout station(s) {sorted(drop & set(holdout))}")
    rows = []
    for sid in sorted(truth - drop):
        s = stations[sid]
        zone = koppen_climate_zone(s["lat"], s["lon"])
        rows.append({"station_id": sid, "lat": s["lat"], "lon": s["lon"], "zone": zone, "zone_group": None,
                     "origin": "moved_2026-10-01" if sid in holdout else "unseen_by_date",
                     "source": s["source"], "airport": airport_tag(s["name"]), "setting": "", "region": "",
                     "visibility": "public", "name": s["name"]})
    return rows, sorted(removed, key=lambda r: r["station_id"]), zone_group


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spans", required=True)
    p.add_argument("--gsod-stations", action="append", default=[])
    p.add_argument("--ghcnd-stations", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    from heatready_downscaling.scorecard import load_spec, score_year
    spec = load_spec(args.spec)
    rows, removed, zone_group = build(args.spans, args.gsod_stations, args.ghcnd_stations, args.holdout,
                                      score_year(spec["tiers"]["fast"]["cutoffs"][0]))
    for r in rows:
        r["zone_group"] = zone_group(r["zone"], spec)
    with open(os.path.join(args.out_dir, "truth_stations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "dedupe_removed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["station_id", "reason", "kept", "distance_km"])
        w.writeheader()
        w.writerows(removed)
    from collections import Counter
    print(f"truth stations {len(rows)}, removed {len(removed)}")
    print(Counter(r["zone"] for r in rows).most_common())
    print(Counter((r["origin"], r["airport"]) for r in rows))


if __name__ == "__main__":
    main()
