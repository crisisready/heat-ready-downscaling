"""
Scorecard v2 truth set = v1 truth stations + the thin-zone pull stations (scripts/build_thin_zone_rows.py rows).

The pull stations are public-domain GHCN-D stations picked by heat-risk-data-api
research/thin-zone-pull/select_candidates.py, already deduplicated (same WMO id, then 1 km) against the
ghcn_training and v1 unseen stations. Dedupe is repeated here against the v1 truth list as a guard. Zone comes
from heatready_downscaling.koppen (the DB's own labeller), airport tag from the station name, urban/rural is
filled later by `scorecard.py build-manifest`. No training row is moved: these stations are in no training set.

Usage:
    python3 scripts/scorecard_extend_truth_set.py --v1-truth scorecard/v1/truth_stations.csv \
        --selected selected_stations.csv --rows thin_zone_rows.json --spec scorecard/v2/spec.yaml --out scorecard/v2/truth_stations.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extend(v1_rows, selected, rows_payload, zone_of, group_of, airport_tag, hav_km,
           origin="thin_zone_pull_2026-10-02", source="ghcnd_thin_zone_pull"):
    have = {r["station_id"]: r for r in v1_rows}
    by_station = rows_payload["rows_by_station"]
    out = list(v1_rows)
    skipped = []
    for s in selected:
        sid = s["id"]
        if sid not in by_station:
            skipped.append((sid, "no rows built"))
            continue
        if sid in have:
            skipped.append((sid, "already in v1 truth"))
            continue
        lat, lon = float(s["lat"]), float(s["lon"])
        twin = next((r["station_id"] for r in out if abs(float(r["lat"]) - lat) < 0.02
                     and hav_km((float(r["lat"]), float(r["lon"])), (lat, lon)) <= 1.0), None)
        if twin:
            skipped.append((sid, f"within 1 km of {twin}"))
            continue
        zone = zone_of(lat, lon)
        out.append({"station_id": sid, "lat": lat, "lon": lon, "zone": zone, "zone_group": group_of(zone),
                    "origin": origin, "source": source,
                    "airport": airport_tag(s.get("name")), "setting": "", "region": "",
                    "visibility": "public", "name": s.get("name", "")})
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1-truth", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--origin", default="thin_zone_pull_2026-10-02")
    ap.add_argument("--source", default="ghcnd_thin_zone_pull")
    args = ap.parse_args()
    from heatready_downscaling.koppen import koppen_climate_zone
    from heatready_downscaling.scorecard import load_spec, zone_group
    from scorecard_build_truth_set import airport_tag, hav_km
    spec = load_spec(args.spec)
    with open(args.v1_truth, newline="") as f:
        v1 = list(csv.DictReader(f))
    with open(args.selected, newline="") as f:
        selected = list(csv.DictReader(f))
    with open(args.rows) as f:
        payload = json.load(f)
    out, skipped = extend(v1, selected, payload, koppen_climate_zone, lambda z: zone_group(z, spec), airport_tag, hav_km,
                          args.origin, args.source)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(v1[0].keys()))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(f"v1 {len(v1)} + new {len(out) - len(v1)} = {len(out)}; skipped {skipped}")
    print(Counter(r["zone"] for r in out if r["origin"] == args.origin).most_common())


if __name__ == "__main__":
    main()
