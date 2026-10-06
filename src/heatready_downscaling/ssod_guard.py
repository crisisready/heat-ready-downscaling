"""
SSOD-v2 ingestion guard. From 2025-01-01 NOAA fills GHCN-D for non-US stations from a new source (SSOD version 2,
the successor to GSOD): through 2024 the same stations carried GSOD, whose daily maximum equals GHCN-D TMAX to
< 0.1 C, but source-2 TMAX runs 0.2 to 3.8 C below GSOD MAX and TMIN runs high (heat-risk-data-api,
archive/reports/2026-10/2026-10-05-aw-dwa-truth-switch-and-refinement-e57.md). ghcn_training has no source column,
so a row cannot be tested directly. A non-US GHCN-D row dated on or after SWITCH_DATE is therefore trusted only when
its station is on an allowlist of stations whose 2025 values were checked against a GSOD twin
(research/nonus-shift-refit/gsod_twin_check.py in heat-risk-data-api): at least MIN_DAYS common days and a mean
difference within AGREE_C on both TMAX and TMIN. Every other such row is dropped from a training corpus. US stations
(GHCN-D ids starting "US") are not touched by the switch, and 8- or 9-character ids (ECA&D, AEMET, GSOD-built rows)
are not GHCN-D.
"""
import csv
import hashlib

SWITCH_DATE = "2025-01-01"
MIN_DAYS = 100
AGREE_C = 0.5


def affected(station_id: str, date) -> bool:
    """True for a row the 2025 source switch can touch: a non-US 11-character GHCN-D id dated on or after SWITCH_DATE."""
    return len(station_id) == 11 and not station_id.startswith("US") and str(date)[:10] >= SWITCH_DATE


def read_allowlist(path: str) -> set:
    """Station ids in the allowlist CSV (column station_id; a header-only file is a valid drop-everything list)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "station_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: the header must include station_id, got {reader.fieldnames}")
        return {r["station_id"] for r in reader}


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _db_affected(r: dict) -> bool:
    """Affected rows are the database's: an extra-rows file (merge_extra_rows tags it _extra) is builder output,
    e.g. real GSOD under a matched GHCN id, which the switch does not touch."""
    return not r.get("_extra") and affected(r["station_id"], r["date"])


def count_affected(rows: list[dict]) -> int:
    return sum(1 for r in rows if _db_affected(r))


def apply_guard(rows: list[dict], allowlist: set) -> tuple[list[dict], dict]:
    """rows without the affected ones whose station is not allowlisted, plus a report of what was dropped."""
    kept, dropped, stations = [], 0, set()
    for r in rows:
        if _db_affected(r) and r["station_id"] not in allowlist:
            dropped += 1
            stations.add(r["station_id"])
        else:
            kept.append(r)
    return kept, {"rows_dropped": dropped, "stations_dropped": len(stations),
                  "rows_affected_kept": count_affected(kept)}
