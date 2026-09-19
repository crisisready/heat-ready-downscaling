"""
Standalone cache-warming companion to validate_lagfill_downscaling.py's real
fetch (#710, 2026-09-19). Live-tested proof: 30 never-touched stations, 0/30
429s during an 8-concurrency/4s-timeout throwaway burst, then 30/30 (100%)
success at concurrency 3 five minutes later (mean latency 2.96s, max 15.11s)
-- the cold-cache-first-request theory holds at 6x the manager's original
5-station sample, not just as a fluke. Fires the SAME (lat, lon, start_date,
end_date) chunk shape the real job will request -- short client timeout,
discards the response body entirely -- for stations a fixed lookahead AHEAD
of wherever the real job's own checkpoint currently is, so by the time the
real job's worker gets to that station, the server-side cache is already
warm.

Deliberately does NOT go through api_call_manager.AdaptiveThrottle/
HttpSession -- this traffic's failures (mostly short-timeout TIMEOUTs, by
design) are expected and meaningless as a throttle signal; coupling it to
the real job's throttle would incorrectly degrade the real job's own
concurrency in response to throwaway prewarm timeouts it was never meant to
see.

Runs forever (SIGTERM/Ctrl-C to stop), re-reading the real job's checkpoint
file every --poll-interval-s to advance its lookahead window as real
progress is made -- read-only access to that file, never writes to it.

Own concurrency is deliberately conservative and SEPARATE from the real
fetch's concurrency (default 2): the validating test ran the burst and the
real fetch SEQUENTIALLY (burst, wait 300s, then measure) -- combined
simultaneous load (this prewarmer + the real fetch running at the same time)
was NOT covered by that test and could behave differently (e.g. hit a rate
limit neither phase alone triggered). Any 429 seen here triggers an
immediate, long cooldown before resuming -- never "push through" a rate
limit.

NOT RUNNABLE STANDALONE IN THIS REPO -- like validate_lagfill_downscaling.py
(see that script's own docstring), this imports open_meteo/api_call_manager/
db, private-repo-only modules. Only `next_window()` is pure logic and unit
tested here; the rest requires crisisready/heat-risk-data-api's own
environment, same as the script it accompanies.

Usage (run on the bastion, same directory as validate_lagfill_downscaling.py,
same env vars as that script's own Usage docstring):
    python3 prewarm_lagfill_chunks.py \\
        --checkpoint /opt/ghcn-build/rf6_band_gate_validation/lag_fill/fetch_checkpoint.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import open_meteo  # noqa: E402 -- for _HOURLY_VARS, matches the real fetch's request shape
from validate_lagfill_downscaling import (  # noqa: E402
    _ANON_URL, _CHUNK_DAYS, _KEYED_URL, _chunk_dates_by_span, _open_meteo_api_key, load_validation_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO %(message)s")
logger = logging.getLogger("prewarm")

_DEFAULT_TIMEOUT_S = 4.0
_DEFAULT_CONCURRENCY = 2
_DEFAULT_LOOKAHEAD = 15  # stations beyond the real job's checkpoint count to start prewarming from
_DEFAULT_WINDOW = 15     # how many not-yet-prewarmed stations to fire per cycle
_DEFAULT_POLL_INTERVAL_S = 60.0
_DEFAULT_COOLDOWN_S = 600.0  # conservative pause after ANY 429 -- "back off immediately, don't
                              # push through" per this feature's own build instructions.


def _read_done_station_ids(checkpoint_path: str) -> set[str]:
    done: set[str] = set()
    try:
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["key"])
    except FileNotFoundError:
        pass
    return done


def _prewarm_one_chunk(lat: float, lon: float, chunk: list[date], url: str,
                        api_key: str | None, timeout_s: float) -> str:
    """Fires one throwaway request, discards the body entirely. Returns a
    short outcome string for logging/counting -- never raises."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": chunk[0].isoformat(), "end_date": chunk[-1].isoformat(),
        "hourly": ",".join(open_meteo._HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    if api_key:
        params["apikey"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=timeout_s)
        if resp.status_code == 429:
            return "429"
        return f"http_{resp.status_code}"
    except requests.exceptions.Timeout:
        return "timeout"  # expected/common -- the whole point of a short client timeout
    except requests.RequestException as e:
        return f"exc_{type(e).__name__}"


def next_window(station_order: list[str], done: set[str], already_prewarmed: set[str],
                 lookahead: int, window: int) -> list[str]:
    """The stations to prewarm THIS cycle: skip everything the real job has
    already finished, skip `lookahead` more (those are next up for the real
    job -- prewarming them now is too late to help), then take up to
    `window` of the ones after that which haven't been prewarmed yet."""
    not_done = [s for s in station_order if s not in done]
    return [s for s in not_done[lookahead:lookahead + window] if s not in already_prewarmed]


def run_forever(checkpoint_path: str, lookahead: int, window: int, concurrency: int,
                 timeout_s: float, poll_interval_s: float, cooldown_s: float) -> None:
    api_key = _open_meteo_api_key()
    url = _KEYED_URL if api_key else _ANON_URL

    rows = load_validation_rows(sample=0, seed=0, zones=None)
    by_station: dict[str, list[dict]] = {}
    for r in rows:
        by_station.setdefault(r["station_id"], []).append(r)
    station_order = list(by_station.keys())  # same insertion order build_paired_rows itself uses

    already_prewarmed: set[str] = set()
    logger.info("prewarmer starting: %d total stations, lookahead=%d window=%d concurrency=%d timeout=%.1fs",
                len(station_order), lookahead, window, concurrency, timeout_s)

    while True:
        done = _read_done_station_ids(checkpoint_path)
        target = next_window(station_order, done, already_prewarmed, lookahead, window)

        if not target:
            if len(done) >= len(station_order):
                logger.info("real job's checkpoint shows all stations done -- prewarmer exiting")
                return
            logger.info("nothing new to prewarm this cycle -- sleeping %.0fs", poll_interval_s)
            time.sleep(poll_interval_s)
            continue

        jobs = []
        for station_id in target:
            station_rows = by_station[station_id]
            lat, lon = station_rows[0]["lat"], station_rows[0]["lon"]
            dates = [r["date"] if isinstance(r["date"], date) else date.fromisoformat(r["date"])
                     for r in station_rows]
            for chunk in _chunk_dates_by_span(dates, _CHUNK_DAYS):
                jobs.append((lat, lon, chunk))

        logger.info("prewarming %d station(s) / %d chunk(s): %s", len(target), len(jobs), target)
        outcomes: dict[str, int] = {}
        saw_429 = False
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_prewarm_one_chunk, lat, lon, chunk, url, api_key, timeout_s)
                    for lat, lon, chunk in jobs]
            for fut in futs:
                outcome = fut.result()
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                if outcome == "429":
                    saw_429 = True

        logger.info("cycle outcomes: %s", outcomes)
        already_prewarmed.update(target)

        if saw_429:
            logger.warning("429 seen during prewarm burst -- backing off %.0fs before resuming "
                            "(never pushing through a rate limit)", cooldown_s)
            time.sleep(cooldown_s)
        else:
            time.sleep(poll_interval_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the REAL job's checkpoint file (read-only)")
    ap.add_argument("--lookahead", type=int, default=_DEFAULT_LOOKAHEAD)
    ap.add_argument("--window", type=int, default=_DEFAULT_WINDOW)
    ap.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    ap.add_argument("--timeout-s", type=float, default=_DEFAULT_TIMEOUT_S)
    ap.add_argument("--poll-interval-s", type=float, default=_DEFAULT_POLL_INTERVAL_S)
    ap.add_argument("--cooldown-s", type=float, default=_DEFAULT_COOLDOWN_S)
    args = ap.parse_args()
    run_forever(args.checkpoint, args.lookahead, args.window, args.concurrency,
                args.timeout_s, args.poll_interval_s, args.cooldown_s)


if __name__ == "__main__":
    main()
