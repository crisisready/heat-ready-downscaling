"""
tests/test_backfill_wind.py, tests/test_build_training_set.py,
tests/test_validate_lagfill_downscaling.py and tests/
test_validate_forecast_downscaling.py cannot be collected in this repo:
they import scripts/backfill_wind.py, scripts/build_training_set.py,
scripts/validate_lagfill_downscaling.py and scripts/
validate_forecast_downscaling.py, all of which import one or more of
dem/era5/db/ghcn/heat_calcs/landscan/lst/vulnerability/open_meteo/
api_call_manager -- private-repo-only modules for live Aurora/CDS/S3/
Open-Meteo access this repo has no path to (see each script's own module
docstring, "NOT RUNNABLE STANDALONE IN THIS REPO"). Ignored here rather
than deleted -- these are real test files, just ones that need
crisisready/heat-risk-data-api's own environment to run, same as the
scripts they test. scripts/validate_station_blend.py and scripts/
publish_band_gate.py / publish_blend_gate.py have NO such imports (see
their own module docstrings) -- their tests collect and run normally here.
"""

collect_ignore = [
    "tests/test_backfill_wind.py",
    "tests/test_build_training_set.py",
    "tests/test_validate_lagfill_downscaling.py",
    "tests/test_validate_forecast_downscaling.py",
]
