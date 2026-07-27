"""
tests/test_backfill_wind.py and tests/test_build_training_set.py cannot be
collected in this repo: they import scripts/backfill_wind.py and
scripts/build_training_set.py, both of which import dem/era5/db/ghcn/
heat_calcs/landscan/lst/vulnerability -- private-repo-only modules for
live Aurora/CDS/S3 access this repo has no path to (see each script's own
module docstring, "NOT RUNNABLE STANDALONE IN THIS REPO"). Ignored here
rather than deleted -- these are real test files, just ones that need
crisisready/heat-risk-data-api's own environment to run, same as the
scripts they test.
"""

collect_ignore = ["tests/test_backfill_wind.py", "tests/test_build_training_set.py"]
