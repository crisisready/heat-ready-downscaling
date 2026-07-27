"""
tests/test_backfill_wind.py cannot even be collected in this repo: it
imports scripts/backfill_wind.py, which imports dem/era5/db/build_training_set
-- private-repo-only modules for a live Aurora write this repo has no path
to (see scripts/backfill_wind.py's own module docstring, "NOT RUNNABLE
STANDALONE IN THIS REPO"). Ignored here rather than deleted -- it's a real
test file, just one that needs crisisready/heat-risk-data-api's own
environment to run, same as the script it tests.
"""

collect_ignore = ["tests/test_backfill_wind.py"]
