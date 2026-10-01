"""
Siting-form schema for station and sensor records, and the minimum contribution a record needs.

A siting form answers the questions the admission rule (admission.py) cannot answer from the
daily numbers alone: is the thermometer shielded, how high is it, what surface is below it, is it
indoors, what instrument changes happened and when, where is it, in what units, and is a "day"
local or UTC. It is machine-checked so a record cannot reach the model with the answers missing.

Open-network records (GSOD, METAR, IMD SYNOP, GHCN) are official, shielded, outdoor stations by
construction; they may be admitted without a form, and the verdict says the siting was assumed.
A contributed record always needs a form.

Minimum contribution (plan 2026-10-01, D2): daily tmax and tmin (or sub-daily values we
aggregate), coordinates, and the siting form. `validate_contribution` checks that shape.
"""

from datetime import date

import jsonschema

SITING_FORM_VERSION = 1

# Open-network sources whose siting is standard by construction.
OFFICIAL_SOURCE_KINDS = frozenset({"gsod", "ghcn", "synop", "metar"})

_ISO_DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}

SITING_FORM_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HeatReady station siting form",
    "type": "object",
    "required": [
        "form_version", "record_id", "latitude", "longitude", "shielded", "height_m",
        "surface", "placement", "units", "time_basis", "instrument_changes",
    ],
    "additionalProperties": False,
    "properties": {
        "form_version": {"const": SITING_FORM_VERSION},
        "record_id": {"type": "string", "minLength": 1},
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        # Radiation shield or Stevenson-type screen, naturally or mechanically ventilated.
        "shielded": {"type": "boolean"},
        # Sensor height above ground in metres; WMO standard is 1.25 to 2 m.
        "height_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
        "surface": {"enum": ["grass", "bare_soil", "concrete", "asphalt", "roof", "water", "other"]},
        "placement": {"enum": ["outdoor", "indoor", "semi_outdoor"]},
        "units": {"enum": ["C", "F", "K"]},
        # Whether the daily tmax/tmin window is the local calendar day or the UTC day. Required
        # because a UTC day over a UTC+5:30 site moves the daily max by up to a day.
        "time_basis": {"enum": ["local_day", "utc_day"]},
        "utc_offset_hours": {"type": "number", "minimum": -12, "maximum": 14},
        # Every sensor, shield, mast or relocation change, with its date. An empty list is an
        # affirmative statement that none are known.
        "instrument_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["date", "change"],
                "additionalProperties": False,
                "properties": {
                    "date": _ISO_DATE,
                    "change": {"enum": ["sensor", "shield", "relocation", "height", "other"]},
                    "note": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

# What a contributed dataset must carry at minimum: daily tmax/tmin, coordinates, siting form.
CONTRIBUTION_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "HeatReady minimum station contribution",
    "type": "object",
    "required": ["record_id", "latitude", "longitude", "daily", "siting_form"],
    "properties": {
        "record_id": {"type": "string", "minLength": 1},
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "daily": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["date"],
                "anyOf": [{"required": ["tmax"]}, {"required": ["tmin"]}],
                "properties": {"date": _ISO_DATE, "tmax": {"type": ["number", "null"]},
                               "tmin": {"type": ["number", "null"]}},
            },
        },
        "siting_form": SITING_FORM_SCHEMA,
    },
}


def validate_siting_form(form: dict) -> list[str]:
    """Machine check of one form. Returns a list of problems (empty means valid)."""
    validator = jsonschema.Draft202012Validator(SITING_FORM_SCHEMA)
    problems = [f"{'/'.join(map(str, e.absolute_path)) or '<form>'}: {e.message}"
                for e in sorted(validator.iter_errors(form), key=lambda e: list(map(str, e.absolute_path)))]
    for i, ch in enumerate(form.get("instrument_changes", []) if isinstance(form, dict) else []):
        try:
            date.fromisoformat(ch.get("date", ""))
        except (ValueError, AttributeError):
            problems.append(f"instrument_changes/{i}/date: not a real calendar date")
    return problems


def validate_contribution(contribution: dict) -> list[str]:
    """The minimum contribution: daily tmax/tmin, coordinates, and a valid siting form whose
    coordinates and record id agree with the record's own."""
    validator = jsonschema.Draft202012Validator(CONTRIBUTION_SCHEMA)
    problems = [f"{'/'.join(map(str, e.absolute_path)) or '<contribution>'}: {e.message}"
                for e in validator.iter_errors(contribution)]
    form = contribution.get("siting_form") if isinstance(contribution, dict) else None
    if isinstance(form, dict):
        problems += [f"siting_form: {p}" for p in validate_siting_form(form)]
        for key in ("record_id", "latitude", "longitude"):
            if key in form and key in contribution and form[key] != contribution[key]:
                problems.append(f"siting_form/{key}: differs from the record's own {key}")
    return problems
