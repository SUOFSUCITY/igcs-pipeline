"""Load numeric and model settings from PIPELINE_PARAMETERS."""

import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("Parameter file must contain a JSON object")
    return values


def parameter(name):
    path = os.environ.get("PIPELINE_PARAMETERS")
    if not path:
        raise ValueError("Set PIPELINE_PARAMETERS to your parameter JSON file")
    values = _load(path)
    if name not in values or values[name] is None:
        raise ValueError(f"Missing parameter: {name}")
    return values[name]


def validate_parameters(*names):
    for name in names:
        parameter(name)
