"""Reading the YAML files that describe robots and stations."""

from pathlib import Path

import yaml

from heron.errors import HeronConfigError


def read_config(path: Path, required: set, optional: set) -> dict:
    """Load a YAML mapping and check its keys.

    Unknown keys are rejected so that a typo ('base_lnk') fails loudly instead of
    silently falling back to a default.
    """
    if not path.is_file():
        raise HeronConfigError(f"{path}: file not found")
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise HeronConfigError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise HeronConfigError(f"{path}: expected a mapping of keys to values")

    missing = required - data.keys()
    if missing:
        raise HeronConfigError(f"{path}: missing key(s) {', '.join(sorted(missing))}")
    unknown = data.keys() - required - optional
    if unknown:
        allowed = ", ".join(sorted(required | optional))
        raise HeronConfigError(f"{path}: unknown key(s) {', '.join(sorted(unknown))} (allowed: {allowed})")
    if data["name"] != path.stem:
        raise HeronConfigError(f"{path}: name '{data['name']}' must match the file name '{path.stem}'")
    return data


def positive_float(path: Path, key: str, value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise HeronConfigError(f"{path}: {key} must be a number, got {value!r}") from None
    if number <= 0:
        raise HeronConfigError(f"{path}: {key} must be positive, got {number}")
    return number


def resolve_file(path: Path, key: str, value) -> Path | None:
    """A file referenced from a config: absolute, or relative to that config's directory."""
    if value in (None, ""):
        return None
    resolved = (path.parent / str(value)).resolve()
    if not resolved.is_file():
        raise HeronConfigError(f"{path}: {key} file not found: {resolved}")
    return resolved
