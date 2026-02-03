from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# Paths are relative to parent directory (since script is in /tools)
WEIGHTS_PATH = Path("../weights.json")
ID_MAPS_PATH = Path("../id_maps.json")
# TEAMS_PATH is constructed dynamically based on league argument

PITCHER_ROLES = ["SP", "RP"]
PITCH_ABILITY_METRICS = ["PotStf", "PotMov", "PotHRA", "PotCtrl"]
PITCH_ABILITY_DISPLAY = {
    "PotStf": "Stuff",
    "PotMov": "Movement",
    "PotHRA": "Home Run Avoid.",
    "PotCtrl": "Control",
}

PITCH_ARSENAL_COLUMNS = [
    "PotFst",
    "PotSnk",
    "PotCutt",
    "PotCrv",
    "PotSld",
    "PotChg",
    "PotSplt",
    "PotFrk",
    "PotCirChg",
    "PotScr",
    "PotKncrv",
    "PotKnbl",
]
PITCH_TYPE_BY_COLUMN = {
    "PotFst": "Fst",
    "PotSnk": "Snk",
    "PotCutt": "Cutt",
    "PotCrv": "Crv",
    "PotSld": "Sld",
    "PotChg": "Chg",
    "PotSplt": "Splt",
    "PotFrk": "Frk",
    "PotCirChg": "CirChg",
    "PotScr": "Scr",
    "PotKncrv": "Kncrv",
    "PotKnbl": "Knbl",
}
PITCH_SPEED_TIERS = {
    "Fst": "hard",
    "Snk": "hard",
    "Cutt": "hard",
    "Sld": "breaker",
    "Crv": "breaker",
    "Kncrv": "breaker",
    "Knbl": "breaker",
    "Chg": "offspeed",
    "CirChg": "offspeed",
    "Splt": "offspeed",
    "Frk": "offspeed",
    "Scr": "offspeed",
}
PITCH_BREAK_PLANES = {
    "Fst": "vertical",
    "Snk": "vertical",
    "Cutt": "horizontal",
    "Sld": "horizontal",
    "Crv": "vertical",
    "Kncrv": "vertical",
    "Knbl": "horizontal",
    "Chg": "vertical",
    "CirChg": "vertical",
    "Splt": "vertical",
    "Frk": "vertical",
    "Scr": "horizontal",
}
PITCH_CURRENT_METRICS = ["Stf", "Mov", "HRA", "Ctrl"]

BAT_TOOL_HEADERS = ["Gap", "Pow", "Eye", "Ks"]
BAT_TOOL_POT_HEADERS = ["PotGap", "PotPow", "PotEye", "PotKs"]
BAT_TOOL_COLUMNS = BAT_TOOL_HEADERS + BAT_TOOL_POT_HEADERS
HITTER_CURRENT_METRICS = BAT_TOOL_HEADERS
HITTER_POTENTIAL_METRICS = BAT_TOOL_POT_HEADERS

BASERUNNING_HEADERS = ["Speed", "StlRt", "StealAbi", "Run"]
BASERUNNING_COLUMN_PREFERENCES: Dict[str, List[str]] = {
    "Speed": ["Speed"],
    "StlRt": ["StlRt"],
    "StealAbi": ["StealAbi", "Steal"],
    "Run": ["Run"],
}

DEFENSE_TOOL_COLUMNS: Dict[str, List[str]] = {
    "C": ["CBlk", "CArm", "CFrm"],
    "1B": ["IFR", "IFE", "IFA", "TDP"],
    "2B": ["IFR", "IFE", "IFA", "TDP"],
    "3B": ["IFR", "IFE", "IFA", "TDP"],
    "SS": ["IFR", "IFE", "IFA", "TDP"],
    "LF": ["OFR", "OFE", "OFA"],
    "CF": ["OFR", "OFE", "OFA"],
    "RF": ["OFR", "OFE", "OFA"],
}

PERSONALITY_TRAITS = ["Int", "WrkEthic", "Greed", "Loy", "Lead"]
BASELINE_RATIO_FLOOR = 0.0


class ValidationError(Exception):
    """
    Custom exception class for collecting validation issues found in the dataset.
    
    This exception is raised when data validation fails, and can contain
    multiple detailed error messages for comprehensive reporting.
    
    Attributes:
        message: Primary error message describing the validation failure
        details: Optional list of additional detailed error messages
    """

    def __init__(self, message: str, details: Iterable[str] | None = None) -> None:
        super().__init__(message)
        self.details = list(details or [])


def load_id_filter(path: Path) -> Set[str]:
    """
    Load player IDs from a filter file for selective processing.
    
    Reads a file containing player IDs (separated by commas, semicolons, tabs, or newlines)
    and returns a set of sanitized identifier strings. Used to limit evaluation to
    a specific subset of players.
    
    Args:
        path: Path to the file containing player IDs to filter
    
    Returns:
        Set of player ID strings extracted from the file
    
    Raises:
        FileNotFoundError: If the filter file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(f"ID filter file not found: {path.resolve()}")

    identifiers: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            sanitized = (
                line.replace(",", " ")
                .replace(";", " ")
                .replace("\t", " ")
            )
            for token in sanitized.split():
                token = token.strip()
                if token:
                    identifiers.add(token)

    return identifiers


def round_optional(value: object, digits: int = 2) -> Optional[float]:
    """
    Safely round a numeric value, handling None and non-numeric inputs.
    
    Attempts to convert the input to a float and round it to the specified
    number of decimal places. Returns None if the value cannot be converted
    or is already None.
    
    Args:
        value: The value to round (can be int, float, string, or None)
        digits: Number of decimal places for rounding (default: 2)
    
    Returns:
        Rounded float value, or None if value is None or cannot be converted
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return round(numeric, digits)


def normalize_int(value: object) -> Optional[int]:
    """
    Convert a value to an integer, handling various input types and edge cases.
    
    Safely converts strings, floats, and integers to int, handling NaN values
    and invalid inputs. Returns None for unparseable values.
    
    Args:
        value: The value to convert (can be int, float, string, or None)
    
    Returns:
        Integer value, or None if value is None, NaN, or cannot be converted
    """
    if value is None:
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def normalize_float(value: object) -> Optional[float]:
    """
    Convert a value to a float, handling various input types and edge cases.
    
    Safely converts strings, integers, and floats to float, handling NaN values
    and invalid inputs. Returns None for unparseable values.
    
    Args:
        value: The value to convert (can be int, float, string, or None)
    
    Returns:
        Float value, or None if value is None, NaN, or cannot be converted
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def combine_team_name(info: Dict[str, object]) -> str:
    """
    Combine team name and nickname into a full team name string.
    
    Intelligently combines the team's name and nickname fields, avoiding
    duplication if the nickname is already contained in the name.
    
    Args:
        info: Dictionary containing team information with "Name" and "Nickname" keys
    
    Returns:
        Combined team name string, or empty string if neither name nor nickname exists
    """
    name = str(info.get("Name") or "").strip()
    nickname = str(info.get("Nickname") or "").strip()
    if name and nickname:
        lower_name = name.lower()
        lower_nickname = nickname.lower()
        if lower_nickname and lower_nickname in lower_name:
            return name
        return f"{name} {nickname}".strip()
    return name or nickname or ""


def print_progress(processed: int, total: int) -> None:
    """
    Print a progress bar to stdout showing processing status.
    
    Displays a visual progress bar with percentage complete, updating in place.
    Automatically adds a newline when processing is complete.
    
    Args:
        processed: Number of items processed so far
        total: Total number of items to process
    """
    if total <= 0:
        return

    bar_length = 30
    ratio = processed / total
    filled = int(bar_length * ratio)
    bar = "#" * filled + "-" * (bar_length - filled)
    percent = ratio * 100
    sys.stdout.write(
        f"\rProgress: [{bar}] {processed}/{total} ({percent:5.1f}%)"
    )
    if processed >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def adjust_rating(raw: float) -> float:
    """
    Apply nonlinear adjustments to emphasize elite tools and penalize weak ones.
    
    Modifies raw ratings with multipliers to create more separation between
    elite and poor ratings. Elite ratings (70+) get a 20% boost, good ratings
    (60+) get a 10% boost, while poor ratings (30-) get penalized by 25%
    and below-average ratings (40-) get penalized by 10%.
    
    Args:
        raw: The raw rating value to adjust
    
    Returns:
        Adjusted rating value with multiplier applied
    """
    if raw >= 70:
        return raw * 1.20
    if raw >= 60:
        return raw * 1.10
    if raw <= 30:
        return raw * 0.75
    if raw <= 40:
        return raw * 0.90
    return raw


def load_data(path: Path) -> pd.DataFrame:
    """
    Load player data from a CSV file into a pandas DataFrame.
    
    Reads the CSV file containing all player statistics and attributes
    needed for evaluation. Validates that the file is not empty.
    
    Args:
        path: Path to the CSV file containing player data
    
    Returns:
        DataFrame containing all player data
    
    Raises:
        FileNotFoundError: If the data file does not exist
        ValidationError: If the data file is empty
    """
    if not path.exists():
        raise FileNotFoundError(f"Player data file not found: {path.resolve()}")

    # Check for potential file corruption (all data on one line)
    with path.open("r", encoding="utf-8") as fh:
        line_count = sum(1 for _ in fh)
        fh.seek(0)
        first_line_length = len(fh.readline())
    
    if line_count < 3:  # Header + at least 1-2 data rows expected
        raise ValidationError(
            f"CSV file appears corrupted: only {line_count} line(s) found. "
            f"Expected multiple lines (one per player). "
            f"The file may have all data on a single line. Please check the file format."
        )
    
    if first_line_length > 10000:  # Suspiciously long first line
        print(
            f"[WARN] First line of CSV is very long ({first_line_length} characters). "
            f"This may indicate file corruption where all data is on one line.",
            flush=True
        )
    
    # Use optimized parsing options to handle large/malformed CSVs
    # low_memory=False forces pandas to read the entire file into memory at once
    # which is more efficient for files with many columns
    # engine='c' uses the faster C parser
    try:
        # Try newer pandas API first (pandas >= 1.3.0)
        data = pd.read_csv(
            path,
            low_memory=False,
            engine='c',
            on_bad_lines='skip',  # Skip malformed lines instead of crashing
        )
    except TypeError:
        # Fallback for older pandas versions (pandas < 1.3.0)
        try:
            data = pd.read_csv(
                path,
                low_memory=False,
                engine='c',
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        except TypeError:
            # Final fallback - just use basic read_csv
            data = pd.read_csv(path, low_memory=False)
    
    if data.empty:
        raise ValidationError("Player data file is empty.")
    
    # Additional validation: check if we got suspiciously few rows
    if len(data) < 10 and line_count > 2:
        print(
            f"[WARN] CSV loaded but only {len(data)} rows found. "
            f"This may indicate parsing issues with the file format.",
            flush=True
        )

    return data


def load_reference_data(
    id_map_path: Path, teams_path: Path
) -> Tuple[Dict[int, str], Dict[int, Dict[str, object]]]:
    """
    Load reference data for league levels and team information.
    
    Reads JSON files containing mappings for league level IDs to labels
    and team IDs to team information. Used for resolving team and league
    names in player reports.
    
    Args:
        id_map_path: Path to JSON file containing league level mappings
        teams_path: Path to JSON file containing team information
    
    Returns:
        Tuple containing:
        - Dictionary mapping league level IDs to label strings
        - Dictionary mapping team IDs to team information dictionaries
    
    Raises:
        FileNotFoundError: If either reference file does not exist
    """
    if not id_map_path.exists():
        raise FileNotFoundError(f"ID map file not found: {id_map_path.resolve()}")
    if not teams_path.exists():
        raise FileNotFoundError(f"Teams file not found: {teams_path.resolve()}")

    with id_map_path.open("r", encoding="utf-8") as fh:
        id_maps_raw = json.load(fh)

    league_lookup: Dict[int, str] = {}
    league_map_raw = id_maps_raw.get("league_level", {})
    if isinstance(league_map_raw, dict):
        for label, value in league_map_raw.items():
            try:
                key = int(value)
            except (TypeError, ValueError):
                continue
            league_lookup[key] = str(label)

    with teams_path.open("r", encoding="utf-8") as fh:
        teams_raw = json.load(fh)

    team_lookup: Dict[int, Dict[str, object]] = {}
    if isinstance(teams_raw, dict):
        for key, info in teams_raw.items():
            try:
                team_id = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(info, dict):
                continue
            parent_id = normalize_int(info.get("Parent")) or 0
            full_name = combine_team_name(info)
            team_lookup[team_id] = {
                "name": info.get("Name"),
                "nickname": info.get("Nickname"),
                "parent": parent_id,
                "full_name": full_name or f"Team {team_id}",
            }

    return league_lookup, team_lookup


def _normalize_baseline_map(
    raw_map: Dict[str, Dict[str, object]], label: str
) -> Dict[str, Dict[str, float]]:
    """
    Normalize and validate a baseline map from raw configuration data.
    
    Converts a dictionary of baseline values (e.g., position-specific baselines)
    from raw JSON format into a validated structure with all values as floats.
    Used internally for processing defense, bat, and baserunning baselines.
    
    Args:
        raw_map: Dictionary mapping positions/metrics to baseline values
        label: Label for the baseline type (used in error messages)
    
    Returns:
        Normalized dictionary with all values as floats
    
    Raises:
        ValidationError: If any baseline values are invalid or non-numeric
    """
    normalized: Dict[str, Dict[str, float]] = {}
    invalid_entries: List[str] = []

    for position, metrics in raw_map.items():
        if not isinstance(metrics, dict):
            invalid_entries.append(f"{label}:{position} -> must be a dictionary")
            continue
        normalized_metrics: Dict[str, float] = {}
        for metric_name, value in metrics.items():
            try:
                normalized_metrics[metric_name] = float(value)
            except (TypeError, ValueError):
                invalid_entries.append(
                    f"{label}:{position}.{metric_name} -> value must be numeric"
                )
        normalized[position] = normalized_metrics

    if invalid_entries:
        raise ValidationError(f"{label.title()} baselines invalid.", invalid_entries)

    return normalized


# Age-level configuration helpers

def _normalize_age_entry(
    entry: Dict[str, object], fallback: Dict[str, float]
) -> Dict[str, float]:
    """
    Normalize a single age-level configuration entry.
    
    Processes a single age-level configuration entry, extracting target age,
    band width, ahead bonus, and behind penalty values. Uses fallback values
    for missing keys and validates that all values are numeric.
    
    Args:
        entry: Dictionary containing age configuration for a specific level
        fallback: Dictionary of fallback values to use for missing keys
    
    Returns:
        Normalized dictionary with target, band, ahead_bonus, and behind_penalty
    
    Raises:
        ValidationError: If any required values are non-numeric
    """
    def _get(key: str, default: float) -> float:
        value = entry.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"Age-level config '{key}' must be numeric.")

    target = _get("target", fallback.get("target", 26.0))
    band = max(0.1, _get("band", fallback.get("band", 2.0)))
    ahead_bonus = _get("ahead_bonus", fallback.get("ahead_bonus", 0.0))
    behind_penalty = _get("behind_penalty", fallback.get("behind_penalty", 0.0))

    return {
        "target": target,
        "band": band,
        "ahead_bonus": ahead_bonus,
        "behind_penalty": behind_penalty,
    }


def _normalize_age_level_config(
    raw_config: object,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Normalize and validate the complete age-level configuration structure.
    
    Processes the age_level configuration section from weights.json, which
    defines age adjustment parameters for hitters and pitchers at different
    league levels. Creates a normalized structure with defaults and level-specific
    overrides.
    
    Args:
        raw_config: Raw age_level configuration object from JSON
    
    Returns:
        Nested dictionary structure: role -> {default, levels} -> level -> {target, band, ahead_bonus, behind_penalty}
    
    Raises:
        ValidationError: If the configuration structure is invalid
    """
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValidationError("'age_level' must be a dictionary.")

    default_defaults = {
        "target": 26.0,
        "band": 2.0,
        "ahead_bonus": 0.0,
        "behind_penalty": 0.0,
    }

    normalized: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for role in ["hitter", "pitcher"]:
        role_raw = raw_config.get(role, {})
        if role_raw is None:
            role_raw = {}
        if not isinstance(role_raw, dict):
            raise ValidationError(f"'age_level.{role}' must be a dictionary.")

        default_raw = role_raw.get("default", {})
        if not isinstance(default_raw, dict):
            default_raw = {}
        default_entry = _normalize_age_entry(default_raw, default_defaults)

        levels_raw = role_raw.get("levels", {})
        if levels_raw is None:
            levels_raw = {}
        if not isinstance(levels_raw, dict):
            raise ValidationError(f"'age_level.{role}.levels' must be a dictionary.")

        levels: Dict[str, Dict[str, float]] = {}
        for level_label, entry in levels_raw.items():
            if not isinstance(entry, dict):
                raise ValidationError(
                    f"'age_level.{role}.levels.{level_label}' must be a dictionary."
                )
            levels[str(level_label)] = _normalize_age_entry(entry, default_entry)

        normalized[role] = {
            "default": default_entry,
            "levels": levels,
        }

    # Ensure both roles exist, even if omitted in config
    for role in ["hitter", "pitcher"]:
        normalized.setdefault(
            role,
            {
                "default": default_defaults.copy(),
                "levels": {},
            },
        )

    return normalized


# Lookup helpers

def resolve_team_label(
    team_value: object, team_lookup: Dict[int, Dict[str, object]]
) -> str:
    """
    Resolve a team ID to its full display name.
    
    Converts a team ID value to a human-readable team name using the
    team lookup dictionary. Returns "Unassigned" for invalid or missing IDs.
    
    Args:
        team_value: Team ID value (can be int, string, or None)
        team_lookup: Dictionary mapping team IDs to team information
    
    Returns:
        Full team name string, or "Unassigned" if ID is invalid/missing
    """
    team_id = normalize_int(team_value)
    if team_id is None or team_id == 0:
        return "Unassigned"
    info = team_lookup.get(team_id)
    if info:
        return str(info.get("full_name") or combine_team_name(info)) or f"Team {team_id}"
    return f"Team {team_id}"


def resolve_league_label(
    league_value: object, league_lookup: Dict[int, str]
) -> str:
    """
    Resolve a league level ID to its display label.
    
    Converts a league level ID value to a human-readable league level name
    using the league lookup dictionary. Returns "Unassigned" for invalid or missing IDs.
    
    Args:
        league_value: League level ID value (can be int, string, or None)
        league_lookup: Dictionary mapping league level IDs to label strings
    
    Returns:
        League level label string, or "Unassigned" if ID is invalid/missing
    """
    league_id = normalize_int(league_value)
    if league_id is None or league_id == 0:
        return "Unassigned"
    return league_lookup.get(league_id, f"Level {league_id}")


def load_weights(
    path: Path,
    required_bat_keys: Iterable[str],
    baserunning_keys: Iterable[str],
    pitcher_roles: Iterable[str],
    pitch_metrics: Iterable[str],
) -> Tuple[Any, ...]:
    """
    Load and validate all weight configurations from weights.json.
    
    Reads the comprehensive weights configuration file and validates all
    sections including pitch ability weights, pitch arsenal configuration,
    bat tool weights, defense weights, baserunning weights, position weights,
    personality modifiers, baselines, and age-level configurations.
    
    Args:
        path: Path to the weights.json configuration file
        required_bat_keys: List of required bat tool metric names
        baserunning_keys: List of required baserunning metric names
        pitcher_roles: List of valid pitcher role names (e.g., ["SP", "RP"])
        pitch_metrics: List of required pitch ability metric names
    
    Returns:
        Tuple containing all loaded weight configurations in this order:
        - pitch_ability_weights: Dict[role, Dict[metric, weight]]
        - pitch_type_weights: Dict[pitch_type, weight]
        - role_slot_weights: Dict[role, Dict[slot, weight]]
        - role_baselines: Dict[role, Dict[baseline_key, value]]
        - diversity_modifiers: Dict[modifier_key, value]
        - development_config: Dict[role, config_dict]
        - pitch_role_adjustments: Dict[role, multiplier]
        - stamina_thresholds: Dict[role, Dict[threshold_key, value]]
        - bat_weights: Dict[metric, weight]
        - defense_weights: Dict[position, Dict[metric, weight]]
        - baserunning_weights: Dict[metric, weight]
        - position_weights: Dict[category, Dict[position, weight]]
        - personality_thresholds: Dict[threshold_type, value]
        - personality_modifiers: Dict[trait, Dict[bucket, modifier]]
        - age_level_config: Normalized age level configuration
        - archetype_config: Archetype classification configuration
        - defense_baselines: Dict[position, Dict[metric, baseline]]
        - bat_baselines: Dict[position, Dict[metric, baseline]]
        - baserunning_baselines: Dict[position, Dict[metric, baseline]]
    
    Raises:
        FileNotFoundError: If the weights file does not exist
        ValidationError: If any section of the weights file is invalid or missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        weights_raw = json.load(fh)

    pitch_ability_raw = weights_raw.get("pitch_ability")
    if pitch_ability_raw is None:
        raise ValidationError("Missing 'pitch_ability' section in weights.json.")
    if not isinstance(pitch_ability_raw, dict):
        raise ValidationError("'pitch_ability' section must be a dictionary.")

    pitch_ability_weights: Dict[str, Dict[str, float]] = {}
    expected_roles = set(pitcher_roles)
    pitch_invalid_entries: List[str] = []

    expected_metrics = set(pitch_metrics)

    for role, metrics in pitch_ability_raw.items():
        if role not in expected_roles:
            pitch_invalid_entries.append(
                f"{role}: unsupported role (expected one of {sorted(expected_roles)})"
            )
            continue
        if not isinstance(metrics, dict):
            pitch_invalid_entries.append(f"{role}: weights must be a dictionary.")
            continue

        normalized_metrics: Dict[str, float] = {}
        missing_metrics = sorted(expected_metrics - set(metrics.keys()))
        if missing_metrics:
            pitch_invalid_entries.append(
                f"{role}: missing metric weights {missing_metrics}"
            )
        for metric_name, value in metrics.items():
            if metric_name not in expected_metrics:
                pitch_invalid_entries.append(
                    f"{role}.{metric_name}: unsupported metric."
                )
                continue
            if not isinstance(value, (int, float)):
                pitch_invalid_entries.append(
                    f"{role}.{metric_name}: weight must be numeric."
                )
                continue
            normalized_metrics[metric_name] = float(value)

        if normalized_metrics:
            pitch_ability_weights[role] = normalized_metrics

    if pitch_invalid_entries:
        raise ValidationError(
            "Pitch ability weights contain invalid entries.", pitch_invalid_entries
        )

    pitch_arsenal_raw = weights_raw.get("pitch_arsenal")
    if pitch_arsenal_raw is None:
        raise ValidationError("Missing 'pitch_arsenal' section in weights.json.")
    if not isinstance(pitch_arsenal_raw, dict):
        raise ValidationError("'pitch_arsenal' section must be a dictionary.")

    pitch_type_weights_raw = pitch_arsenal_raw.get("pitch_type_weights")
    if not isinstance(pitch_type_weights_raw, dict):
        raise ValidationError(
            "'pitch_arsenal.pitch_type_weights' section must be a dictionary."
        )

    pitch_type_weights: Dict[str, float] = {}
    invalid_pitch_types: List[str] = []
    expected_pitch_types = set(PITCH_TYPE_BY_COLUMN.values())
    for pitch_type, weight in pitch_type_weights_raw.items():
        if pitch_type not in expected_pitch_types:
            invalid_pitch_types.append(
                f"{pitch_type}: unsupported pitch type (expected {sorted(expected_pitch_types)})"
            )
            continue
        if not isinstance(weight, (int, float)):
            invalid_pitch_types.append(
                f"{pitch_type}: weight must be numeric."
            )
            continue
        pitch_type_weights[pitch_type] = float(weight)

    missing_type_weights = sorted(expected_pitch_types - set(pitch_type_weights.keys()))
    if missing_type_weights:
        invalid_pitch_types.append(
            f"Missing weights for pitch types {missing_type_weights}"
        )

    if invalid_pitch_types:
        raise ValidationError(
            "Pitch type weights contain invalid entries.", invalid_pitch_types
        )

    role_slot_weights_raw = pitch_arsenal_raw.get("role_slot_weights")
    if not isinstance(role_slot_weights_raw, dict):
        raise ValidationError(
            "'pitch_arsenal.role_slot_weights' section must be a dictionary."
        )

    role_slot_weights: Dict[str, Dict[str, float]] = {}
    invalid_slot_entries: List[str] = []
    for role, slots in role_slot_weights_raw.items():
        if role not in expected_roles:
            invalid_slot_entries.append(
                f"{role}: unsupported role for slot weights (expected {sorted(expected_roles)})"
            )
            continue
        if not isinstance(slots, dict):
            invalid_slot_entries.append(f"{role}: slot weights must be a dictionary.")
            continue
        normalized_slots: Dict[str, float] = {}
        for slot_name, value in slots.items():
            if not isinstance(value, (int, float)):
                invalid_slot_entries.append(
                    f"{role}.{slot_name}: slot weight must be numeric."
                )
                continue
            normalized_slots[slot_name] = float(value)
        if normalized_slots:
            role_slot_weights[role] = normalized_slots

    if invalid_slot_entries:
        raise ValidationError(
            "Pitch arsenal slot weights contain invalid entries.", invalid_slot_entries
        )

    role_baselines_raw = pitch_arsenal_raw.get("role_baselines")
    if not isinstance(role_baselines_raw, dict):
        raise ValidationError(
            "'pitch_arsenal.role_baselines' section must be a dictionary."
        )

    role_baselines: Dict[str, Dict[str, float]] = {}
    invalid_baseline_entries: List[str] = []
    expected_baseline_keys = {
        "min_pitches",
        "target_pitches",
        "min_speed_tiers",
        "min_break_planes",
    }
    for role, baseline_values in role_baselines_raw.items():
        if role not in expected_roles:
            invalid_baseline_entries.append(
                f"{role}: unsupported role for baselines (expected {sorted(expected_roles)})"
            )
            continue
        if not isinstance(baseline_values, dict):
            invalid_baseline_entries.append(
                f"{role}: baseline values must be a dictionary."
            )
            continue
        normalized_baselines: Dict[str, float] = {}
        missing_keys = sorted(
            expected_baseline_keys - set(baseline_values.keys())
        )
        if missing_keys:
            invalid_baseline_entries.append(
                f"{role}: missing baseline keys {missing_keys}"
            )
        for key, value in baseline_values.items():
            if key not in expected_baseline_keys:
                invalid_baseline_entries.append(
                    f"{role}.{key}: unsupported baseline key."
                )
                continue
            if not isinstance(value, (int, float)):
                invalid_baseline_entries.append(
                    f"{role}.{key}: baseline value must be numeric."
                )
                continue
            normalized_baselines[key] = float(value)
        role_baselines[role] = normalized_baselines

    if invalid_baseline_entries:
        raise ValidationError(
            "Pitch arsenal baselines contain invalid entries.", invalid_baseline_entries
        )

    diversity_modifiers_raw = pitch_arsenal_raw.get("diversity_modifiers", {})
    if not isinstance(diversity_modifiers_raw, dict):
        raise ValidationError(
            "'pitch_arsenal.diversity_modifiers' section must be a dictionary."
        )

    diversity_modifiers: Dict[str, float] = {}
    invalid_diversity_entries: List[str] = []
    for key, value in diversity_modifiers_raw.items():
        if not isinstance(value, (int, float)):
            invalid_diversity_entries.append(
                f"{key}: diversity modifier must be numeric."
            )
            continue
        diversity_modifiers[key] = float(value)

    if invalid_diversity_entries:
        raise ValidationError(
            "Pitch arsenal diversity modifiers contain invalid entries.",
            invalid_diversity_entries,
        )

    development_raw = weights_raw.get("development", {})
    if not isinstance(development_raw, dict):
        raise ValidationError("'development' section must be a dictionary.")

    development_config: Dict[str, Dict[str, object]] = {}
    development_errors: List[str] = []
    for role in ["hitter", "pitcher"]:
        role_entry = development_raw.get(role)
        if role_entry is None:
            development_errors.append(f"development.{role} section is missing.")
            continue
        if not isinstance(role_entry, dict):
            development_errors.append(f"development.{role} must be a dictionary.")
            continue

        ratio_thresholds_raw = role_entry.get("ratio_thresholds")
        modifiers_raw = role_entry.get("modifiers")
        if not isinstance(ratio_thresholds_raw, dict):
            development_errors.append(
                f"development.{role}.ratio_thresholds must be a dictionary."
            )
            continue
        if not isinstance(modifiers_raw, dict):
            development_errors.append(
                f"development.{role}.modifiers must be a dictionary."
            )
            continue

        normalized_thresholds: Dict[str, float] = {}
        for key, value in ratio_thresholds_raw.items():
            try:
                normalized_thresholds[key] = float(value)
            except (TypeError, ValueError):
                development_errors.append(
                    f"development.{role}.ratio_thresholds.{key} must be numeric."
                )

        normalized_modifiers: Dict[str, float] = {}
        for key, value in modifiers_raw.items():
            try:
                normalized_modifiers[key] = float(value)
            except (TypeError, ValueError):
                development_errors.append(
                    f"development.{role}.modifiers.{key} must be numeric."
                )

        try:
            min_potential = float(role_entry.get("min_potential_for_bonus", 0.0))
        except (TypeError, ValueError):
            development_errors.append(
                f"development.{role}.min_potential_for_bonus must be numeric."
            )
            min_potential = 0.0

        development_config[role] = {
            "ratio_thresholds": normalized_thresholds,
            "modifiers": normalized_modifiers,
            "min_potential_for_bonus": min_potential,
        }

    if development_errors:
        raise ValidationError("Development configuration invalid.", development_errors)

    pitch_role_adjustments_raw = weights_raw.get("pitch_role_adjustments", {})
    if not isinstance(pitch_role_adjustments_raw, dict):
        raise ValidationError("'pitch_role_adjustments' must be a dictionary.")

    pitch_role_adjustments: Dict[str, float] = {}
    invalid_role_adjustments: List[str] = []
    for role, value in pitch_role_adjustments_raw.items():
        if role not in expected_roles:
            invalid_role_adjustments.append(
                f"{role}: unsupported role for pitch adjustments (expected {sorted(expected_roles)})"
            )
            continue
        try:
            pitch_role_adjustments[role] = float(value)
        except (TypeError, ValueError):
            invalid_role_adjustments.append(
                f"{role}: role adjustment must be numeric."
            )

    if invalid_role_adjustments:
        raise ValidationError(
            "Pitch role adjustments invalid.", invalid_role_adjustments
        )

    stamina_thresholds_raw = weights_raw.get("stamina_thresholds", {})
    if not isinstance(stamina_thresholds_raw, dict):
        raise ValidationError("'stamina_thresholds' must be a dictionary.")

    stamina_thresholds: Dict[str, Dict[str, float]] = {}
    invalid_stamina_entries: List[str] = []
    for role, threshold_config in stamina_thresholds_raw.items():
        if role not in expected_roles:
            invalid_stamina_entries.append(
                f"{role}: unsupported role for stamina thresholds (expected {sorted(expected_roles)})"
            )
            continue
        if not isinstance(threshold_config, dict):
            invalid_stamina_entries.append(
                f"{role}: stamina threshold config must be a dictionary."
            )
            continue
        required_keys = {"minimum", "penalty_per_point_below"}
        missing_keys = required_keys - set(threshold_config.keys())
        if missing_keys:
            invalid_stamina_entries.append(
                f"{role}: missing required keys {sorted(missing_keys)}"
            )
            continue
        try:
            stamina_thresholds[role] = {
                "minimum": float(threshold_config["minimum"]),
                "penalty_per_point_below": float(threshold_config["penalty_per_point_below"]),
            }
        except (TypeError, ValueError, KeyError) as e:
            invalid_stamina_entries.append(
                f"{role}: stamina threshold values must be numeric."
            )

    if invalid_stamina_entries:
        raise ValidationError(
            "Stamina thresholds invalid.", invalid_stamina_entries
        )

    bat_weights = weights_raw.get("bat_tools")
    if not isinstance(bat_weights, dict):
        raise ValidationError("Missing 'bat_tools' section in weights.json.")

    missing = [key for key in required_bat_keys if key not in bat_weights]
    if missing:
        raise ValidationError(
            "Weights file is missing bat tool entries.", [f"Missing keys: {missing}"]
        )

    defense_weights_raw = weights_raw.get("defense_tools")
    if defense_weights_raw is None:
        raise ValidationError("Missing 'defense_tools' section in weights.json.")
    if not isinstance(defense_weights_raw, dict):
        raise ValidationError("'defense_tools' section must be a dictionary.")

    defense_weights: Dict[str, Dict[str, float]] = {}
    invalid_entries: List[str] = []

    for position, metrics in defense_weights_raw.items():
        if not isinstance(metrics, dict):
            invalid_entries.append(f"{position}: weights must be a dictionary.")
            continue

        valid_metrics: Dict[str, float] = {}
        for metric_name, weight_value in metrics.items():
            if not isinstance(weight_value, (int, float)):
                invalid_entries.append(
                    f"{position}.{metric_name}: weight must be numeric."
                )
                continue
            valid_metrics[metric_name] = float(weight_value)

        if valid_metrics:
            defense_weights[position] = valid_metrics

    if invalid_entries:
        raise ValidationError(
            "Defense weights contain invalid entries.", invalid_entries
        )

    baserunning_weights = weights_raw.get("baserunning_tools")
    if baserunning_weights is None:
        raise ValidationError("Missing 'baserunning_tools' section in weights.json.")
    if not isinstance(baserunning_weights, dict):
        raise ValidationError("'baserunning_tools' section must be a dictionary.")

    missing_baserunning = [
        key for key in baserunning_keys if key not in baserunning_weights
    ]
    if missing_baserunning:
        raise ValidationError(
            "Weights file is missing baserunning entries.",
            [f"Missing keys: {missing_baserunning}"],
        )

    position_weights_raw = weights_raw.get("position_weights")
    if position_weights_raw is None:
        raise ValidationError("Missing 'position_weights' section in weights.json.")
    if not isinstance(position_weights_raw, dict):
        raise ValidationError("'position_weights' section must be a dictionary.")

    expected_categories = {"bat_tools", "defense_tools", "baserunning_tools"}
    missing_categories = sorted(expected_categories - set(position_weights_raw.keys()))
    if missing_categories:
        raise ValidationError(
            "Position weights are missing categories.", missing_categories
        )

    position_weights: Dict[str, Dict[str, float]] = {}
    invalid_position_entries: List[str] = []

    for category, mapping in position_weights_raw.items():
        if not isinstance(mapping, dict):
            invalid_position_entries.append(
                f"{category}: position mapping must be a dictionary."
            )
            continue

        normalized_mapping: Dict[str, float] = {}
        for position, weight in mapping.items():
            if not isinstance(weight, (int, float)):
                invalid_position_entries.append(
                    f"{category}.{position}: weight must be numeric."
                )
                continue
            normalized_mapping[position] = float(weight)

        position_weights[category] = normalized_mapping

    if invalid_position_entries:
        raise ValidationError(
            "Position weights contain invalid entries.", invalid_position_entries
        )

    personality_thresholds = weights_raw.get("personality_thresholds", {})
    if not isinstance(personality_thresholds, dict):
        raise ValidationError("'personality_thresholds' must be a dictionary.")

    # Provide defaults if keys missing
    high_threshold = personality_thresholds.get("high", 60)
    medium_threshold = personality_thresholds.get("medium", 45)
    try:
        personality_thresholds = {
            "high": float(high_threshold),
            "medium": float(medium_threshold),
        }
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Personality thresholds must be numeric values (high, medium)."
        ) from exc

    personality_modifiers_raw = weights_raw.get("personality_modifiers", {})
    if not isinstance(personality_modifiers_raw, dict):
        raise ValidationError("'personality_modifiers' must be a dictionary.")

    personality_modifiers: Dict[str, Dict[str, float]] = {}
    invalid_personality_entries: List[str] = []
    for trait in PERSONALITY_TRAITS:
        mapping = personality_modifiers_raw.get(trait)
        if mapping is None:
            continue
        if not isinstance(mapping, dict):
            invalid_personality_entries.append(
                f"{trait}: personality mapping must be a dictionary."
            )
            continue
        normalized_mapping: Dict[str, float] = {}
        for label, value in mapping.items():
            if label not in {"high", "medium", "low"}:
                invalid_personality_entries.append(
                    f"{trait}: invalid category '{label}' (expected high/medium/low)."
                )
                continue
            if not isinstance(value, (int, float)):
                invalid_personality_entries.append(
                    f"{trait}.{label}: value must be numeric."
                )
                continue
            normalized_mapping[label] = float(value)
        if normalized_mapping:
            personality_modifiers[trait] = normalized_mapping

    if invalid_personality_entries:
        raise ValidationError(
            "Personality modifiers contain invalid entries.",
            invalid_personality_entries,
        )

    defense_baselines_raw = weights_raw.get("defense_baselines", {})
    bat_baselines_raw = weights_raw.get("bat_baselines", {})
    baserunning_baselines_raw = weights_raw.get("baserunning_baselines", {})

    for baseline_name, baseline_raw in [
        ("defense_baselines", defense_baselines_raw),
        ("bat_baselines", bat_baselines_raw),
        ("baserunning_baselines", baserunning_baselines_raw),
    ]:
        if not isinstance(baseline_raw, dict):
            raise ValidationError(f"'{baseline_name}' must be a dictionary.")

    defense_baselines = _normalize_baseline_map(defense_baselines_raw, "defense")
    bat_baselines = _normalize_baseline_map(bat_baselines_raw, "bat")
    baserunning_baselines = _normalize_baseline_map(
        baserunning_baselines_raw, "baserunning"
    )

    age_level_raw = weights_raw.get("age_level")
    age_level_config = _normalize_age_level_config(age_level_raw)

    archetype_config = weights_raw.get("archetypes", {})

    return (
        pitch_ability_weights,
        pitch_type_weights,
        role_slot_weights,
        role_baselines,
        diversity_modifiers,
        development_config,
        pitch_role_adjustments,
        stamina_thresholds,
        bat_weights,
        defense_weights,
        baserunning_weights,
        position_weights,
        personality_thresholds,
        personality_modifiers,
        age_level_config,
        archetype_config,
        defense_baselines,
        bat_baselines,
        baserunning_baselines,
    )


def validate_columns(data: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Validate that the dataset contains all required columns.
    
    Checks that all specified required columns are present in the DataFrame.
    Used to ensure the player data file has all necessary fields for evaluation.
    
    Args:
        data: DataFrame containing player data
        required_columns: Iterable of column names that must be present
    
    Raises:
        ValidationError: If any required columns are missing from the dataset
    """
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValidationError(
            "Player data is missing required columns.", missing_columns
        )


def validate_defense_columns(
    data: pd.DataFrame, defense_weights: Dict[str, Dict[str, float]]
) -> None:
    """
    Validate that all required defensive columns exist in the dataset.
    
    Checks that for each position in the defense weights configuration,
    all required defensive metric columns are present in the dataset.
    Also validates that the metrics are supported for each position.
    
    Args:
        data: DataFrame containing player data
        defense_weights: Dictionary mapping positions to their defensive metric weights
    
    Raises:
        ValidationError: If defensive weights are missing or if required columns are missing
    """
    if not defense_weights:
        raise ValidationError(
            "No defensive weights configured.",
            ["Add weighted metrics under 'defense_tools' in weights.json."],
        )

    dataset_columns = set(data.columns)
    missing_by_position: List[str] = []

    for position, metrics in defense_weights.items():
        supported_columns = DEFENSE_TOOL_COLUMNS.get(position, [])
        required_columns = set(metrics.keys())

        unsupported = sorted(required_columns - set(supported_columns or []))
        if unsupported:
            missing_by_position.append(
                f"{position}: unsupported metrics {unsupported}"
            )
            continue

        missing = sorted(required_columns - dataset_columns)
        if missing:
            missing_by_position.append(f"{position}: missing columns {missing}")

    if missing_by_position:
        raise ValidationError(
            "Player data is missing defensive columns.", missing_by_position
        )


def validate_baserunning_columns(
    data: pd.DataFrame, baserunning_keys: Iterable[str]
) -> Dict[str, str]:
    """
    Validate baserunning columns and create a mapping to actual column names.
    
    Checks for the presence of baserunning metric columns, using preferred
    column name alternatives when available (e.g., "Steal" as fallback for "StealAbi").
    Returns a mapping from metric names to the actual column names found.
    
    Args:
        data: DataFrame containing player data
        baserunning_keys: List of baserunning metric names to validate
    
    Returns:
        Dictionary mapping metric names to actual column names in the dataset
    
    Raises:
        ValidationError: If any required baserunning columns are missing
    """
    column_mapping: Dict[str, str] = {}
    missing_columns: List[str] = []

    for metric in baserunning_keys:
        candidates = BASERUNNING_COLUMN_PREFERENCES.get(metric, [metric])
        for candidate in candidates:
            if candidate in data.columns:
                column_mapping[metric] = candidate
                break
        else:
            missing_columns.append(f"{metric} -> {candidates}")

    if missing_columns:
        raise ValidationError(
            "Player data is missing baserunning columns.", missing_columns
        )

    return column_mapping


def validate_pitcher_columns(
    data: pd.DataFrame,
    ability_columns: Iterable[str],
    arsenal_columns: Iterable[str],
) -> None:
    """
    Validate that all required pitcher-specific columns exist in the dataset.
    
    Checks that all pitch ability metrics (Stuff, Movement, Control, etc.) and
    pitch arsenal columns (Fastball, Curveball, etc.) are present in the dataset.
    
    Args:
        data: DataFrame containing player data
        ability_columns: List of pitch ability metric column names
        arsenal_columns: List of pitch type column names
    
    Raises:
        ValidationError: If any required pitcher columns are missing
    """
    missing_columns = sorted(
        column_name
        for column_name in set(ability_columns) | set(arsenal_columns)
        if column_name not in data.columns
    )
    if missing_columns:
        raise ValidationError(
            "Player data is missing pitcher-specific columns.", missing_columns
        )


def extract_player(data: pd.DataFrame, index: int = 0) -> pd.Series:
    """
    Extract a single player's data from the dataset by index.
    
    Retrieves a specific row from the DataFrame as a pandas Series,
    representing a single player's complete data record.
    
    Args:
        data: DataFrame containing all player data
        index: Zero-based index of the player row to extract (default: 0)
    
    Returns:
        Series containing the player's data with column names as index
    
    Raises:
        ValidationError: If the index is out of range for the dataset
    """
    try:
        player = data.iloc[index]
    except IndexError as exc:
        raise ValidationError(
            f"Player index {index} is out of range for the dataset."
        ) from exc

    return player


def calculate_bat_tools(
    player: pd.Series, weights: Dict[str, float]
) -> Tuple[Dict[str, float], float, Dict[str, Dict[str, float]]]:
    """
    Calculate weighted bat tool scores for a player.
    
    Evaluates a player's batting tools (Gap, Power, Eye, Ks) using
    their potential ratings. Applies rating adjustments and weights each tool
    according to the configuration. Returns individual scores, total, and detailed breakdown.
    
    Args:
        player: Series containing player data with potential bat tool columns
        weights: Dictionary mapping bat tool names to their weight values
    
    Returns:
        Tuple containing:
        - Dictionary mapping tool names to weighted scores
        - Total weighted bat tool score (sum of all weighted scores)
        - Dictionary with detailed breakdown for each tool (raw, adjusted, weight, weighted)
    """
    weighted_scores: Dict[str, float] = {}
    total_score = 0.0
    details: Dict[str, Dict[str, float]] = {}

    for raw_key, pot_key in zip(BAT_TOOL_HEADERS, BAT_TOOL_POT_HEADERS):
        raw_value = float(player[pot_key])
        adjusted_value = adjust_rating(raw_value)
        weight = float(weights[raw_key])
        score = adjusted_value * weight
        weighted_scores[raw_key] = score
        total_score += score
        details[raw_key] = {
            "raw": raw_value,
            "adjusted": adjusted_value,
            "weight": weight,
            "weighted": score,
        }

    return weighted_scores, total_score, details


def calculate_defense_tools(
    player: pd.Series,
    defense_weights: Dict[str, Dict[str, float]],
    defense_baselines: Dict[str, Dict[str, float]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]], Dict[str, float]
]:
    """
    Calculate weighted defensive tool scores for a player across all positions.
    
    Evaluates a player's defensive abilities at each position using position-specific
    metrics and weights. Applies baseline modifiers to adjust scores based on
    position-appropriate baselines. Returns scores for all positions evaluated.
    
    Args:
        player: Series containing player data with defensive metric columns
        defense_weights: Dictionary mapping positions to their metric weights
        defense_baselines: Dictionary mapping positions to their metric baselines
    
    Returns:
        Tuple containing:
        - Nested dictionary: position -> metric -> {rating, adjusted, baseline, modifier, weight, weighted}
        - Dictionary mapping positions to their total weighted defense scores
    
    Raises:
        ValidationError: If player has missing defensive ratings
    """
    defense_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    defense_totals: Dict[str, float] = {}
    missing_values: List[str] = []

    for position, metrics in defense_weights.items():
        position_scores: Dict[str, Dict[str, float]] = {}
        total = 0.0

        baselines_for_position = defense_baselines.get(position, {})

        for metric_name, weight in metrics.items():
            raw_value = player.get(metric_name)
            if pd.isna(raw_value):
                missing_values.append(f"{position}.{metric_name}")
                continue

            rating = float(raw_value)
            adjusted_rating = adjust_rating(rating)
            baseline_value = baselines_for_position.get(metric_name)
            modifier = 1.0
            if baseline_value is not None and baseline_value > 0:
                modifier = min(1.0, max(BASELINE_RATIO_FLOOR, rating / baseline_value))
            weighted_value = adjusted_rating * float(weight) * modifier

            position_scores[metric_name] = {
                "rating": rating,
                "adjusted": adjusted_rating,
                "baseline": baseline_value,
                "modifier": modifier,
                "weight": float(weight),
                "weighted": weighted_value,
            }
            total += weighted_value

        if position_scores:
            defense_scores[position] = position_scores
            defense_totals[position] = total

    if missing_values:
        raise ValidationError(
            "Player has missing defensive ratings.", missing_values
        )

    return defense_scores, defense_totals


def calculate_baserunning_tools(
    player: pd.Series,
    baserunning_weights: Dict[str, float],
    column_mapping: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Calculate weighted baserunning tool scores for a player.
    
    Evaluates a player's baserunning abilities (Speed, Steal Rate, Steal Ability, Run)
    using their ratings. Applies rating adjustments and weights each metric
    according to the configuration. Uses column mapping to find actual column names.
    
    Args:
        player: Series containing player data with baserunning metric columns
        baserunning_weights: Dictionary mapping metric names to their weight values
        column_mapping: Dictionary mapping metric names to actual column names in dataset
    
    Returns:
        Tuple containing:
        - Dictionary mapping metric names to detailed breakdowns (rating, adjusted, weight, weighted, column)
        - Total weighted baserunning score
    
    Raises:
        ValidationError: If player has missing baserunning ratings
    """
    scores: Dict[str, Dict[str, float]] = {}
    total = 0.0
    missing_metrics: List[str] = []

    for metric, weight in baserunning_weights.items():
        column_name = column_mapping.get(metric, metric)
        raw_value = player.get(column_name)
        if pd.isna(raw_value):
            missing_metrics.append(metric)
            continue

        rating = float(raw_value)
        adjusted_rating = adjust_rating(rating)
        weighted = adjusted_rating * float(weight)
        scores[metric] = {
            "rating": rating,
            "adjusted": adjusted_rating,
            "weight": float(weight),
            "weighted": weighted,
            "column": column_name,
        }
        total += weighted

    if missing_metrics:
        raise ValidationError(
            "Player has missing baserunning ratings.", missing_metrics
        )

    return scores, total


def apply_bat_baselines(
    bat_details: Dict[str, Dict[str, float]],
    baselines: Dict[str, Dict[str, float]],
    position: str,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Apply position-specific baselines to bat tool scores.
    
    Adjusts bat tool scores based on position-appropriate baselines. Creates
    modifiers that reduce scores when ratings are below baseline expectations
    for a given position. Returns adjusted scores and updated details.
    
    Args:
        bat_details: Dictionary of bat tool details (from calculate_bat_tools)
        baselines: Dictionary mapping positions to their bat tool baselines
        position: Position name to apply baselines for
    
    Returns:
        Tuple containing:
        - Total adjusted bat tool score after baseline modifiers
        - Updated bat details dictionary with baseline and modifier information
    """
    baseline_metrics = baselines.get(position, {})
    adjusted_details: Dict[str, Dict[str, float]] = {}
    total = 0.0

    for metric, detail in bat_details.items():
        baseline_value = baseline_metrics.get(metric)
        modifier = 1.0
        raw_value = detail["raw"]
        adjusted_value = detail["adjusted"]
        weight = detail["weight"]

        if baseline_value is not None and baseline_value > 0:
            modifier = min(1.0, max(BASELINE_RATIO_FLOOR, raw_value / baseline_value))

        weighted_value = adjusted_value * weight * modifier
        adjusted_details[metric] = {
            "raw": raw_value,
            "adjusted": adjusted_value,
            "baseline": baseline_value,
            "modifier": modifier,
            "weight": weight,
            "weighted": weighted_value,
        }
        total += weighted_value

    return total, adjusted_details


def apply_baserunning_baselines(
    baserunning_scores: Dict[str, Dict[str, float]],
    baselines: Dict[str, Dict[str, float]],
    position: str,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Apply position-specific baselines to baserunning scores.
    
    Adjusts baserunning scores based on position-appropriate baselines. Creates
    modifiers that reduce scores when ratings are below baseline expectations
    for a given position. Returns adjusted scores and updated details.
    
    Args:
        baserunning_scores: Dictionary of baserunning metric details
        baselines: Dictionary mapping positions to their baserunning baselines
        position: Position name to apply baselines for
    
    Returns:
        Tuple containing:
        - Total adjusted baserunning score after baseline modifiers
        - Updated baserunning details dictionary with baseline and modifier information
    """
    baseline_metrics = baselines.get(position, {})
    adjusted_details: Dict[str, Dict[str, float]] = {}
    total = 0.0

    for metric, values in baserunning_scores.items():
        baseline_value = baseline_metrics.get(metric)
        modifier = 1.0
        rating = values["rating"]
        adjusted_value = values["adjusted"]
        weight = values["weight"]

        if baseline_value is not None and baseline_value > 0:
            modifier = min(1.0, max(BASELINE_RATIO_FLOOR, rating / baseline_value))

        weighted_value = adjusted_value * weight * modifier
        adjusted_details[metric] = {
            "rating": rating,
            "adjusted": adjusted_value,
            "baseline": baseline_value,
            "modifier": modifier,
            "weight": weight,
            "weighted": weighted_value,
            "column": values.get("column"),
        }
        total += weighted_value

    return total, adjusted_details


def calculate_pitch_ability(
    player: pd.Series,
    role: str,
    pitch_weights: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Calculate weighted pitch ability scores for a pitcher.
    
    Evaluates a pitcher's core abilities (Stuff, Movement, Home Run Avoidance, Control)
    using their potential ratings. Applies rating adjustments and weights each metric
    according to role-specific weights (SP vs RP have different weightings).
    
    Args:
        player: Series containing player data with pitch ability metric columns
        role: Pitcher role ("SP" or "RP") to determine which weights to use
        pitch_weights: Dictionary mapping roles to their metric weights
    
    Returns:
        Tuple containing:
        - Dictionary mapping metric names to detailed breakdowns (rating, adjusted, weight, weighted, display)
        - Total weighted pitch ability score
    
    Raises:
        ValidationError: If role is not configured or player has missing pitch ability metrics
    """
    weights_for_role = pitch_weights.get(role)
    if weights_for_role is None:
        raise ValidationError(
            f"No pitch ability weights configured for role '{role}'.",
            [f"Available roles: {sorted(pitch_weights.keys())}"],
        )

    details: Dict[str, Dict[str, float]] = {}
    total = 0.0
    missing_metrics: List[str] = []

    for column_name in PITCH_ABILITY_METRICS:
        weight = weights_for_role.get(column_name)
        if weight is None:
            continue
        raw_value = player.get(column_name)
        if pd.isna(raw_value):
            missing_metrics.append(column_name)
            continue
        rating = float(raw_value)
        adjusted = adjust_rating(rating)
        weighted = adjusted * float(weight)
        display_name = PITCH_ABILITY_DISPLAY.get(column_name, column_name)
        details[column_name] = {
            "rating": rating,
            "adjusted": adjusted,
            "weight": float(weight),
            "weighted": weighted,
            "display": display_name,
        }
        total += weighted

    if missing_metrics:
        raise ValidationError(
            f"Player missing pitcher metrics for role '{role}'.", missing_metrics
        )

    return details, total


def _slot_sort_key(slot_name: str) -> Tuple[int, str]:
    """
    Generate a sort key for pitch slot names to ensure proper ordering.
    
    Extracts numeric digits from slot names (e.g., "slot1", "slot2") to enable
    numeric sorting rather than alphabetical. Used to order pitch slots correctly.
    
    Args:
        slot_name: Pitch slot name string (e.g., "slot1", "slot2")
    
    Returns:
        Tuple of (extracted integer, original string) for sorting
    """
    digits = "".join(ch for ch in slot_name if ch.isdigit())
    index = int(digits) if digits else 0
    return index, slot_name


def calculate_pitch_arsenal(
    player: pd.Series,
    role: str,
    pitch_type_weights: Dict[str, float],
    role_slot_weights: Dict[str, Dict[str, float]],
    role_baselines: Dict[str, Dict[str, float]],
    diversity_modifiers: Dict[str, float],
) -> Tuple[List[Dict[str, object]], Dict[str, object], float, float, float]:
    """
    Calculate weighted pitch arsenal scores for a pitcher.
    
    Evaluates a pitcher's complete pitch arsenal by:
    1. Collecting all usable pitches (ratings > 0)
    2. Sorting pitches by value (adjusted rating * type weight)
    3. Assigning pitches to slots based on role-specific slot weights
    4. Applying diversity bonuses/penalties for speed tiers, break planes, and pitch count
    5. Calculating final arsenal score
    
    Args:
        player: Series containing player data with pitch type columns
        role: Pitcher role ("SP" or "RP") to determine slot weights and baselines
        pitch_type_weights: Dictionary mapping pitch types to their base weights
        role_slot_weights: Dictionary mapping roles to their slot weight configurations
        role_baselines: Dictionary mapping roles to their arsenal baseline requirements
        diversity_modifiers: Dictionary of diversity bonus/penalty values
    
    Returns:
        Tuple containing:
        - List of pitch entry dictionaries with all pitch details
        - Diversity summary dictionary (pitch count, speed tiers, break planes, adjustments)
        - Total slot-weighted arsenal contribution
        - Total diversity adjustment (bonuses/penalties)
        - Final arsenal total (slot contribution + diversity adjustment)
    
    Raises:
        ValidationError: If role configuration is missing or player has no usable pitches
    """
    slot_weights = role_slot_weights.get(role)
    if slot_weights is None:
        raise ValidationError(
            f"No pitch arsenal slot weights configured for role '{role}'."
        )
    baselines = role_baselines.get(role)
    if baselines is None:
        raise ValidationError(
            f"No pitch arsenal baselines configured for role '{role}'."
        )

    pitch_entries: List[Dict[str, object]] = []
    for column in PITCH_ARSENAL_COLUMNS:
        pitch_type = PITCH_TYPE_BY_COLUMN[column]
        raw_value = player.get(column)
        if pd.isna(raw_value):
            continue
        rating = float(raw_value)
        if rating <= 0:
            continue
        adjusted = adjust_rating(rating)
        type_weight = pitch_type_weights.get(pitch_type, 0.0)
        base_value = adjusted * type_weight
        speed_tier = PITCH_SPEED_TIERS.get(pitch_type)
        break_plane = PITCH_BREAK_PLANES.get(pitch_type)
        pitch_entries.append(
            {
                "column": column,
                "pitch_type": pitch_type,
                "rating": rating,
                "adjusted": adjusted,
                "type_weight": type_weight,
                "base_value": base_value,
                "speed_tier": speed_tier,
                "break_plane": break_plane,
                "slot": None,
                "slot_weight": 0.0,
                "slot_contribution": 0.0,
            }
        )

    if not pitch_entries:
        raise ValidationError(
            f"Player has no usable pitch ratings for role '{role}'.",
            [column for column in PITCH_ARSENAL_COLUMNS],
        )

    pitch_entries.sort(key=lambda entry: entry["base_value"], reverse=True)  # type: ignore[arg-type]

    slot_order = sorted(slot_weights.keys(), key=_slot_sort_key)
    total_slot_contribution = 0.0
    for index, slot_name in enumerate(slot_order):
        if index >= len(pitch_entries):
            break
        entry = pitch_entries[index]
        slot_weight = float(slot_weights[slot_name])
        slot_contribution = entry["base_value"] * slot_weight  # type: ignore[operator]
        entry["slot"] = slot_name
        entry["slot_weight"] = slot_weight
        entry["slot_contribution"] = slot_contribution
        total_slot_contribution += slot_contribution

    pitch_count = len(pitch_entries)
    unique_speed_tiers = {
        entry["speed_tier"] for entry in pitch_entries if entry["speed_tier"]
    }
    unique_break_planes = {
        entry["break_plane"] for entry in pitch_entries if entry["break_plane"]
    }

    adjustments: List[Dict[str, object]] = []
    diversity_total = 0.0

    speed_tier_bonus = float(diversity_modifiers.get("speed_tier_bonus", 0.0))
    if unique_speed_tiers and len(unique_speed_tiers) >= baselines.get(
        "min_speed_tiers", 0
    ):
        diversity_total += speed_tier_bonus
        adjustments.append(
            {
                "category": "speed_tiers",
                "value": speed_tier_bonus,
                "met": True,
                "count": len(unique_speed_tiers),
                "required": baselines.get("min_speed_tiers"),
            }
        )
    else:
        adjustments.append(
            {
                "category": "speed_tiers",
                "value": 0.0,
                "met": False,
                "count": len(unique_speed_tiers),
                "required": baselines.get("min_speed_tiers"),
            }
        )

    break_plane_bonus = float(diversity_modifiers.get("break_plane_bonus", 0.0))
    if unique_break_planes and len(unique_break_planes) >= baselines.get(
        "min_break_planes", 0
    ):
        diversity_total += break_plane_bonus
        adjustments.append(
            {
                "category": "break_planes",
                "value": break_plane_bonus,
                "met": True,
                "count": len(unique_break_planes),
                "required": baselines.get("min_break_planes"),
            }
        )
    else:
        adjustments.append(
            {
                "category": "break_planes",
                "value": 0.0,
                "met": False,
                "count": len(unique_break_planes),
                "required": baselines.get("min_break_planes"),
            }
        )

    redundancy_penalty = float(diversity_modifiers.get("redundancy_penalty", 0.0))
    if baselines.get("min_pitches") and pitch_count < baselines["min_pitches"]:
        diversity_total += redundancy_penalty
        adjustments.append(
            {
                "category": "pitch_count",
                "value": redundancy_penalty,
                "met": False,
                "count": pitch_count,
                "required": baselines.get("min_pitches"),
            }
        )
    else:
        adjustments.append(
            {
                "category": "pitch_count",
                "value": 0.0,
                "met": True,
                "count": pitch_count,
                "required": baselines.get("min_pitches"),
            }
        )

    diversity_summary = {
        "pitch_count": pitch_count,
        "speed_tiers": sorted(unique_speed_tiers),
        "break_planes": sorted(unique_break_planes),
        "baselines": baselines,
        "adjustments": adjustments,
    }

    final_total = total_slot_contribution + diversity_total

    return (
        pitch_entries,
        diversity_summary,
        total_slot_contribution,
        diversity_total,
        final_total,
    )


def combine_pitch_scores(
    role: str,
    ability_total: float,
    arsenal_total: float,
    position_weights: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Combine pitch ability and arsenal scores into a single weighted total.
    
    Takes the separate pitch ability and arsenal scores and combines them
    using role-specific weights. Different roles (SP vs RP) may weight
    ability and arsenal differently.
    
    Args:
        role: Pitcher role ("SP" or "RP") to determine combination weights
        ability_total: Total weighted pitch ability score
        arsenal_total: Total weighted pitch arsenal score
        position_weights: Dictionary containing pitch_ability and pitch_arsenal weight categories
    
    Returns:
        Tuple containing:
        - Breakdown dictionary with ability and arsenal scores, weights, and weighted values
        - Combined total score (weighted ability + weighted arsenal)
    
    Raises:
        ValidationError: If position weights are missing required categories or role entries
    """
    ability_weights = position_weights.get("pitch_ability")
    arsenal_weights = position_weights.get("pitch_arsenal")

    if ability_weights is None or arsenal_weights is None:
        raise ValidationError(
            "Position weights missing 'pitch_ability' or 'pitch_arsenal' categories."
        )

    ability_weight = ability_weights.get(role)
    arsenal_weight = arsenal_weights.get(role)

    if ability_weight is None or arsenal_weight is None:
        details: List[str] = []
        if ability_weight is None:
            details.append(f"pitch_ability.{role}")
        if arsenal_weight is None:
            details.append(f"pitch_arsenal.{role}")
        raise ValidationError(
            f"Position '{role}' missing pitch weighting entries.", details
        )

    breakdown = {
        "pitch_ability": {
            "score": ability_total,
            "weight": float(ability_weight),
            "weighted": ability_total * float(ability_weight),
        },
        "pitch_arsenal": {
            "score": arsenal_total,
            "weight": float(arsenal_weight),
            "weighted": arsenal_total * float(arsenal_weight),
        },
    }

    combined_total = breakdown["pitch_ability"]["weighted"] + breakdown[
        "pitch_arsenal"
    ]["weighted"]

    return breakdown, combined_total


def calculate_development_adjustment(
    player: pd.Series,
    role_config: Dict[str, object],
    current_metrics: Iterable[str],
    potential_metrics: Iterable[str],
) -> Tuple[float, Dict[str, object]]:
    """
    Calculate development adjustment based on current vs potential ratings.
    
    Evaluates how developed a player is by comparing their current ratings
    to their potential ratings. Players who are ahead of development get bonuses,
    while players lagging behind get penalties. The adjustment is based on
    the ratio of average current to average potential ratings.
    
    Args:
        player: Series containing player data with current and potential metric columns
        role_config: Configuration dictionary for the role (hitter or pitcher)
        current_metrics: List of column names for current ability ratings
        potential_metrics: List of column names for potential ability ratings
    
    Returns:
        Tuple containing:
        - Development adjustment value (positive for ahead, negative for behind)
        - Details dictionary with avg_current, avg_potential, ratio, bucket, modifier, reason
    """
    if not role_config:
        return 0.0, {}

    def _collect_average(columns: Iterable[str]) -> Optional[float]:
        values: List[float] = []
        for column in columns:
            raw = player.get(column)
            if pd.isna(raw):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return sum(values) / len(values)

    avg_current = _collect_average(current_metrics)
    avg_potential = _collect_average(potential_metrics)

    details: Dict[str, object] = {
        "avg_current": avg_current,
        "avg_potential": avg_potential,
        "ratio": None,
        "bucket_initial": "neutral",
        "bucket_applied": "neutral",
        "modifier": 0.0,
        "reason": "",
    }

    if avg_current is None or avg_potential is None or avg_potential <= 0:
        details["reason"] = "insufficient data"
        return 0.0, details

    ratio = avg_current / avg_potential
    details["ratio"] = ratio

    ratio_thresholds_raw = role_config.get("ratio_thresholds", {})
    modifiers_raw = role_config.get("modifiers", {})
    min_potential = float(role_config.get("min_potential_for_bonus", 0.0))

    if not isinstance(ratio_thresholds_raw, dict) or not isinstance(
        modifiers_raw, dict
    ):
        details["reason"] = "invalid configuration"
        return 0.0, details

    fast_threshold = float(ratio_thresholds_raw.get("fast_track", float("inf")))
    ahead_threshold = float(ratio_thresholds_raw.get("ahead", fast_threshold))
    lag_threshold = float(ratio_thresholds_raw.get("lag", 0.0))

    bucket = "neutral"
    if ratio >= fast_threshold:
        bucket = "fast_track"
    elif ratio >= ahead_threshold:
        bucket = "ahead"
    elif ratio <= lag_threshold:
        bucket = "lag"

    applied_bucket = bucket
    if avg_potential < min_potential and bucket in {"fast_track", "ahead"}:
        applied_bucket = "neutral"
        details["reason"] = "potential below bonus threshold"

    modifier = float(modifiers_raw.get(applied_bucket, 0.0))

    details.update(
        {
            "bucket_initial": bucket,
            "bucket_applied": applied_bucket,
            "modifier": modifier,
            "min_potential_for_bonus": min_potential,
        }
    )

    return modifier, details


def calculate_age_adjustment(
    age_value: object,
    league_label: str,
    age_config: Dict[str, Any],
) -> Tuple[float, Dict[str, object]]:
    """
    Calculate age adjustment based on player age relative to league level expectations.
    
    Evaluates whether a player is ahead of or behind the expected age for their
    league level. Players younger than the target age get bonuses, while older
    players get penalties. The adjustment is proportional to how far from the target
    age the player is, within a configurable band.
    
    Args:
        age_value: Player's age (can be int, float, string, or None)
        league_label: League level label to determine age expectations
        age_config: Age level configuration dictionary (from _normalize_age_level_config)
    
    Returns:
        Tuple containing:
        - Age adjustment value (positive for ahead, negative for behind)
        - Details dictionary with age, target, delta, band, bucket, ratio, adjustment, reason
    """
    details: Dict[str, object] = {
        "age": None,
        "level": league_label,
        "target": None,
        "delta": None,
        "band": None,
        "ahead_bonus": None,
        "behind_penalty": None,
        "bucket": "neutral",
        "ratio": 0.0,
        "adjustment": 0.0,
        "reason": "",
    }

    age = normalize_float(age_value)
    if age is None:
        details["reason"] = "age unavailable"
        return 0.0, details

    details["age"] = age

    if not age_config:
        details["reason"] = "age configuration unavailable"
        return 0.0, details

    default_cfg = age_config.get("default", {})
    levels_cfg = age_config.get("levels", {})

    cfg = levels_cfg.get(league_label) or default_cfg
    if not cfg:
        details["reason"] = "age defaults unavailable"
        return 0.0, details

    target = float(cfg.get("target", age))
    band = max(0.1, float(cfg.get("band", default_cfg.get("band", 2.0))))
    ahead_bonus = float(cfg.get("ahead_bonus", default_cfg.get("ahead_bonus", 0.0)))
    behind_penalty = float(
        cfg.get("behind_penalty", default_cfg.get("behind_penalty", 0.0))
    )

    delta = age - target
    details.update(
        {
            "target": target,
            "delta": delta,
            "band": band,
            "ahead_bonus": ahead_bonus,
            "behind_penalty": behind_penalty,
        }
    )

    if delta < 0:
        ratio = min(1.0, abs(delta) / band)
        adjustment = ahead_bonus * ratio
        bucket = "ahead" if ratio > 0 else "neutral"
    elif delta > 0:
        ratio = min(1.0, delta / band)
        adjustment = behind_penalty * ratio
        bucket = "behind" if ratio > 0 else "neutral"
    else:
        ratio = 0.0
        adjustment = 0.0
        bucket = "neutral"

    details["ratio"] = ratio
    details["bucket"] = bucket
    details["adjustment"] = adjustment

    return adjustment, details


def classify_hitter_archetype(
    position: str,
    bat_total: float,
    defense_total: float,
    baserunning_total: float,
    bat_details: Dict[str, Dict[str, float]],
    baserunning_details: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Tuple[str, Dict[str, object]]:
    """
    Classify a hitter into an archetype based on their skill profile.
    
    Analyzes the relative contributions of batting, defense, and baserunning
    to determine the player's archetype. Considers power ratios, gap/eye
    skills, and overall balance. Returns archetypes like "power_slugger",
    "table_setter", "defensive_wizard", "speedster", "balanced_regular", etc.
    
    Args:
        position: Player's position
        bat_total: Total weighted bat tool score
        defense_total: Total weighted defense score for the position
        baserunning_total: Total weighted baserunning score
        bat_details: Dictionary of bat tool details (for power/gap/eye ratios)
        baserunning_details: Dictionary of baserunning metric details
        config: Archetype classification configuration with thresholds
    
    Returns:
        Tuple containing:
        - Archetype label string (e.g., "power_slugger", "defensive_wizard")
        - Details dictionary with shares, ratios, and summary explanation
    """
    cfg = config or {}
    total = bat_total + defense_total + baserunning_total
    if total <= 0:
        return "insufficient_data", {
            "summary": "No component totals available.",
            "bat_share": 0.0,
            "defense_share": 0.0,
            "baserunning_share": 0.0,
        }

    min_total = float(cfg.get("min_total", 20.0))
    offense_share_threshold = float(cfg.get("offense_share", 0.5))
    defense_share_threshold = float(cfg.get("defense_share", 0.45))
    speed_share_threshold = float(cfg.get("speed_share", 0.3))
    speed_vs_offense = float(cfg.get("speed_vs_offense", 0.8))
    power_ratio_threshold = float(cfg.get("power_ratio", 0.35))
    table_gap_threshold = float(cfg.get("table_setter_gap", 0.32))
    table_eye_threshold = float(cfg.get("table_setter_eye", 0.18))
    balanced_tolerance = float(cfg.get("balanced_tolerance", 0.1))

    bat_share = bat_total / total
    defense_share = defense_total / total
    baserunning_share = baserunning_total / total

    pow_weighted = bat_details.get("Pow", {}).get("weighted", 0.0)
    gap_weighted = bat_details.get("Gap", {}).get("weighted", 0.0)
    eye_weighted = bat_details.get("Eye", {}).get("weighted", 0.0)

    bat_total_safe = max(bat_total, 1e-6)
    pow_ratio = pow_weighted / bat_total_safe
    gap_ratio = gap_weighted / bat_total_safe
    eye_ratio = eye_weighted / bat_total_safe

    details: Dict[str, object] = {
        "position": position,
        "bat_share": bat_share,
        "defense_share": defense_share,
        "baserunning_share": baserunning_share,
        "power_ratio": pow_ratio,
        "gap_ratio": gap_ratio,
        "eye_ratio": eye_ratio,
    }

    if total < min_total:
        details["summary"] = "Overall contribution below configurable minimum."
        return "depth_piece", details

    if defense_share >= defense_share_threshold and defense_total >= bat_total:
        details["summary"] = "Defensive impact outweighs offensive value."
        return "defensive_wizard", details

    if (
        baserunning_share >= speed_share_threshold
        and baserunning_total >= bat_total * speed_vs_offense
    ):
        details["summary"] = "Baserunning value drives the profile."
        return "speedster", details

    if bat_share >= offense_share_threshold:
        if pow_ratio >= power_ratio_threshold and pow_weighted > gap_weighted:
            details["summary"] = "Power makes up the largest share of offensive value."
            return "power_slugger", details

        if (
            gap_ratio >= table_gap_threshold
            and eye_ratio >= table_eye_threshold
            and pow_ratio < power_ratio_threshold
        ):
            details["summary"] = "Gap power and on-base skills lead the profile."
            return "table_setter", details

        details["summary"] = "Offense-first profile without a single standout skill."
        return "run_producer", details

    if (
        abs(bat_share - defense_share) <= balanced_tolerance
        and abs(bat_share - baserunning_share) <= balanced_tolerance
    ):
        details["summary"] = "Well-rounded contributions across offense, defense, and speed."
        return "balanced_regular", details

    if defense_share >= defense_share_threshold:
        details["summary"] = "Defense-heavy contribution paired with adequate offense."
        return "glove_first_regular", details

    if baserunning_share >= speed_share_threshold:
        details["summary"] = "Speed complements modest offensive output."
        return "utility_speedster", details

    details["summary"] = "Solid all-around contributions."
    return "balanced_regular", details


def classify_pitcher_archetype(
    role: str,
    ability_total: float,
    arsenal_total: float,
    combined_total: float,
    ability_details: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Tuple[str, Dict[str, object]]:
    """
    Classify a pitcher into an archetype based on their skill profile.
    
    Analyzes the pitcher's ability scores, combined totals, and relative
    strengths (stuff vs control vs movement) to determine their archetype.
    Different thresholds apply for starters vs relievers. Returns archetypes
    like "rotation_anchor", "pitchability_starter", "high_leverage_reliever",
    "power_reliever", etc.
    
    Args:
        role: Pitcher role ("SP" or "RP")
        ability_total: Total weighted pitch ability score
        arsenal_total: Total weighted pitch arsenal score
        combined_total: Combined ability + arsenal score
        ability_details: Dictionary of pitch ability details (for stuff/control/movement ratios)
        config: Archetype classification configuration with thresholds
    
    Returns:
        Tuple containing:
        - Archetype label string (e.g., "rotation_anchor", "power_reliever")
        - Details dictionary with shares, totals, and summary explanation
    """
    cfg = config or {}
    ability_total_safe = max(ability_total, 1e-6)

    stuff = ability_details.get("PotStf", {}).get("weighted", 0.0)
    control = ability_details.get("PotCtrl", {}).get("weighted", 0.0)
    movement = ability_details.get("PotMov", {}).get("weighted", 0.0)

    stuff_share = stuff / ability_total_safe
    control_share = control / ability_total_safe
    movement_share = movement / ability_total_safe

    details: Dict[str, object] = {
        "role": role,
        "ability_total": ability_total,
        "arsenal_total": arsenal_total,
        "combined_total": combined_total,
        "stuff_share": stuff_share,
        "control_share": control_share,
        "movement_share": movement_share,
    }

    ace_ability = float(cfg.get("ace_ability", 40.0))
    ace_combined = float(cfg.get("ace_combined", 75.0))
    mid_ability = float(cfg.get("mid_ability", 30.0))
    mid_combined = float(cfg.get("mid_combined", 55.0))
    back_combined = float(cfg.get("back_combined", 40.0))
    control_focus = float(cfg.get("control_focus", 0.35))
    stuff_focus = float(cfg.get("stuff_focus", 0.45))
    high_lev_ability = float(cfg.get("high_leverage_ability", 32.0))
    high_lev_combined = float(cfg.get("high_leverage_combined", 48.0))
    swingman_combined = float(cfg.get("swingman_combined", 35.0))

    label = "developmental_arm"
    summary = "Needs additional development time."

    if role == "SP":
        if ability_total >= ace_ability and combined_total >= ace_combined:
            label = "rotation_anchor"
            summary = "Ability and combined totals meet ace thresholds."
        elif control_share >= control_focus and stuff_share < stuff_focus:
            label = "pitchability_starter"
            summary = "Command-driven starter profile."
        elif ability_total >= mid_ability and combined_total >= mid_combined:
            label = "mid_rotation_starter"
            summary = "Solid ability totals for middle-rotation role."
        elif combined_total >= back_combined:
            label = "back_rotation_starter"
            summary = "Profiles as a depth starter."
        elif combined_total >= swingman_combined:
            label = "swingman"
            summary = "Could toggle between rotation and bullpen."
    else:
        if ability_total >= high_lev_ability and combined_total >= high_lev_combined:
            label = "high_leverage_reliever"
            summary = "Ability metrics support leverage bullpen work."
        elif stuff_share >= stuff_focus and ability_total >= mid_ability:
            label = "power_reliever"
            summary = "Stuff-driven relief profile."
        elif control_share >= control_focus and ability_total >= swingman_combined:
            label = "control_reliever"
            summary = "Command-first bullpen arm."
        elif combined_total >= swingman_combined:
            label = "middle_reliever"
            summary = "Adequate arsenal for middle relief."

    details["summary"] = summary
    return label, details


def evaluate_pitcher(
    player: pd.Series,
    role: str,
    pitch_ability_weights: Dict[str, Dict[str, float]],
    pitch_type_weights: Dict[str, float],
    pitch_role_slot_weights: Dict[str, Dict[str, float]],
    pitch_role_baselines: Dict[str, Dict[str, float]],
    pitch_diversity_modifiers: Dict[str, float],
    position_weights: Dict[str, Dict[str, float]],
    development_config: Dict[str, Dict[str, object]],
    age_config: Dict[str, Any],
    archetype_config: Dict[str, Any],
    pitch_role_adjustments: Dict[str, float],
    stamina_thresholds: Dict[str, Dict[str, float]],
    team_lookup: Dict[int, Dict[str, object]],
    league_lookup: Dict[int, str],
) -> Tuple[str, Dict[str, object]]:
    """
    Complete evaluation of a pitcher's overall value and profile.
    
    Performs a comprehensive evaluation of a pitcher by:
    1. Calculating pitch ability scores (Stuff, Movement, Control, etc.)
    2. Calculating pitch arsenal scores (pitch types, diversity, etc.)
    3. Combining ability and arsenal scores
    4. Applying development and age adjustments
    5. Classifying the pitcher's archetype
    6. Applying role-specific multipliers
    7. Generating a detailed report and summary
    8. Evaluating both SP and RP roles
    
    Args:
        player: Series containing all pitcher data
        role: Pitcher role ("SP" or "RP") - used for the detailed report
        pitch_ability_weights: Dictionary of role -> metric -> weight
        pitch_type_weights: Dictionary of pitch type -> weight
        pitch_role_slot_weights: Dictionary of role -> slot -> weight
        pitch_role_baselines: Dictionary of role -> baseline requirements
        pitch_diversity_modifiers: Dictionary of diversity bonus/penalty values
        position_weights: Dictionary containing pitch_ability and pitch_arsenal weight categories
        development_config: Development adjustment configuration
        age_config: Age adjustment configuration
        archetype_config: Archetype classification configuration
        pitch_role_adjustments: Dictionary of role -> multiplier
        stamina_thresholds: Dictionary of role -> threshold configuration
        team_lookup: Dictionary mapping team IDs to team information
        league_lookup: Dictionary mapping league level IDs to labels
    
    Returns:
        Tuple containing:
        - Detailed formatted report string (for the declared role)
        - Summary dictionary with key evaluation metrics for CSV export (includes both SP and RP scores)
    """
    # Calculate shared values once (same for both SP and RP)
    team_label = resolve_team_label(player.get("Team"), team_lookup)
    org_label = resolve_team_label(player.get("Org"), team_lookup)
    league_label = resolve_league_label(player.get("LgLvl"), league_lookup)
    development_adjust, development_details = calculate_development_adjustment(
        player,
        development_config.get("pitcher", {}),
        PITCH_CURRENT_METRICS,
        PITCH_ABILITY_METRICS,
    )
    age_adjust, age_details = calculate_age_adjustment(
        player.get("Age"), league_label, age_config
    )
    
    # Evaluate both SP and RP roles
    role_results: Dict[str, Dict[str, object]] = {}
    
    for eval_role in ["SP", "RP"]:
        try:
            ability_details, ability_total = calculate_pitch_ability(
                player, eval_role, pitch_ability_weights
            )
            (
                arsenal_entries,
                diversity_summary,
                arsenal_slot_total,
                diversity_total,
                arsenal_total,
            ) = calculate_pitch_arsenal(
                player,
                eval_role,
                pitch_type_weights,
                pitch_role_slot_weights,
                pitch_role_baselines,
                pitch_diversity_modifiers,
            )
            combination_breakdown, combined_total = combine_pitch_scores(
                eval_role,
                ability_total,
                arsenal_total,
                position_weights,
            )
            archetype_label, archetype_details = classify_pitcher_archetype(
                eval_role,
                ability_total,
                arsenal_total,
                combined_total,
                ability_details,
                archetype_config,
            )
            role_adjustment = pitch_role_adjustments.get(eval_role, 1.0)
            adjusted_total = combined_total + development_adjust + age_adjust
            
            # Apply stamina penalty for SP role if stamina is below threshold
            stamina_penalty = 0.0
            if eval_role == "SP" and "SP" in stamina_thresholds:
                stamina_config = stamina_thresholds["SP"]
                stamina_minimum = stamina_config.get("minimum", 45.0)
                penalty_per_point = stamina_config.get("penalty_per_point_below", 1.5)
                
                # Try to get stamina value (common column names: "Stm", "Stam", "Stamina", "Sta")
                stamina_value = None
                for col_name in ["Stm", "Stam", "Stamina", "Sta"]:
                    if col_name in player.index:
                        stamina_raw = player.get(col_name)
                        if not pd.isna(stamina_raw):
                            try:
                                stamina_value = float(stamina_raw)
                                break
                            except (TypeError, ValueError):
                                continue
                
                if stamina_value is not None and stamina_value < stamina_minimum:
                    points_below = stamina_minimum - stamina_value
                    stamina_penalty = points_below * penalty_per_point
            
            adjusted_total_with_stamina = adjusted_total - stamina_penalty
            final_total = adjusted_total_with_stamina * role_adjustment
            total_adjustment = final_total - combined_total
            
            role_results[eval_role] = {
                "ability_details": ability_details,
                "ability_total": ability_total,
                "arsenal_entries": arsenal_entries,
                "arsenal_slot_total": arsenal_slot_total,
                "diversity_summary": diversity_summary,
                "diversity_total": diversity_total,
                "arsenal_total": arsenal_total,
                "combination_breakdown": combination_breakdown,
                "combined_total": combined_total,
                "archetype_label": archetype_label,
                "archetype_details": archetype_details,
                "final_total": final_total,
                "development_adjust": development_adjust,
                "age_adjust": age_adjust,
                "role_adjustment": role_adjustment,
                "total_adjustment": total_adjustment,
            }
        except ValidationError:
            # If evaluation fails for a role, skip it
            continue
    
    if not role_results:
        raise ValidationError(
            "Unable to evaluate pitcher for any role.",
            [f"Player ID: {player.get('ID')}"],
        )
    
    # Use the declared role for the report, or the first available role
    report_role = role if role in role_results else list(role_results.keys())[0]
    report_data = role_results[report_role]
    
    # Determine best role (highest final_total)
    best_role = max(role_results.keys(), key=lambda r: role_results[r]["final_total"])  # type: ignore[arg-type]
    best_data = role_results[best_role]
    
    # Get shared values (already calculated above)
    player_id = player.get("ID")
    
    # Generate report for the declared role (or best if declared not available)
    report = format_pitch_report(
        player,
        report_role,
        report_data["ability_details"],  # type: ignore[index]
        report_data["ability_total"],  # type: ignore[index]
        report_data["arsenal_entries"],  # type: ignore[index]
        report_data["arsenal_slot_total"],  # type: ignore[index]
        report_data["diversity_summary"],  # type: ignore[index]
        report_data["diversity_total"],  # type: ignore[index]
        report_data["arsenal_total"],  # type: ignore[index]
        report_data["combination_breakdown"],  # type: ignore[index]
        report_data["combined_total"],  # type: ignore[index]
        org_label,
        team_label,
        league_label,
        age_details,
        report_data["age_adjust"],  # type: ignore[index]
        report_data["archetype_label"],  # type: ignore[index]
        report_data["archetype_details"],  # type: ignore[index]
        development_details,
        report_data["development_adjust"],  # type: ignore[index]
        report_data["role_adjustment"],  # type: ignore[index]
        report_data["total_adjustment"],  # type: ignore[index]
        report_data["final_total"],  # type: ignore[index]
    )
    
    # Get SP and RP scores
    sp_score = round_optional(role_results["SP"]["final_total"]) if "SP" in role_results else None  # type: ignore[index]
    rp_score = round_optional(role_results["RP"]["final_total"]) if "RP" in role_results else None  # type: ignore[index]
    
    # Use the declared role's values for current, or best if declared not available
    current_role = role if role in role_results else best_role
    current_data = role_results[current_role]
    
    summary = {
        "ID": player_id,
        "Pos": player.get("Pos"),
        "Name": player.get("Name"),
        "B": player.get("Bats"),
        "T": player.get("Throws"),
        "Current Value": round_optional(current_data["final_total"]),  # type: ignore[index]
        "Ideal Pos": best_role,
        "Ideal Value": round_optional(best_data["final_total"]),  # type: ignore[index]
        "Bat Score": None,
        "Defense Score": None,
        "Baserunning Score": None,
        "Pitching Ability Score": round_optional(best_data["ability_total"]),  # type: ignore[index]
        "Pitching Arsenal Score": round_optional(best_data["arsenal_total"]),  # type: ignore[index]
        "Development Adjustment": round_optional(current_data["development_adjust"]),  # type: ignore[index]
        "Age Adjustment": round_optional(current_data["age_adjust"]),  # type: ignore[index]
        "Archetype": current_data["archetype_label"],  # type: ignore[index]
        "Archetype Detail": current_data["archetype_details"].get("summary", ""),  # type: ignore[index]
        "Ideal Archetype": best_data["archetype_label"],  # type: ignore[index]
        "Ideal Archetype Detail": best_data["archetype_details"].get("summary", ""),  # type: ignore[index]
        "Role Multiplier": round_optional(current_data["role_adjustment"]),  # type: ignore[index]
        "Organization": org_label,
        "Team": team_label,
        "League Level": league_label,
        "Role Adjustment": round_optional(current_data["total_adjustment"]),  # type: ignore[index]
        "SP Score": sp_score,
        "RP Score": rp_score,
    }

    return report, summary


def categorize_personality_rating(
    rating: float, thresholds: Dict[str, float]
) -> str:
    """
    Categorize a personality rating into high/medium/low bucket.
    
    Compares a numeric personality rating against configured thresholds
    to determine if it's high, medium, or low. Used for applying
    personality-based adjustments to player evaluations.
    
    Args:
        rating: Numeric personality rating value
        thresholds: Dictionary with "high" and "medium" threshold values
    
    Returns:
        Category string: "high", "medium", or "low"
    """
    high_threshold = thresholds.get("high", 60.0)
    medium_threshold = thresholds.get("medium", 45.0)

    if rating >= high_threshold:
        return "high"
    if rating >= medium_threshold:
        return "medium"
    return "low"


def calculate_personality_adjustment(
    player: pd.Series,
    thresholds: Dict[str, float],
    modifiers: Dict[str, Dict[str, float]],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Calculate total personality adjustment based on player's personality traits.
    
    Evaluates all personality traits (Intelligence, Work Ethic, Greed, Loyalty, Leadership)
    and applies configured modifiers based on whether each trait is high, medium, or low.
    Supports both numeric ratings and string values (h/m/l, high/medium/low, etc.).
    
    Args:
        player: Series containing player data with personality trait columns
        thresholds: Dictionary with "high" and "medium" threshold values for categorization
        modifiers: Dictionary mapping trait names to {high/medium/low: modifier_value}
    
    Returns:
        Tuple containing:
        - Total personality adjustment (sum of all trait adjustments)
        - Dictionary mapping trait names to {rating, bucket, adjustment}
    """
    total_adjustment = 0.0
    details: Dict[str, Dict[str, float]] = {}

    for trait in PERSONALITY_TRAITS:
        if trait not in player:
            continue

        raw_value = player.get(trait)
        if pd.isna(raw_value):
            continue

        rating_display = raw_value
        bucket: str | None = None

        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"h", "high"}:
                bucket = "high"
            elif normalized in {"n", "m", "med", "medium", "avg", "average"}:
                bucket = "medium"
            elif normalized in {"l", "low"}:
                bucket = "low"
        else:
            try:
                rating = float(raw_value)
            except (TypeError, ValueError):
                continue
            bucket = categorize_personality_rating(rating, thresholds)
            rating_display = rating

        if bucket is None:
            continue

        trait_mapping = modifiers.get(trait)
        if trait_mapping is None:
            continue

        adjustment = trait_mapping.get(bucket, 0.0)
        details[trait] = {
            "rating": rating_display,
            "bucket": bucket,
            "adjustment": adjustment,
        }
        total_adjustment += adjustment

    return total_adjustment, details


def calculate_position_weighted_score(
    position: str,
    bat_total: float,
    defense_totals: Dict[str, float],
    baserunning_total: float,
    position_weights: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Calculate position-weighted combined score for a hitter.
    
    Combines bat, defense, and baserunning scores using position-specific weights.
    Different positions value these components differently (e.g., 1B values bat more,
    SS values defense more). Returns the weighted combination and breakdown.
    
    Args:
        position: Position name to apply weights for
        bat_total: Total weighted bat tool score
        defense_totals: Dictionary mapping positions to their defense scores
        baserunning_total: Total weighted baserunning score
        position_weights: Dictionary containing bat_tools, defense_tools, baserunning_tools weight categories
    
    Returns:
        Tuple containing:
        - Breakdown dictionary with category scores, weights, and weighted values
        - Combined total score (weighted bat + weighted defense + weighted baserunning)
    
    Raises:
        ValidationError: If position is missing, weights are missing, or defense score unavailable
    """
    if not position:
        raise ValidationError("Player position is missing or empty.")

    breakdown: Dict[str, Dict[str, float]] = {}
    combined_total = 0.0

    for category, category_value in [
        ("bat_tools", bat_total),
        ("defense_tools", defense_totals.get(position)),
        ("baserunning_tools", baserunning_total),
    ]:
        weights_for_category = position_weights.get(category)
        if weights_for_category is None:
            raise ValidationError(
                f"Position weights missing category '{category}'."
            )

        weight = weights_for_category.get(position)
        if weight is None:
            raise ValidationError(
                f"Position '{position}' missing weight for '{category}'."
            )

        if category == "defense_tools" and category_value is None:
            raise ValidationError(
                "Defense score unavailable for player's position.",
                [f"Position: {position}"],
            )

        category_value = float(category_value or 0.0)
        weighted_value = category_value * float(weight)

        breakdown[category] = {
            "score": category_value,
            "weight": float(weight),
            "weighted": weighted_value,
        }
        combined_total += weighted_value

    return breakdown, combined_total


def format_report(
    player: pd.Series,
    bat_details: Dict[str, Dict[str, float]],
    total_score: float,
    defense_scores: Dict[str, Dict[str, Dict[str, float]]],
    defense_totals: Dict[str, float],
    baserunning_scores: Dict[str, Dict[str, float]],
    baserunning_total: float,
    position_breakdown: Dict[str, Dict[str, float]],
    combined_total: float,
    age_details: Dict[str, object],
    age_adjust: float,
    archetype_label: str,
    archetype_details: Dict[str, object],
    org_label: str,
    team_label: str,
    league_label: str,
    personality_details: Dict[str, Dict[str, float]],
    personality_total: float,
    development_details: Dict[str, object],
    development_adjust: float,
    final_score: float,
) -> str:
    """
    Format a detailed text report for a hitter evaluation.
    
    Creates a comprehensive human-readable report showing all aspects of
    the player evaluation including bat tools, defense, baserunning, position
    weighting, personality adjustments, development status, age context,
    archetype classification, and final score.
    
    Args:
        player: Series containing player data
        bat_details: Dictionary of bat tool details
        total_score: Total weighted bat tool score
        defense_scores: Nested dictionary of defense scores by position and metric
        defense_totals: Dictionary mapping positions to total defense scores
        baserunning_scores: Dictionary of baserunning metric details
        baserunning_total: Total weighted baserunning score
        position_breakdown: Dictionary of position-weighted category breakdowns
        combined_total: Combined position-weighted score
        age_details: Dictionary of age adjustment details
        age_adjust: Age adjustment value
        archetype_label: Player's archetype classification
        archetype_details: Dictionary of archetype classification details
        org_label: Organization name string
        team_label: Team name string
        league_label: League level label string
        personality_details: Dictionary of personality trait adjustments
        personality_total: Total personality adjustment value
        development_details: Dictionary of development adjustment details
        development_adjust: Development adjustment value
        final_score: Final evaluated score after all adjustments
    
    Returns:
        Multi-line formatted report string
    """
    lines: List[str] = []

    player_name = player.get("Name", "Unknown Player")
    player_id = player.get("ID", "N/A")

    lines.append(f"Player: {player_name} (ID: {player_id})")
    lines.append(f"Organization: {org_label}")
    lines.append(f"Team: {team_label}")
    lines.append(f"League Level: {league_label}")
    if age_details.get("age") is not None:
        age_value = age_details.get("age")
        target = age_details.get("target")
        delta = age_details.get("delta")
        bucket = age_details.get("bucket")
        lines.append(
            "Age Context: "
            f"Age={age_value:.1f} "
            f"Target={target:.1f} "
            f"Delta={delta:+.1f} "
            f"Bucket={bucket} "
            f"Adj={age_adjust:+.2f}"
        )
    else:
        reason = age_details.get("reason", "unavailable")
        lines.append(f"Age Context: {reason}")
    lines.append(f"Archetype: {archetype_label}")
    archetype_summary = archetype_details.get("summary")
    if archetype_summary:
        lines.append(f"  Note: {archetype_summary}")
    lines.append("Bat Tool Validation:")
    for metric in BAT_TOOL_HEADERS:
        detail = bat_details.get(metric)
        if detail is None:
            continue
        potential_raw = detail["raw"]
        baseline_value = detail.get("baseline")
        modifier = detail.get("modifier", 1.0)
        lines.append(
            f"  {metric:<5} Pot={potential_raw:<5.1f} "
            f"Adj={detail['adjusted']:<5.1f} "
            f"Weight={detail['weight']:<4.2f} "
            f"Weighted={detail['weighted']:.2f}"
            f"{f' Baseline={baseline_value:.1f}' if baseline_value is not None else ''}"
            f"{f' Mod={modifier:.2f}' if baseline_value is not None else ''}"
        )
    lines.append(f"Total Weighted Bat Tool Score: {total_score:.2f}")

    if defense_scores:
        lines.append("")
        lines.append("Defense Validation:")
        primary_position = player.get("Pos", "N/A")
        lines.append(f"  Declared Position: {primary_position}")
        for position, metric_scores in defense_scores.items():
            lines.append(f"  Position: {position}")
            for metric_name, values in metric_scores.items():
                rating = values["rating"]
                adjusted = values["adjusted"]
                baseline_value = values.get("baseline")
                modifier = values.get("modifier", 1.0)
                weight = values["weight"]
                weighted_value = values["weighted"]
                lines.append(
                    "    "
                    f"{metric_name:<5} Rating={rating:<5.1f} "
                    f"Adj={adjusted:<5.1f} "
                    f"Weight={weight:<4.2f} Weighted={weighted_value:.2f}"
                    f"{f' Baseline={baseline_value:.1f}' if baseline_value is not None else ''}"
                    f"{f' Mod={modifier:.2f}' if baseline_value is not None else ''}"
                )
            lines.append(
                f"    Total Weighted {position}: {defense_totals[position]:.2f}"
            )

    if baserunning_scores:
        lines.append("")
        lines.append("Baserunning Validation:")
        for metric_name, values in baserunning_scores.items():
            rating = values["rating"]
            adjusted = values["adjusted"]
            baseline_value = values.get("baseline")
            modifier = values.get("modifier", 1.0)
            weight = values["weight"]
            weighted_value = values["weighted"]
            source_column = values.get("column", metric_name)
            lines.append(
                f"  {metric_name:<8} (src: {source_column}) "
                f"Rating={rating:<5.1f} Adj={adjusted:<5.1f} "
                f"Weight={weight:<4.2f} "
                f"Weighted={weighted_value:.2f}"
                f"{f' Baseline={baseline_value:.1f}' if baseline_value is not None else ''}"
                f"{f' Mod={modifier:.2f}' if baseline_value is not None else ''}"
            )
        lines.append(f"  Total Baserunning Score: {baserunning_total:.2f}")

    if position_breakdown:
        lines.append("")
        lines.append("Positional Weighting Summary:")
        for category, values in position_breakdown.items():
            lines.append(
                f"  {category:<16} Score={values['score']:<6.2f} "
                f"Weight={values['weight']:<4.2f} "
                f"Weighted={values['weighted']:.2f}"
            )
        lines.append(f"  Combined Player Score: {combined_total:.2f}")

    if personality_details:
        lines.append("")
        lines.append("Personality Adjustment:")
        for trait, values in personality_details.items():
            rating_display = str(values["rating"])
            lines.append(
                f"  {trait:<9} Rating={rating_display:<5} "
                f"Bucket={values['bucket']:<6} "
                f"Adj={values['adjustment']:+.2f}"
            )
        lines.append(f"  Total Personality Adjustment: {personality_total:+.2f}")

    lines.append("")
    lines.append("Development Adjustment:")
    avg_current = development_details.get("avg_current")
    avg_potential = development_details.get("avg_potential")
    ratio = development_details.get("ratio")
    lines.append(
        "  Avg Current: "
        f"{avg_current:.2f}" if isinstance(avg_current, (int, float)) else "  Avg Current: N/A"
    )
    lines.append(
        "  Avg Potential: "
        f"{avg_potential:.2f}" if isinstance(avg_potential, (int, float)) else "  Avg Potential: N/A"
    )
    lines.append(
        "  Ratio: "
        f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "  Ratio: N/A"
    )
    lines.append(
        f"  Bucket Initial: {development_details.get('bucket_initial', 'neutral')} "
        f"=> Applied: {development_details.get('bucket_applied', 'neutral')}"
    )
    reason = development_details.get("reason")
    if isinstance(reason, str) and reason:
        lines.append(f"  Note: {reason}")
    lines.append(f"  Modifier: {development_adjust:+.2f}")

    lines.append("")
    lines.append(f"Final Evaluated Score: {final_score:.2f}")

    return "\n".join(lines)


def evaluate_hitter(
    player: pd.Series,
    bat_weights: Dict[str, float],
    defense_weights: Dict[str, Dict[str, float]],
    baserunning_weights: Dict[str, float],
    position_weights: Dict[str, Dict[str, float]],
    personality_thresholds: Dict[str, float],
    personality_modifiers: Dict[str, Dict[str, float]],
    age_config: Dict[str, Any],
    archetype_config: Dict[str, Any],
    defense_baselines: Dict[str, Dict[str, float]],
    bat_baselines: Dict[str, Dict[str, float]],
    baserunning_baselines: Dict[str, Dict[str, float]],
    baserunning_columns: Dict[str, str],
    development_config: Dict[str, Dict[str, object]],
    team_lookup: Dict[int, Dict[str, object]],
    league_lookup: Dict[int, str],
) -> Tuple[str, Dict[str, object]]:
    """
    Complete evaluation of a hitter's overall value and profile.
    
    Performs a comprehensive evaluation of a hitter by:
    1. Calculating bat tool scores
    2. Calculating defense scores for all positions
    3. Calculating baserunning scores
    4. Evaluating the player at all eligible positions with position-specific baselines
    5. Applying personality, development, and age adjustments
    6. Classifying the player's archetype at each position
    7. Determining best position and current position values
    8. Generating a detailed report and summary
    
    Args:
        player: Series containing all hitter data
        bat_weights: Dictionary mapping bat tool names to weights
        defense_weights: Dictionary mapping positions to their metric weights
        baserunning_weights: Dictionary mapping baserunning metrics to weights
        position_weights: Dictionary containing position weight categories
        personality_thresholds: Dictionary with high/medium thresholds
        personality_modifiers: Dictionary mapping traits to bucket modifiers
        age_config: Age adjustment configuration
        archetype_config: Archetype classification configuration
        defense_baselines: Dictionary mapping positions to defense baselines
        bat_baselines: Dictionary mapping positions to bat baselines
        baserunning_baselines: Dictionary mapping positions to baserunning baselines
        baserunning_columns: Dictionary mapping metric names to column names
        development_config: Development adjustment configuration
        team_lookup: Dictionary mapping team IDs to team information
        league_lookup: Dictionary mapping league level IDs to labels
    
    Returns:
        Tuple containing:
        - Detailed formatted report string
        - Summary dictionary with key evaluation metrics for CSV export
    """
    _, _, bat_details = calculate_bat_tools(player, bat_weights)
    defense_scores, defense_totals = calculate_defense_tools(
        player, defense_weights, defense_baselines
    )
    baserunning_scores, baserunning_total = calculate_baserunning_tools(
        player, baserunning_weights, baserunning_columns
    )
    personality_total, personality_details = calculate_personality_adjustment(
        player, personality_thresholds, personality_modifiers
    )
    development_adjust, development_details = calculate_development_adjustment(
        player,
        development_config.get("hitter", {}),
        HITTER_CURRENT_METRICS,
        HITTER_POTENTIAL_METRICS,
    )
    team_label = resolve_team_label(player.get("Team"), team_lookup)
    org_label = resolve_team_label(player.get("Org"), team_lookup)
    league_label = resolve_league_label(player.get("LgLvl"), league_lookup)
    age_adjust, age_details = calculate_age_adjustment(
        player.get("Age"), league_label, age_config
    )
    available_positions = sorted(
        {
            pos
            for mapping in position_weights.values()
            for pos in mapping.keys()
        }
    )

    position_results: List[Dict[str, object]] = []
    for pos in available_positions:
        if pos not in defense_totals:
            continue

        # Hard cutoff for SS: skip if IFR is below SS baseline
        if pos == "SS":
            ss_baseline = defense_baselines.get("SS", {}).get("IFR")
            if ss_baseline is not None:
                player_ifr = normalize_float(player.get("IFR"))
                if player_ifr is not None and player_ifr < ss_baseline:
                    continue

        # Hard cutoff for C: skip if CArm is below 40
        if pos == "C":
            player_carm = normalize_float(player.get("CArm"))
            if player_carm is not None and player_carm < 40.0:
                continue

        bat_total_pos, bat_details_pos = apply_bat_baselines(
            bat_details, bat_baselines, pos
        )
        baserunning_total_pos, baserunning_details_pos = apply_baserunning_baselines(
            baserunning_scores, baserunning_baselines, pos
        )

        try:
            breakdown, combined_total = calculate_position_weighted_score(
                pos,
                bat_total_pos,
                defense_totals,
                baserunning_total_pos,
                position_weights,
            )
        except ValidationError:
            continue

        final_score = (
            combined_total + personality_total + development_adjust + age_adjust
        )
        archetype_label, archetype_details = classify_hitter_archetype(
            pos,
            bat_total_pos,
            defense_totals.get(pos, 0.0),
            baserunning_total_pos,
            bat_details_pos,
            baserunning_details_pos,
            archetype_config,
        )
        position_results.append(
            {
                "position": pos,
                "bat_total": bat_total_pos,
                "bat_details": bat_details_pos,
                "defense_details": defense_scores.get(pos, {}),
                "defense_total": defense_totals.get(pos, 0.0),
                "baserunning_total": baserunning_total_pos,
                "baserunning_details": baserunning_details_pos,
                "breakdown": breakdown,
                "combined_total": combined_total,
                "final_score": final_score,
                "archetype_label": archetype_label,
                "archetype_details": archetype_details,
            }
        )

    if not position_results:
        raise ValidationError(
            "Unable to evaluate any positions for the player.",
            [f"Player ID: {player.get('ID')}"],
        )

    declared_position_raw = str(player.get("Pos", "")).strip()
    declared_position = declared_position_raw.upper()

    declared_result = next(
        (res for res in position_results if res["position"] == declared_position),
        None,
    )
    best_result = max(position_results, key=lambda item: item["final_score"])  # type: ignore[arg-type]
    current_result = declared_result or best_result

    current_position = current_result["position"]  # type: ignore[index]
    current_defense_details = current_result["defense_details"]  # type: ignore[index]
    current_defense_total = current_result["defense_total"]  # type: ignore[index]
    current_bat_details = current_result["bat_details"]  # type: ignore[index]
    current_bat_total = current_result["bat_total"]  # type: ignore[index]
    current_baserunning_details = current_result["baserunning_details"]  # type: ignore[index]
    current_baserunning_total = current_result["baserunning_total"]  # type: ignore[index]
    current_archetype_label = current_result["archetype_label"]  # type: ignore[index]
    current_archetype_details = current_result["archetype_details"]  # type: ignore[index]

    report = format_report(
        player,
        current_bat_details,
        current_bat_total,
        {current_position: current_defense_details},
        {current_position: current_defense_total},
        current_baserunning_details,
        current_baserunning_total,
        current_result["breakdown"],  # type: ignore[arg-type]
        current_result["combined_total"],  # type: ignore[arg-type]
        age_details,
        age_adjust,
        current_archetype_label,
        current_archetype_details,
        org_label,
        team_label,
        league_label,
        personality_details,
        personality_total,
        development_details,
        development_adjust,
        current_result["final_score"],  # type: ignore[arg-type]
    )

    alternate_candidates = [
        res for res in position_results if res["position"] != current_position
    ]
    if alternate_candidates:
        best_alternate = max(
            alternate_candidates, key=lambda item: item["final_score"]
        )
        improvement = best_alternate["final_score"] - current_result["final_score"]  # type: ignore[index]
        label = (
            "Best Alternate Position:"
            if improvement > 0.0
            else "Alternate Position (highest non-current):"
        )
        report = (
            f"{report}\n{label}\n"
            f"  Position: {best_alternate['position']} "
            f"Final Score: {best_alternate['final_score']:.2f} "
            f"(delta {improvement:+.2f})"
        )
    else:
        report = f"{report}\nNo alternate positions available for evaluation."

    declared_value = (
        declared_result["final_score"] if declared_result is not None else None
    )
    best_archetype_label = best_result["archetype_label"]  # type: ignore[index]
    best_archetype_details = best_result["archetype_details"]  # type: ignore[index]

    # Create a dictionary mapping positions to their final scores
    position_scores: Dict[str, Optional[float]] = {}
    all_hitter_positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    for pos in all_hitter_positions:
        position_scores[f"{pos} Score"] = None
    
    for result in position_results:
        pos = result["position"]  # type: ignore[index]
        final_score = result["final_score"]  # type: ignore[index]
        position_scores[f"{pos} Score"] = round_optional(final_score)
    
    summary = {
        "ID": player.get("ID"),
        "Pos": declared_position_raw,
        "Name": player.get("Name"),
        "B": player.get("Bats"),
        "T": player.get("Throws"),
        "Current Value": round_optional(
            declared_value if declared_value is not None else current_result["final_score"]  # type: ignore[index]
        ),
        "Ideal Pos": best_result["position"],
        "Ideal Value": round_optional(best_result["final_score"]),
        "Bat Score": round_optional(best_result["bat_total"]),
        "Defense Score": round_optional(best_result["defense_total"]),
        "Baserunning Score": round_optional(best_result["baserunning_total"]),
        "Pitching Ability Score": None,
        "Pitching Arsenal Score": None,
        "Development Adjustment": round_optional(development_adjust),
        "Age Adjustment": round_optional(age_adjust),
        "Archetype": current_archetype_label,
        "Archetype Detail": current_archetype_details.get("summary", ""),
        "Ideal Archetype": best_archetype_label,
        "Ideal Archetype Detail": best_archetype_details.get("summary", ""),
        "Organization": org_label,
        "Team": team_label,
        "League Level": league_label,
        "Role Multiplier": None,
        "Role Adjustment": None,
    }
    # Add position score columns
    summary.update(position_scores)

    return report, summary


def format_pitch_report(
    player: pd.Series,
    role: str,
    ability_details: Dict[str, Dict[str, float]],
    ability_total: float,
    arsenal_entries: List[Dict[str, object]],
    arsenal_slot_total: float,
    diversity_summary: Dict[str, object],
    diversity_total: float,
    arsenal_total: float,
    combination_breakdown: Dict[str, Dict[str, float]],
    combined_total: float,
    org_label: str,
    team_label: str,
    league_label: str,
    age_details: Dict[str, object],
    age_adjust: float,
    archetype_label: str,
    archetype_details: Dict[str, object],
    development_details: Dict[str, object],
    development_adjust: float,
    role_multiplier: float,
    role_adjustment: float,
    final_total: float,
) -> str:
    """
    Format a detailed text report for a pitcher evaluation.
    
    Creates a comprehensive human-readable report showing all aspects of
    the pitcher evaluation including pitch ability metrics, pitch arsenal
    details, diversity adjustments, ability/arsenal combination, development
    status, age context, archetype classification, role adjustments, and final score.
    
    Args:
        player: Series containing player data
        role: Pitcher role ("SP" or "RP")
        ability_details: Dictionary of pitch ability metric details
        ability_total: Total weighted pitch ability score
        arsenal_entries: List of pitch entry dictionaries with all pitch details
        arsenal_slot_total: Total slot-weighted arsenal contribution
        diversity_summary: Dictionary of arsenal diversity information
        diversity_total: Total diversity adjustment (bonuses/penalties)
        arsenal_total: Final arsenal score (slot + diversity)
        combination_breakdown: Dictionary of ability/arsenal combination breakdown
        combined_total: Combined ability + arsenal score
        org_label: Organization name string
        team_label: Team name string
        league_label: League level label string
        age_details: Dictionary of age adjustment details
        age_adjust: Age adjustment value
        archetype_label: Pitcher's archetype classification
        archetype_details: Dictionary of archetype classification details
        development_details: Dictionary of development adjustment details
        development_adjust: Development adjustment value
        role_multiplier: Role-specific multiplier applied
        role_adjustment: Total role adjustment impact
        final_total: Final evaluated score after all adjustments
    
    Returns:
        Multi-line formatted report string
    """
    lines: List[str] = []
    player_name = player.get("Name", "Unknown Pitcher")
    player_id = player.get("ID", "N/A")

    lines.append(f"Pitcher: {player_name} (ID: {player_id})")
    lines.append(f"Role: {role}")
    lines.append(f"Organization: {org_label}")
    lines.append(f"Team: {team_label}")
    lines.append(f"League Level: {league_label}")
    if age_details.get("age") is not None:
        age_value = age_details.get("age")
        target = age_details.get("target")
        delta = age_details.get("delta")
        bucket = age_details.get("bucket")
        lines.append(
            "Age Context: "
            f"Age={age_value:.1f} "
            f"Target={target:.1f} "
            f"Delta={delta:+.1f} "
            f"Bucket={bucket} "
            f"Adj={age_adjust:+.2f}"
        )
    else:
        reason = age_details.get("reason", "unavailable")
        lines.append(f"Age Context: {reason}")
    lines.append(f"Archetype: {archetype_label}")
    archetype_summary = archetype_details.get("summary")
    if archetype_summary:
        lines.append(f"  Note: {archetype_summary}")
    lines.append("Pitch Ability Validation:")

    for column in PITCH_ABILITY_METRICS:
        detail = ability_details.get(column)
        if detail is None:
            continue
        display_name = detail.get("display", column)
        rating = detail["rating"]
        adjusted = detail["adjusted"]
        weight = detail["weight"]
        weighted = detail["weighted"]
        lines.append(
            f"  {display_name:<15} (src: {column}) "
            f"Pot={rating:<5.1f} Adj={adjusted:<5.1f} "
            f"Weight={weight:<4.2f} Weighted={weighted:.2f}"
        )

    lines.append(f"Total Weighted Pitch Ability Score: {ability_total:.2f}")

    if arsenal_entries:
        lines.append("")
        lines.append("Pitch Arsenal Validation:")
        for entry in arsenal_entries:
            pitch_type = entry["pitch_type"]
            column = entry["column"]
            rating = entry["rating"]
            adjusted = entry["adjusted"]
            type_weight = entry["type_weight"]
            base_value = entry["base_value"]
            slot = entry.get("slot") or "-"
            slot_weight = entry.get("slot_weight", 0.0)
            slot_contribution = entry.get("slot_contribution", 0.0)
            speed_tier = entry.get("speed_tier") or "-"
            break_plane = entry.get("break_plane") or "-"
            lines.append(
                f"  {pitch_type:<7} (src: {column}) "
                f"Pot={rating:<5.1f} Adj={adjusted:<5.1f} "
                f"TypeWt={type_weight:<4.2f} Base={base_value:<6.2f} "
                f"Slot={slot:<6} SlotWt={slot_weight:<4.2f} "
                f"Weighted={slot_contribution:<6.2f} "
                f"[Speed={speed_tier} Break={break_plane}]"
            )
        lines.append(
            f"Total Slot-Weighted Arsenal Score: {arsenal_slot_total:.2f}"
        )

    if diversity_summary:
        lines.append("")
        lines.append("Arsenal Diversity Adjustments:")
        baselines = diversity_summary.get("baselines", {})
        lines.append(
            "  Pitch Count: "
            f"{diversity_summary.get('pitch_count', 0)} "
            f"(min {baselines.get('min_pitches')}, target {baselines.get('target_pitches')})"
        )
        speed_tiers = diversity_summary.get("speed_tiers") or []
        break_planes = diversity_summary.get("break_planes") or []
        lines.append(
            "  Speed Tiers Present: "
            f"{', '.join(speed_tiers) if speed_tiers else 'None'}"
        )
        lines.append(
            "  Break Planes Present: "
            f"{', '.join(break_planes) if break_planes else 'None'}"
        )
        for adjustment in diversity_summary.get("adjustments", []):
            value = adjustment["value"]
            if value > 0:
                label = "Bonus"
            elif value < 0:
                label = "Penalty"
            else:
                label = "Adj"
            lines.append(
                f"  {label:<6} {adjustment['category']:<13} "
                f"Value={value:+.2f} "
                f"(met: {'yes' if adjustment['met'] else 'no'}, "
                f"count={adjustment.get('count')}, "
                f"required={adjustment.get('required')})"
            )
        lines.append(f"  Diversity Adjustment Total: {diversity_total:+.2f}")

    lines.append(f"Final Arsenal Score: {arsenal_total:.2f}")

    lines.append("")
    lines.append("Pitch Weighting Summary:")
    label_map = {
        "pitch_ability": "Ability",
        "pitch_arsenal": "Arsenal",
    }
    for key in ["pitch_ability", "pitch_arsenal"]:
        values = combination_breakdown.get(key)
        if not values:
            continue
        label = label_map.get(key, key)
        lines.append(
            f"  {label:<10} Score={values['score']:<6.2f} "
            f"Weight={values['weight']:<4.2f} "
            f"Weighted={values['weighted']:.2f}"
        )
    lines.append(f"Combined Pitch Score: {combined_total:.2f}")

    lines.append("")
    lines.append("Development Adjustment:")
    avg_current = development_details.get("avg_current")
    avg_potential = development_details.get("avg_potential")
    ratio = development_details.get("ratio")
    lines.append(
        "  Avg Current: "
        f"{avg_current:.2f}" if isinstance(avg_current, (int, float)) else "  Avg Current: N/A"
    )
    lines.append(
        "  Avg Potential: "
        f"{avg_potential:.2f}" if isinstance(avg_potential, (int, float)) else "  Avg Potential: N/A"
    )
    lines.append(
        "  Ratio: "
        f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "  Ratio: N/A"
    )
    lines.append(
        f"  Bucket Initial: {development_details.get('bucket_initial', 'neutral')} "
        f"=> Applied: {development_details.get('bucket_applied', 'neutral')}"
    )
    reason = development_details.get("reason")
    if isinstance(reason, str) and reason:
        lines.append(f"  Note: {reason}")
    lines.append(f"  Modifier: {development_adjust:+.2f}")

    lines.append("")
    lines.append("Role Adjustment:")
    lines.append(f"  Role Multiplier: {role_multiplier:.2f}")
    lines.append(f"  Role Impact: {role_adjustment:+.2f}")

    lines.append("")
    lines.append(f"Final Evaluated Pitch Score: {final_total:.2f}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the player evaluation script.
    
    Sets up argument parser with options for league selection, ID filtering, 
    output file path, and whether to print the first detailed report.
    
    Returns:
        Namespace object containing parsed command-line arguments:
        - league: Required league identifier (e.g., 'woba', 'sahl')
        - ids_file: Optional path to file containing player IDs to filter
        - output: Path for summary CSV output (default: evaluation_summary_[league].csv)
        - print_first: Boolean flag to print first detailed report
    """
    parser = argparse.ArgumentParser(
        description="Evaluate baseball players using configured scouting weights."
    )
    parser.add_argument(
        "--league",
        dest="league",
        required=True,
        help="League identifier (e.g., 'woba', 'sahl'). Required to locate PlayerData-[league].csv file.",
    )
    parser.add_argument(
        "--ids-file",
        dest="ids_file",
        help="Optional path to a file containing player IDs to evaluate "
        "(comma, tab, or newline separated). If omitted, all players are processed.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        help="Path for the summary CSV file (default: evaluation_summary_[league].csv).",
    )
    parser.add_argument(
        "--print-first",
        dest="print_first",
        action="store_true",
        help="Print the detailed report for the first successfully evaluated player.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the player evaluation script.
    
    Orchestrates the complete evaluation process:
    1. Parses command-line arguments
    2. Loads player data and validates required columns
    3. Optionally filters players by ID list
    4. Loads all weight configurations and reference data
    5. Validates dataset has all required columns for hitters/pitchers
    6. Iterates through all players, evaluating each one
    7. Handles errors gracefully and tracks progress
    8. Outputs summary CSV with all evaluation results
    9. Optionally prints detailed report for first player
    
    The script processes both hitters and pitchers, applying different
    evaluation logic based on position. All results are compiled into
    a summary CSV file for further analysis.
    """
    args = parse_args()
    print("[INFO] Starting player evaluation...", flush=True)

    # Construct paths based on league argument
    data_path = Path(f"../PlayerData-{args.league}.csv")
    teams_path = Path(f"../teams-{args.league}.json")
    
    # Set default output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"../evaluation_summary_{args.league}.csv")

    try:
        print(f"[INFO] Loading player data from {data_path}...", flush=True)
        data = load_data(data_path)
        print(f"[INFO] Loaded {len(data)} players from CSV.", flush=True)
        validate_columns(data, BAT_TOOL_COLUMNS)
        print("[INFO] Column validation passed.", flush=True)

        id_filters: Optional[Set[str]] = None
        if args.ids_file:
            id_filters = load_id_filter(Path(args.ids_file))
            if id_filters:
                data = data[
                    data["ID"].astype(str).isin(id_filters)  # type: ignore[arg-type]
                ]
            else:
                print(
                    "[WARN] ID filter file was empty; no players will be processed."
                )
                return

        if data.empty:
            print("[WARN] No players available for evaluation.")
            return

        print(f"[INFO] Loading weights from {WEIGHTS_PATH}...", flush=True)
        (
            pitch_ability_weights,
            pitch_type_weights,
            pitch_role_slot_weights,
            pitch_role_baselines,
            pitch_diversity_modifiers,
            development_config,
            pitch_role_adjustments,
            stamina_thresholds,
            bat_weights,
            defense_weights,
            baserunning_weights,
            position_weights,
            personality_thresholds,
            personality_modifiers,
            age_level_config,
            archetype_config,
            defense_baselines,
            bat_baselines,
            baserunning_baselines,
        ) = load_weights(
            WEIGHTS_PATH,
            BAT_TOOL_HEADERS,
            BASERUNNING_HEADERS,
            PITCHER_ROLES,
            PITCH_ABILITY_METRICS,
        )
        print("[INFO] Weights loaded successfully.", flush=True)
        print(f"[INFO] Loading reference data from {ID_MAPS_PATH} and {teams_path}...", flush=True)
        league_lookup, team_lookup = load_reference_data(ID_MAPS_PATH, teams_path)
        print("[INFO] Reference data loaded successfully.", flush=True)

        positions_upper = data["Pos"].astype(str).str.upper()
        has_pitchers = positions_upper.isin(PITCHER_ROLES).any()
        has_hitters = (~positions_upper.isin(PITCHER_ROLES)).any()

        baserunning_columns: Dict[str, str] = {}
        if has_hitters:
            validate_defense_columns(data, defense_weights)
            baserunning_columns = validate_baserunning_columns(
                data, BASERUNNING_HEADERS
            )
        if has_pitchers:
            validate_pitcher_columns(
                data, PITCH_ABILITY_METRICS, PITCH_ARSENAL_COLUMNS
            )

        total_players = len(data)
        print(f"[INFO] Starting evaluation of {total_players} players...", flush=True)
        summary_rows: List[Dict[str, object]] = []
        first_report_printed = False
        error_count = 0

        for processed, (_, player) in enumerate(data.iterrows(), start=1):
            role = str(player.get("Pos", "")).strip().upper()
            # Convert CL to RP (closer is a sub-category of relief pitcher)
            if role == "CL":
                role = "RP"
            try:
                if role in PITCHER_ROLES:
                    report, summary = evaluate_pitcher(
                        player,
                        role,
                        pitch_ability_weights,
                        pitch_type_weights,
                        pitch_role_slot_weights,
                        pitch_role_baselines,
                        pitch_diversity_modifiers,
                        position_weights,
                        development_config,
                        age_level_config.get("pitcher", {}),
                        archetype_config.get("pitcher", {}),
                        pitch_role_adjustments,
                        stamina_thresholds,
                        team_lookup,
                        league_lookup,
                    )
                else:
                    if not baserunning_columns:
                        raise ValidationError(
                            "Baserunning columns unavailable for hitter evaluation."
                        )
                    report, summary = evaluate_hitter(
                        player,
                        bat_weights,
                        defense_weights,
                        baserunning_weights,
                        position_weights,
                        personality_thresholds,
                        personality_modifiers,
                        age_level_config.get("hitter", {}),
                        archetype_config.get("hitter", {}),
                        defense_baselines,
                        bat_baselines,
                        baserunning_baselines,
                        baserunning_columns,
                        development_config,
                        team_lookup,
                        league_lookup,
                    )
                summary_rows.append(summary)
                if args.print_first and not first_report_printed:
                    print(report)
                    first_report_printed = True
            except ValidationError as error:
                player_id = player.get("ID")
                print(f"[ERROR] Player ID {player_id}: {error}")
                if error.details:
                    for detail in error.details:
                        print(f"  - {detail}")
                error_count += 1
            except Exception as exc:  # pragma: no cover - safeguard
                player_id = player.get("ID")
                print(f"[ERROR] Player ID {player_id}: {exc}")
                error_count += 1
            finally:
                print_progress(processed, total_players)

        if not summary_rows:
            print("[WARN] All player evaluations failed.")
            return

        pd.DataFrame(summary_rows).to_csv(output_path, index=False)
        print(f"Saved evaluation summary to {output_path.resolve()}")

        if error_count:
            print(
                f"Completed with {error_count} player(s) reporting validation issues."
            )
    except (FileNotFoundError, ValidationError) as error:
        print(f"[ERROR] {error}")
        if isinstance(error, ValidationError) and error.details:
            for detail in error.details:
                print(f"  - {detail}")


if __name__ == "__main__":
    main()
