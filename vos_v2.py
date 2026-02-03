#!/usr/bin/env python3
"""
VOS v2 (VOS Optimized Score) — Baseball player evaluation using a weighted scoring system.

Calculates normalized 20–80 scores for hitters and pitchers from PlayerData CSV and
config (weights_v2.json, id_maps, teams). Outputs evaluation_summary_{league}_{timestamp}.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data")
DEFAULT_CONFIG_DIR = Path("config")
WEIGHTS_FILENAME = "weights_v2.json"
ID_MAPS_FILENAME = "id_maps.json"
TEAMS_FILENAME_TEMPLATE = "teams-{league}.json"
PLAYER_DATA_FILENAME_TEMPLATE = "PlayerData-{league}.csv"

# CSV column alternatives (first present wins)
BASERUNNING_STEAL_COLS = ["StealAbi", "Steal"]
PITCHER_ABILITY_CSV_TO_CONFIG = {
    "Stf": "Stuff",
    "Mov": "Movement",
    "Ctrl": "Control",   # CSV may have Ctrl_R, Ctrl_L only
    "HRA": "HR_Avoid",
}
# Control: CSV often has Ctrl_R/Ctrl_L only, no "Ctrl"
PITCHER_ABILITY_COL_ALTERNATIVES: Dict[str, List[str]] = {
    "Control": ["Ctrl", "Ctrl_R", "Ctrl_L"],
}
# Current → potential column names for potential VOS (batting and pitcher ability only; defense/baserunning have no Pot* in CSV)
HITTER_BATTING_CURRENT_TO_POTENTIAL = {"Gap": "PotGap", "Pow": "PotPow", "Eye": "PotEye", "Ks": "PotKs"}
PITCHER_ABILITY_CURRENT_TO_POTENTIAL = {"Stf": "PotStf", "Mov": "PotMov", "HRA": "PotHRA", "Ctrl": "PotCtrl"}
POT_PITCH_COLUMN_TO_TYPE = {
    "PotFst": "Fastball",
    "PotSnk": "Sinker",
    "PotCutt": "Cutter",
    "PotCrv": "Curve",
    "PotSld": "Slider",
    "PotChg": "Changeup",
    "PotSplt": "Splitter",
    "PotFrk": "Forkball",
    "PotCirChg": "Circle_Change",
    "PotScr": "Screwball",
    "PotKncrv": "Knuckle_Curve",
    "PotKnbl": "Knuckleball",
}
PITCH_SPEED_TIERS = {
    "Fastball": "hard", "Sinker": "hard", "Cutter": "hard",
    "Slider": "breaker", "Curve": "breaker", "Knuckle_Curve": "breaker", "Knuckleball": "breaker",
    "Changeup": "offspeed", "Circle_Change": "offspeed", "Splitter": "offspeed",
    "Forkball": "offspeed", "Screwball": "offspeed",
}
PITCH_BREAK_PLANES = {
    "Fastball": "vertical", "Sinker": "vertical", "Cutter": "horizontal",
    "Slider": "horizontal", "Curve": "vertical", "Knuckle_Curve": "vertical",
    "Knuckleball": "horizontal", "Changeup": "vertical", "Circle_Change": "vertical",
    "Splitter": "vertical", "Forkball": "vertical", "Screwball": "horizontal",
}
PERSONALITY_CSV_TO_CONFIG = {
    "Int": "Intelligence",
    "WrkEthic": "Work_Ethic",
    "Greed": "Greed",
    "Loy": "Loyalty",
    "Lead": "Leadership",
}

HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
LEVEL_LABEL_TO_CONFIG = {"R": "Rookie"}  # id_maps uses "R", config uses "Rookie"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file; return empty dict on missing/invalid."""
    if not path.exists():
        logger.warning("Config not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_weights(config_dir: Path) -> Dict[str, Any]:
    """Load weights_v2.json."""
    return load_json(config_dir / WEIGHTS_FILENAME)


def load_id_maps(config_dir: Path) -> Dict[int, str]:
    """Build league level id -> label (e.g. 1 -> 'ML')."""
    raw = load_json(config_dir / ID_MAPS_FILENAME)
    level_map = raw.get("league_level") or raw.get("league_levels")
    if not isinstance(level_map, dict):
        return {}
    lookup: Dict[int, str] = {}
    for label, value in level_map.items():
        if label.startswith("_"):
            continue
        try:
            key = int(value)
            lookup[key] = str(label)
        except (TypeError, ValueError):
            continue
    return lookup


def load_teams(config_dir: Path, league: str) -> Dict[int, str]:
    """Build team id -> display name (e.g. 'Arizona Diamondbacks')."""
    path = config_dir / TEAMS_FILENAME_TEMPLATE.format(league=league)
    raw = load_json(path)
    if not isinstance(raw, dict):
        return {}
    result: Dict[int, str] = {}
    for tid_str, info in raw.items():
        if tid_str.startswith("_") or not isinstance(info, dict):
            continue
        try:
            tid = int(tid_str)
        except (TypeError, ValueError):
            continue
        name = info.get("Name") or ""
        nick = info.get("Nickname") or ""
        result[tid] = f"{name} {nick}".strip() or f"Team {tid}"
    return result


def resolve_float(row: Dict[str, str], *col_candidates: str) -> Optional[float]:
    """First non-empty numeric value from row for given column names."""
    for col in col_candidates:
        if col not in row:
            continue
        val = row.get(col, "").strip()
        if val == "" or val.upper() in ("NA", "N/A", "."):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def resolve_int(row: Dict[str, str], col: str) -> Optional[int]:
    """Integer value for column; None if missing or invalid."""
    v = resolve_float(row, col)
    return int(v) if v is not None else None


def load_player_data(data_dir: Path, league: str, id_filter: Optional[Set[str]] = None) -> List[Dict[str, str]]:
    """Load PlayerData-{league}.csv; optionally filter by ID set. Skip rows that fail basic validation."""
    path = data_dir / PLAYER_DATA_FILENAME_TEMPLATE.format(league=league)
    if not path.exists():
        logger.error("Player data not found: %s", path)
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            logger.error("CSV missing ID column")
            return []
        for row in reader:
            pid = (row.get("ID") or "").strip()
            if not pid:
                continue
            if id_filter is not None and pid not in id_filter:
                continue
            rows.append(row)
    logger.info("Loaded %d players from %s", len(rows), path.name)
    return rows


def load_id_filter(file_path: Optional[Path]) -> Optional[Set[str]]:
    """Load set of player IDs from file (one per line or comma/semicolon/tab separated)."""
    if file_path is None or not file_path.exists():
        return None
    ids: Set[str] = set()
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            for sep in (",", ";", "\t", " "):
                line = line.replace(sep, " ")
            for token in line.split():
                t = token.strip()
                if t:
                    ids.add(t)
    return ids if ids else None


# -----------------------------------------------------------------------------
# Park factors (optional)
# -----------------------------------------------------------------------------

def load_park_factors(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load park factor adjustments from JSON file.

    Args:
        path: Path to park-factors.json file (can be None).

    Returns:
        Dictionary with parks, team_to_park_mapping, application_rules; or None if not provided/not found/invalid.
    """
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning("Park factors file not found: %s", path)
        return None
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded park factors from %s", path)
        return data
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in park factors file: %s", e)
        return None


def _is_single_park_format(park_factors: Dict[str, Any]) -> bool:
    """True if file is single-park (LVK) format: tool_adjustments at root, no team lookup."""
    return isinstance(park_factors.get("tool_adjustments"), dict)


def _build_single_park_config(park_factors: Dict[str, Any]) -> Dict[str, Any]:
    """Build park config from single-park file (e.g. park-factors-lvk.json)."""
    tool_adjustments = park_factors.get("tool_adjustments") or {}
    team_info = park_factors.get("team_info") or {}
    name = team_info.get("park_name") if isinstance(team_info, dict) else None
    if not name or not isinstance(name, str):
        name = "Park"
    handedness_raw = park_factors.get("handedness_splits") or {}
    handedness_splits = {}
    if isinstance(handedness_raw, dict):
        for k in ("RHB", "LHB"):
            if k in handedness_raw and isinstance(handedness_raw[k], dict):
                handedness_splits[k] = handedness_raw[k]
    return {
        "name": name,
        "tool_adjustments": tool_adjustments,
        "handedness_splits": handedness_splits,
    }


def get_player_park_config(
    row: Dict[str, str],
    park_factors: Optional[Dict[str, Any]],
    teams: Dict[int, str],
    league_lookup: Dict[int, str],
) -> Optional[Dict[str, Any]]:
    """
    Determine which park configuration applies to this player.

    Two formats supported:
    - Single-park (e.g. park-factors-lvk.json): tool_adjustments at root; same park applied to
      all players (subject to application_rules). No team lookup.
    - Multi-park: parks[key] and team_to_park_mapping; park chosen by player's team.

    Args:
        row: Player row (CSV dict).
        park_factors: Loaded park factors data (can be None).
        teams: Team ID (int) -> display name.
        league_lookup: League level ID -> label (e.g. 1 -> 'ML').

    Returns:
        Park configuration dict (tool_adjustments, handedness_splits, name, etc.) or None.
    """
    if not park_factors:
        return None
    rules = park_factors.get("application_rules", {})
    if not isinstance(rules, dict):
        rules = {}
    lg_lvl = resolve_int(row, "LgLvl")
    league_label = get_league_label(lg_lvl, league_lookup) if lg_lvl is not None else ""
    # Don't apply to prospects if rule says so (non-ML = prospect)
    if not rules.get("apply_to_prospects", False) and league_label != "ML":
        return None
    # Don't apply to major leaguers if rule says so
    if not rules.get("apply_to_major_leaguers", True) and league_label == "ML":
        return None

    if _is_single_park_format(park_factors):
        return _build_single_park_config(park_factors)

    team_id_raw = row.get("Team", "").strip()
    team_id_int = resolve_int(row, "Team")
    team_name = get_team_display(team_id_int, teams) if team_id_int is not None else ""
    team_to_park = park_factors.get("team_to_park_mapping", {})
    if not isinstance(team_to_park, dict):
        team_to_park = {}
    park_key = team_to_park.get(team_id_raw) or team_to_park.get(team_name)
    if not park_key:
        return None
    parks = park_factors.get("parks", {})
    if not isinstance(parks, dict):
        return None
    return parks.get(park_key)


def apply_park_adjustments(
    tool_scores: Dict[str, float],
    tool_category: str,
    park_config: Optional[Dict[str, Any]],
    adjustment_strength: float,
    player_handedness: Optional[str] = None,
    use_handedness_splits: bool = False,
) -> Dict[str, float]:
    """
    Apply park factor multipliers to tool scores (multiplicative, before weighting).

    Only tools with explicit multipliers in the park config are adjusted; others unchanged.
    Formula: effective_multiplier = 1.0 + ((base_multiplier - 1.0) * adjustment_strength).

    Args:
        tool_scores: Dictionary of {tool_name: score}.
        tool_category: One of 'batting', 'defense', 'baserunning', 'pitcher_ability'.
        park_config: Park configuration from park_factors.json (tool_adjustments, handedness_splits).
        adjustment_strength: Strength multiplier (0.0–1.0) from application_rules.
        player_handedness: 'L' or 'R' for batting handedness (optional).
        use_handedness_splits: Whether to use handedness-specific adjustments (batting only).

    Returns:
        Adjusted tool scores dictionary (same keys; values multiplied where config has multiplier).
    """
    if not park_config:
        return tool_scores.copy()
    tool_adjustments = (park_config.get("tool_adjustments") or {}).get(tool_category, {})
    if not isinstance(tool_adjustments, dict):
        return tool_scores.copy()
    if (
        tool_category == "batting"
        and use_handedness_splits
        and player_handedness in ("L", "R")
    ):
        handedness_key = "LHB" if player_handedness == "L" else "RHB"
        handedness_adj = (park_config.get("handedness_splits") or {}).get(handedness_key, {})
        if isinstance(handedness_adj, dict):
            tool_adjustments = {**tool_adjustments, **handedness_adj}
    adjusted = tool_scores.copy()
    for tool_name, score in adjusted.items():
        if tool_name not in tool_adjustments:
            continue
        base_mult = tool_adjustments[tool_name]
        try:
            base_mult = float(base_mult)
        except (TypeError, ValueError):
            continue
        effective = 1.0 + ((base_mult - 1.0) * adjustment_strength)
        adjusted[tool_name] = score * effective
    return adjusted


# -----------------------------------------------------------------------------
# Normalization (20–80 sigmoid)
# -----------------------------------------------------------------------------

def normalize_to_20_80(
    raw_score: float,
    center: float = 50.0,
    scale: float = 15.0,
    floor: float = 20.0,
    ceiling: float = 80.0,
) -> float:
    """
    Sigmoid-based normalization to 20–80 scale.

    Formula: center + (shifted / (scale * (1 + abs(shifted / scale)))) * 30
    so scores near center stay close; extremes compress smoothly.
    """
    shifted = raw_score - center
    denom = scale * (1.0 + abs(shifted / scale))
    normalized = (shifted / denom) * 30.0
    out = center + normalized
    return max(floor, min(ceiling, out))


# -----------------------------------------------------------------------------
# League / team labels
# -----------------------------------------------------------------------------

def get_league_label(lg_lvl: Optional[int], league_lookup: Dict[int, str]) -> str:
    """League level label for display and config lookup (R -> Rookie for config)."""
    if lg_lvl is None:
        return ""
    label = league_lookup.get(lg_lvl, "")
    return label


def get_league_key_for_config(display_label: str) -> str:
    """Key to use in config level_targets (e.g. R -> Rookie)."""
    return LEVEL_LABEL_TO_CONFIG.get(display_label, display_label)


def get_team_display(team_id: Optional[int], teams: Dict[int, str]) -> str:
    """Team display name."""
    if team_id is None:
        return ""
    return teams.get(team_id, str(team_id) if team_id else "")


# -----------------------------------------------------------------------------
# Hitter evaluation
# -----------------------------------------------------------------------------

def _weighted_sum_from_dict(tool_dict: Dict[str, float], weights: Dict[str, float]) -> Optional[float]:
    """Weighted average from tool->value dict and config weights. Returns None if no overlap."""
    total = 0.0
    weight_sum = 0.0
    for tool, w in weights.items():
        if tool.startswith("_") or w <= 0:
            continue
        if tool not in tool_dict:
            continue
        total += tool_dict[tool] * w
        weight_sum += w
    if weight_sum <= 0:
        return None
    return total / weight_sum


def hitter_batting_score(
    row: Dict[str, str],
    weights: Dict[str, float],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
    use_potential: bool = False,
) -> Optional[float]:
    """Weighted average of Gap, Pow, Eye, Ks (or Pot* when use_potential); optionally park-adjusted."""
    tool_dict: Dict[str, float] = {}
    for tool, w in weights.items():
        if tool.startswith("_"):
            continue
        col = (HITTER_BATTING_CURRENT_TO_POTENTIAL.get(tool) or tool) if use_potential else tool
        v = resolve_float(row, col)
        if v is not None:
            tool_dict[tool] = v
    if park_config and park_rules:
        strength = float(park_rules.get("adjustment_strength", 1.0))
        use_splits = bool(park_rules.get("use_handedness_splits", False))
        bats = (row.get("Bats") or "").strip().upper()
        handedness = bats[:1] if bats and bats[0] in ("L", "R") else None
        tool_dict = apply_park_adjustments(
            tool_dict, "batting", park_config, strength, handedness, use_splits
        )
    return _weighted_sum_from_dict(tool_dict, weights)


def hitter_defense_score(
    row: Dict[str, str],
    pos: str,
    pos_weights: Dict[str, float],
    standards: Dict[str, int],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Defense score for one position; None if standards not met. Optionally park-adjusted."""
    for attr, minimum in (standards or {}).items():
        if attr.startswith("_"):
            continue
        v = resolve_float(row, attr)
        if v is not None and v < minimum:
            return None
    tool_dict: Dict[str, float] = {}
    for attr, w in (pos_weights or {}).items():
        if attr.startswith("_") or w <= 0:
            continue
        v = resolve_float(row, attr)
        if v is not None:
            tool_dict[attr] = v
    if park_config and park_rules:
        strength = float(park_rules.get("adjustment_strength", 1.0))
        tool_dict = apply_park_adjustments(
            tool_dict, "defense", park_config, strength, None, False
        )
    return _weighted_sum_from_dict(tool_dict, pos_weights or {})


def hitter_baserunning_score(
    row: Dict[str, str],
    weights: Dict[str, float],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Weighted sum of Speed, Run, StealAbi/Steal, StlRt; optionally park-adjusted."""
    tool_dict: Dict[str, float] = {}
    for tool, w in weights.items():
        if tool.startswith("_"):
            continue
        if tool == "StealAbi":
            v = resolve_float(row, *BASERUNNING_STEAL_COLS)
        else:
            v = resolve_float(row, tool)
        if v is not None:
            tool_dict[tool] = v
    if park_config and park_rules:
        strength = float(park_rules.get("adjustment_strength", 1.0))
        tool_dict = apply_park_adjustments(
            tool_dict, "baserunning", park_config, strength, None, False
        )
    return _weighted_sum_from_dict(tool_dict, weights)


def hitter_position_scores(
    row: Dict[str, str],
    cfg: Dict[str, Any],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
    use_potential: bool = False,
) -> Tuple[float, float, float, Dict[str, Optional[float]], str, float]:
    """
    Batting, defense, baserunning; per-position scores (composite for viable, bat-only for DH); ideal position; ideal value.
    use_potential=True uses PotGap/PotPow/PotEye/PotKs for batting only (defense/baserunning have no Pot* in CSV).
    """
    h = cfg.get("hitters", {})
    tool_cats = h.get("tool_categories", {})
    bat_weights = tool_cats.get("batting", {})
    base_weights = tool_cats.get("baserunning", {})
    def_weights_by_pos = tool_cats.get("defense", {})
    pos_cat_weights = h.get("position_category_weights", {})
    standards = h.get("positional_standards", {})

    bat = hitter_batting_score(row, bat_weights, park_config, park_rules, use_potential) or 0.0
    base = hitter_baserunning_score(row, base_weights, park_config, park_rules) or 0.0

    pos_scores: Dict[str, Optional[float]] = {}
    def_sum = 0.0
    def_count = 0
    for pos in HITTER_POSITIONS:
        if pos == "DH":
            pos_scores[pos] = bat
            continue
        def_w = def_weights_by_pos.get(pos)
        std = standards.get(pos, {})
        def_score = (
            hitter_defense_score(row, pos, def_w or {}, std, park_config, park_rules)
            if def_w
            else None
        )
        if def_score is None:
            pos_scores[pos] = None
            continue
        def_sum += def_score
        def_count += 1
        cat_w = pos_cat_weights.get(pos, {})
        if not cat_w:
            pos_scores[pos] = def_score
            continue
        bat_w = cat_w.get("batting", 0.0) or 0.0
        def_wt = cat_w.get("defense", 0.0) or 0.0
        base_wt = cat_w.get("baserunning", 0.0) or 0.0
        pos_value = bat * bat_w + def_score * def_wt + base * base_wt
        pos_scores[pos] = pos_value

    def_avg = def_sum / def_count if def_count else 0.0
    ideal_value = bat
    ideal_pos = "DH"
    for pos in HITTER_POSITIONS:
        s = pos_scores.get(pos)
        if s is not None and s > ideal_value:
            ideal_value = s
            ideal_pos = pos
    return bat, def_avg, base, pos_scores, ideal_pos, ideal_value


# -----------------------------------------------------------------------------
# Pitcher evaluation
# -----------------------------------------------------------------------------

def pitcher_ability_score(
    row: Dict[str, str],
    role_weights: Dict[str, float],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
    use_potential: bool = False,
) -> Optional[float]:
    """Ability = weighted sum of Stuff, Movement, Control, HR_Avoid (or Pot* when use_potential); optionally park-adjusted."""
    tool_dict: Dict[str, float] = {}
    for csv_col, config_key in PITCHER_ABILITY_CSV_TO_CONFIG.items():
        if use_potential:
            pot_col = PITCHER_ABILITY_CURRENT_TO_POTENTIAL.get(csv_col, csv_col)
            alts = [pot_col] if pot_col else PITCHER_ABILITY_COL_ALTERNATIVES.get(config_key, [csv_col])
        else:
            alts = PITCHER_ABILITY_COL_ALTERNATIVES.get(config_key, [csv_col])
        v = resolve_float(row, *alts)
        if v is not None:
            tool_dict[config_key] = v
    if park_config and park_rules:
        strength = float(park_rules.get("adjustment_strength", 1.0))
        tool_dict = apply_park_adjustments(
            tool_dict, "pitcher_ability", park_config, strength, None, False
        )
    return _weighted_sum_from_dict(tool_dict, role_weights)


def pitcher_arsenal_score(
    row: Dict[str, str],
    role: str,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Arsenal score and diversity adjustment for SP or RP.
    Returns (arsenal_raw, diversity_adj).
    """
    ae = cfg.get("pitchers", {}).get("arsenal_evaluation", {})
    type_values = ae.get("pitch_type_values", {})
    slot_weights = ae.get("pitch_slot_weights", {}).get(role, {})
    div_req = ae.get("diversity_requirements", {}).get(role, {})
    div_mod = ae.get("diversity_modifiers", {})

    min_pitches = int(div_req.get("min_pitches", 3))
    min_vel = int(div_req.get("min_velocity_tiers", 2))
    min_break = int(div_req.get("min_break_planes", 2))
    vel_bonus = float(div_mod.get("velocity_tier_bonus", 0.0))
    break_bonus = float(div_mod.get("break_plane_bonus", 0.0))
    insufficient_penalty = float(div_mod.get("insufficient_pitches_penalty", 0.0))

    # Rank pitches by (rating * type_value), take top 4 for SP, 3 for RP
    slots = ["primary", "secondary", "tertiary", "quaternary"] if role == "SP" else ["primary", "secondary", "tertiary"]
    pitch_values: List[Tuple[float, str, str]] = []
    speed_tiers: Set[str] = set()
    break_planes: Set[str] = set()

    for col, ptype in POT_PITCH_COLUMN_TO_TYPE.items():
        v = resolve_float(row, col)
        if v is None or v <= 0:
            continue
        val = type_values.get(ptype, 1.0)
        if not isinstance(val, (int, float)):
            val = 1.0
        pitch_values.append((v * val, ptype, col))
        speed_tiers.add(PITCH_SPEED_TIERS.get(ptype, "other"))
        break_planes.add(PITCH_BREAK_PLANES.get(ptype, "other"))

    pitch_values.sort(key=lambda x: -x[0])
    raw_arsenal = 0.0
    for i, slot in enumerate(slots):
        if i >= len(pitch_values):
            break
        w = slot_weights.get(slot, 0.0) or 0.0
        raw_arsenal += pitch_values[i][0] * w
    # Scale to roughly 20–80: assume pitch ratings ~20–80, so sum of weighted ratings
    num_pitches = len(pitch_values)
    diversity_adj = 0.0
    if num_pitches < min_pitches:
        diversity_adj += insufficient_penalty
    if len(speed_tiers) >= min_vel:
        diversity_adj += vel_bonus
    if len(break_planes) >= min_break:
        diversity_adj += break_bonus
    return raw_arsenal, diversity_adj


def pitcher_combined_score(
    row: Dict[str, str],
    role: str,
    cfg: Dict[str, Any],
    park_config: Optional[Dict[str, Any]] = None,
    park_rules: Optional[Dict[str, Any]] = None,
    use_potential: bool = False,
) -> Tuple[float, float, float]:
    """Ability score, arsenal score (with diversity), and combined. use_potential uses Pot* for ability; arsenal already uses Pot* pitches."""
    pit = cfg.get("pitchers", {})
    ability_weights = pit.get("ability_weights", {}).get(role, {})
    role_balance = pit.get("role_balance", {}).get(role, {})
    stamina_cfg = pit.get("stamina_requirements", {}).get("SP", {})

    ability = pitcher_ability_score(row, ability_weights, park_config, park_rules, use_potential) or 0.0
    arsenal_raw, div_adj = pitcher_arsenal_score(row, role, cfg)
    arsenal = arsenal_raw + div_adj  # raw is on scale; div_adj is small bonus/penalty

    ab_w = float(role_balance.get("ability_weight", 0.8))
    ar_w = float(role_balance.get("arsenal_weight", 0.2))
    combined = ability * ab_w + arsenal * ar_w

    stamina_penalty = 0.0
    if role == "SP" and stamina_cfg:
        min_sta = float(stamina_cfg.get("minimum_stamina", 50))
        per_pt = float(stamina_cfg.get("penalty_per_point_below", 0.5))
        sta = resolve_float(row, "Stm")
        if sta is not None and sta < min_sta:
            stamina_penalty = (min_sta - sta) * per_pt
    combined -= stamina_penalty
    return ability, arsenal, combined


# -----------------------------------------------------------------------------
# Adjustments
# -----------------------------------------------------------------------------

def development_adjustment_hitter(row: Dict[str, str], cfg: Dict[str, Any]) -> float:
    """Current rating bonus + (gap to potential * 0.05). Only if avg potential >= 50."""
    tools = ["Gap", "Pow", "Eye", "Ks"]
    pots = ["PotGap", "PotPow", "PotEye", "PotKs"]
    cur = [resolve_float(row, t) for t in tools]
    pot = [resolve_float(row, p) for p in pots]
    cur = [c for c in cur if c is not None]
    pot = [p for p in pot if p is not None]
    if not cur or not pot:
        return 0.0
    avg_current = sum(cur) / len(cur)
    avg_potential = sum(pot) / len(pot)
    dev_cfg = (cfg.get("adjustments") or {}).get("development_trajectory") or {}
    hitter_cfg = dev_cfg.get("hitter") if isinstance(dev_cfg, dict) else {}
    min_pot = float(hitter_cfg.get("minimum_potential_for_bonus", 50)) if isinstance(hitter_cfg, dict) else 50.0
    if avg_potential < min_pot:
        return 0.0
    gap = avg_potential - avg_current
    if avg_current >= 55:
        current_bonus = 2.0
    elif avg_current >= 45:
        current_bonus = 1.0
    elif avg_current >= 35:
        current_bonus = 0.0
    elif avg_current >= 25:
        current_bonus = -0.5
    else:
        current_bonus = -1.5
    return current_bonus + (gap * 0.05)


def development_adjustment_pitcher(row: Dict[str, str], cfg: Dict[str, Any]) -> float:
    """Same idea for pitchers: Stf, Mov, HRA, Ctrl and Pot*."""
    tools = ["Stf", "Mov", "HRA", "Ctrl"]
    pots = ["PotStf", "PotMov", "PotHRA", "PotCtrl"]
    cur = [resolve_float(row, t) for t in tools]
    pot = [resolve_float(row, p) for p in pots]
    cur = [c for c in cur if c is not None]
    pot = [p for p in pot if p is not None]
    if not cur or not pot:
        return 0.0
    avg_current = sum(cur) / len(cur)
    avg_potential = sum(pot) / len(pot)
    dev = (cfg.get("adjustments") or {}).get("development_trajectory") or {}
    pit = dev.get("pitcher") if isinstance(dev, dict) else {}
    min_pot = float(pit.get("minimum_potential_for_bonus", 50)) if isinstance(pit, dict) else 50.0
    if avg_potential < min_pot:
        return 0.0
    gap = avg_potential - avg_current
    if avg_current >= 55:
        current_bonus = 2.0
    elif avg_current >= 45:
        current_bonus = 1.0
    elif avg_current >= 35:
        current_bonus = 0.0
    elif avg_current >= 25:
        current_bonus = -0.5
    else:
        current_bonus = -1.5
    return current_bonus + (gap * 0.05)


def age_adjustment(
    age: Optional[float],
    league_label: str,
    cfg: Dict[str, Any],
    role: str,
) -> float:
    """Bonus if young for level, penalty if old (from config level_targets)."""
    if age is None:
        return 0.0
    adj_cfg = (cfg.get("adjustments") or {}).get("age_vs_level") or {}
    role_cfg = adj_cfg.get(role, {}) if isinstance(adj_cfg, dict) else {}
    level_targets = role_cfg.get("level_targets", {}) if isinstance(role_cfg, dict) else {}
    key = get_league_key_for_config(league_label)
    level_cfg = level_targets.get(key) or level_targets.get(league_label) or {}
    if not level_cfg:
        return 0.0
    target_age = float(level_cfg.get("target_age", age))
    tolerance = max(0.1, float(level_cfg.get("tolerance_band", 2.0)))
    max_bonus = float(role_cfg.get("max_bonus", 3.0))
    max_penalty = float(role_cfg.get("max_penalty", -3.0))
    if age < target_age:
        ratio = min(1.0, (target_age - age) / tolerance)
        return ratio * max_bonus
    if age > target_age:
        ratio = min(1.0, (age - target_age) / tolerance)
        return ratio * max_penalty
    return 0.0


def _personality_bucket_from_cell(value: str) -> Optional[str]:
    """Map personality cell to config bucket: U=unknown (no modifier), H=high, N=normal, L=low."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().upper()
    if v == "H":
        return "high"
    if v == "N":
        return "normal"
    if v == "L":
        return "low"
    # U (unknown) or any other value: no modifier
    return None


def personality_adjustment(row: Dict[str, str], cfg: Dict[str, Any]) -> float:
    """Sum of trait modifiers from personality cells. Cells use U (unknown), H (high), N (normal), L (low).
    U or missing/other = no modifier. Only H/N/L apply the corresponding trait_modifiers."""
    impact = (cfg.get("adjustments") or {}).get("personality_impact") or {}
    if not isinstance(impact, dict):
        return 0.0
    mods = impact.get("trait_modifiers") or {}
    total = 0.0
    for csv_col, config_trait in PERSONALITY_CSV_TO_CONFIG.items():
        trait_mods = mods.get(config_trait) if isinstance(mods, dict) else {}
        if not isinstance(trait_mods, dict):
            continue
        raw = row.get(csv_col, "").strip() if csv_col in row else ""
        bucket = _personality_bucket_from_cell(raw)
        if bucket is None:
            continue
        m = trait_mods.get(bucket, 0.0)
        if isinstance(m, (int, float)):
            total += float(m)
    return total


# -----------------------------------------------------------------------------
# Output row building
# -----------------------------------------------------------------------------

def _normalization_params(cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    n = (cfg.get("normalization") or {})
    return (
        float(n.get("target_center", 50.0)),
        float(n.get("scale_parameter", 15.0)),
        float(n.get("hard_floor", 20.0)),
        float(n.get("hard_ceiling", 80.0)),
    )


def build_hitter_row(
    row: Dict[str, str],
    cfg: Dict[str, Any],
    league_lookup: Dict[int, str],
    teams: Dict[int, str],
    park_factors: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build one output row for a hitter. Returns None if insufficient data. Optionally applies park factors."""
    park_config = (
        get_player_park_config(row, park_factors, teams, league_lookup)
        if park_factors
        else None
    )
    park_rules = (park_factors.get("application_rules", {}) or {}) if park_factors else None
    try:
        bat, def_avg, base, pos_scores, ideal_pos, ideal_value = hitter_position_scores(
            row, cfg, park_config, park_rules, use_potential=False
        )
        _, _, _, _, _, ideal_value_pot = hitter_position_scores(
            row, cfg, park_config, park_rules, use_potential=True
        )
        h = cfg.get("hitters", {})
        bat_weights = (h.get("tool_categories") or {}).get("batting") or {}
        bat_pot = hitter_batting_score(row, bat_weights, park_config, park_rules, use_potential=True) or 0.0
    except Exception as e:
        logger.debug("Hitter score error for %s: %s", row.get("ID"), e)
        return None
    age = resolve_float(row, "Age")
    lg_lvl = resolve_int(row, "LgLvl")
    league_label = get_league_label(lg_lvl, league_lookup)
    team_id = resolve_int(row, "Team")
    org_id = resolve_int(row, "Org")
    dev_adj = development_adjustment_hitter(row, cfg)
    age_adj = age_adjustment(age, league_label, cfg, "hitter")
    pers_adj = personality_adjustment(row, cfg)
    raw_total = ideal_value + dev_adj + age_adj + pers_adj
    center, scale, floor, ceiling = _normalization_params(cfg)
    vos = normalize_to_20_80(raw_total, center, scale, floor, ceiling)
    # Potential VOS: base from potential ratings only; no development adj (already potential); age/personality apply
    raw_total_pot = ideal_value_pot + 0.0 + age_adj + pers_adj
    vos_potential = normalize_to_20_80(raw_total_pot, center, scale, floor, ceiling)
    out: Dict[str, Any] = {
        "ID": row.get("ID", ""),
        "Name": row.get("Name", ""),
        "Pos": row.get("Pos", ""),
        "Age": age if age is not None else "",
        "Team": get_team_display(team_id, teams),
        "Org": get_team_display(org_id, teams),
        "League_Level": league_label,
        "VOS_Score": round(vos, 2),
        "VOS_Potential": round(vos_potential, 2),
        "Batting_Score": round(bat, 2),
        "Batting_Potential": round(bat_pot, 2),
        "Defense_Score": round(def_avg, 2),
        "Baserunning_Score": round(base, 2),
        "Pitching_Ability_Score": "",
        "Pitching_Arsenal_Score": "",
        "Development_Adj": round(dev_adj, 2),
        "Age_Adj": round(age_adj, 2),
        "Personality_Adj": round(pers_adj, 2),
        "Park_Name": (park_config.get("name", "N/A") if park_config else "N/A"),
        "Park_Applied": park_config is not None,
    }
    for pos in HITTER_POSITIONS:
        s = pos_scores.get(pos)
        col = f"{pos}_Score"
        out[col] = round(s, 2) if s is not None else ""
    out["Ideal_Position"] = ideal_pos
    out["Ideal_Value"] = round(ideal_value, 2)
    return out


def build_pitcher_row(
    row: Dict[str, str],
    cfg: Dict[str, Any],
    league_lookup: Dict[int, str],
    teams: Dict[int, str],
    role: str = "SP",
    park_factors: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build one output row for a pitcher (evaluated as SP or RP). Optionally applies park factors to ability."""
    park_config = (
        get_player_park_config(row, park_factors, teams, league_lookup)
        if park_factors
        else None
    )
    park_rules = (park_factors.get("application_rules", {}) or {}) if park_factors else None
    try:
        ability, arsenal, combined = pitcher_combined_score(
            row, role, cfg, park_config, park_rules, use_potential=False
        )
        _, _, combined_pot = pitcher_combined_score(
            row, role, cfg, park_config, park_rules, use_potential=True
        )
    except Exception as e:
        logger.debug("Pitcher score error for %s: %s", row.get("ID"), e)
        return None
    age = resolve_float(row, "Age")
    lg_lvl = resolve_int(row, "LgLvl")
    league_label = get_league_label(lg_lvl, league_lookup)
    team_id = resolve_int(row, "Team")
    org_id = resolve_int(row, "Org")
    dev_adj = development_adjustment_pitcher(row, cfg)
    age_adj = age_adjustment(age, league_label, cfg, "pitcher")
    pers_adj = personality_adjustment(row, cfg)
    raw_total = combined + dev_adj + age_adj + pers_adj
    center, scale, floor, ceiling = _normalization_params(cfg)
    vos = normalize_to_20_80(raw_total, center, scale, floor, ceiling)
    # Potential VOS: ability from PotStf/PotMov/PotHRA/PotCtrl; arsenal already uses Pot* pitches; no dev adj
    raw_total_pot = combined_pot + 0.0 + age_adj + pers_adj
    vos_potential = normalize_to_20_80(raw_total_pot, center, scale, floor, ceiling)
    out: Dict[str, Any] = {
        "ID": row.get("ID", ""),
        "Name": row.get("Name", ""),
        "Pos": row.get("Pos", ""),
        "Age": age if age is not None else "",
        "Team": get_team_display(team_id, teams),
        "Org": get_team_display(org_id, teams),
        "League_Level": league_label,
        "VOS_Score": round(vos, 2),
        "VOS_Potential": round(vos_potential, 2),
        "Batting_Score": "",
        "Batting_Potential": "",
        "Defense_Score": "",
        "Baserunning_Score": "",
        "Pitching_Ability_Score": round(ability, 2),
        "Pitching_Arsenal_Score": round(arsenal, 2),
        "Development_Adj": round(dev_adj, 2),
        "Age_Adj": round(age_adj, 2),
        "Personality_Adj": round(pers_adj, 2),
        "Park_Name": (park_config.get("name", "N/A") if park_config else "N/A"),
        "Park_Applied": park_config is not None,
    }
    for pos in HITTER_POSITIONS:
        out[f"{pos}_Score"] = ""
    out["Ideal_Position"] = role
    out["Ideal_Value"] = round(combined, 2)
    return out


def is_pitcher(row: Dict[str, str]) -> bool:
    """True if primary position is pitcher (SP/RP)."""
    pos = (row.get("Pos") or "").strip().upper()
    return pos in ("SP", "RP", "P")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def write_output_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write evaluation summary CSV with consistent column order."""
    if not rows:
        logger.warning("No rows to write")
        return
    cols = [
        "ID", "Name", "Pos", "Age", "Team", "Org", "League_Level",
        "VOS_Score", "VOS_Potential", "Batting_Score", "Batting_Potential", "Defense_Score", "Baserunning_Score",
        "Pitching_Ability_Score", "Pitching_Arsenal_Score",
        "Development_Adj", "Age_Adj", "Personality_Adj",
        "Park_Name", "Park_Applied",
    ]
    pos_cols = [f"{p}_Score" for p in HITTER_POSITIONS]
    cols += pos_cols
    cols += ["Ideal_Position", "Ideal_Value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="VOS v2: Baseball player evaluation (20-80 normalized scores).")
    parser.add_argument("--league", required=True, help="League slug (e.g. woba, sky)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: evaluation_summary_{league}_{timestamp}.csv)")
    parser.add_argument("--ids-file", default=None, type=Path, help="Optional file of player IDs to include")
    parser.add_argument("--park-factors", default=None, type=str, help="Optional path to park-factors.json for ballpark-specific adjustments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="Config directory")
    args = parser.parse_args()

    config_dir = args.config_dir
    data_dir = args.data_dir
    league = args.league.strip()
    id_filter = load_id_filter(args.ids_file)

    cfg = load_weights(config_dir)
    if not cfg:
        logger.error("Weights config missing or invalid. Need %s", config_dir / WEIGHTS_FILENAME)
        return 1
    league_lookup = load_id_maps(config_dir)
    teams = load_teams(config_dir, league)
    park_factors = load_park_factors(args.park_factors)
    players = load_player_data(data_dir, league, id_filter)
    if not players:
        logger.error("No players loaded.")
        return 1

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output
    if out_path is None:
        out_path = Path(f"evaluation_summary_{league}_{ts}.csv")
    else:
        out_path = Path(out_path)

    rows: List[Dict[str, Any]] = []
    for row in players:
        if is_pitcher(row):
            out_row = build_pitcher_row(
                row, cfg, league_lookup, teams, role="SP", park_factors=park_factors
            )
        else:
            out_row = build_hitter_row(
                row, cfg, league_lookup, teams, park_factors=park_factors
            )
        if out_row is not None:
            rows.append(out_row)
        else:
            logger.debug("Skipped row ID %s", row.get("ID"))

    write_output_csv(rows, out_path)
    logger.info("Wrote %d rows to %s", len(rows), out_path)

    # Validation: VOS scores in 20-80
    vos_values = [r["VOS_Score"] for r in rows if isinstance(r.get("VOS_Score"), (int, float))]
    if vos_values:
        lo, hi = min(vos_values), max(vos_values)
        if lo < 20 or hi > 80:
            logger.warning("VOS range [%.2f, %.2f] outside 20-80", lo, hi)
        else:
            logger.info("VOS range [%.2f, %.2f] (within 20-80)", lo, hi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
