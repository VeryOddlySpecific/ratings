# VOS v2 (VOS Optimized Score)

Python script for baseball player evaluation using a weighted scoring system. Replaces the legacy `analyze.py` with a smaller, maintainable design and proper 20–80 normalization.

## Usage

```bash
python vos_v2.py --league <league>
```

**Examples:**

```bash
python vos_v2.py --league sky
python vos_v2.py --league woba --output evaluation_summary_woba.csv
python vos_v2.py --league sky --ids-file filter.txt
python vos_v2.py --league sky --park-factors config/park-factors-example.json
```

**Options:**

| Option | Description |
|--------|-------------|
| `--league` | **Required.** League slug (e.g. `sky`, `woba`). |
| `--output` | Output CSV path. Default: `evaluation_summary_{league}_{timestamp}.csv` |
| `--ids-file` | Optional file of player IDs (one per line or comma/semicolon separated) to limit evaluation. |
| `--park-factors` | Optional path to park-factors.json for ballpark-specific tool adjustments. |
| `--data-dir` | Data directory (default: `data`). |
| `--config-dir` | Config directory (default: `config`). |

## Inputs

- **PlayerData-{league}.csv** — In `data/`. Must include ID, Name, Pos, Age, Team, Org, LgLvl, and position-specific ratings (see script docstring).
- **weights_v2.json** — In `config/`. Defines batting/defense/baserunning and pitcher weights, positional standards, adjustments, and normalization.
- **teams-{league}.json** — In `config/`. Maps team IDs to names (e.g. `{"31": {"Name": "Arizona", "Nickname": "Diamondbacks", ...}}`).
- **id_maps.json** — In `config/`. Maps league level labels to numeric IDs (e.g. `{"league_level": {"ML": 1, "AAA": 2, ...}}`).

## Output

**evaluation_summary_{league}_{timestamp}.csv** (or path given by `--output`) with:

- **ID, Name, Pos, Age, Team, Org, League_Level**
- **VOS_Score** — Normalized to 20–80 scale (sigmoid-based).
- **Component scores:** Batting_Score, Defense_Score, Baserunning_Score (hitters); Pitching_Ability_Score, Pitching_Arsenal_Score (pitchers).
- **Adjustments:** Development_Adj, Age_Adj, Personality_Adj.
- **Position scores:** C_Score, 1B_Score, … DH_Score (hitters; empty for pitchers).
- **Park_Name, Park_Applied** — Home park name (or "N/A") and whether park factors were applied.
- **Ideal_Position, Ideal_Value** — Best position and its composite score (or SP/RP and combined score for pitchers).

## Park factors (optional)

When `--park-factors path/to/park-factors.json` is provided, two formats are supported:

### Single-park format (e.g. park-factors-lvk.json)

- **Use case:** Compare **all** players to one reference park (e.g. “how would everyone look in Las Vegas Knights Ballpark?”). No team lookup.
- **Input:** JSON with **top-level** `tool_adjustments` (batting, defense, baserunning, pitcher_ability), `team_info.park_name` (display name), optional `handedness_splits` (RHB/LHB), and `application_rules`.
- **Behavior:** The same park is applied to every player (subject to application_rules: apply_to_prospects, apply_to_major_leaguers).

### Multi-park format (e.g. park-factors.json)

- **Use case:** Apply each player’s **home team** park (team_to_park_mapping).
- **Input:** JSON with `parks` (park key → tool_adjustments, name), `team_to_park_mapping` (team ID or name → park key), and `application_rules`.

**Common:** Park multipliers are applied to **raw tool values before weighting**. Only tools with explicit multipliers are adjusted. `adjustment_strength` (0.0–1.0) scales strength. **Output:** `Park_Name` and `Park_Applied` in the CSV. **Fallback:** Missing or invalid file → warning and no park factors.

See `config/park-factors-lvk.json` (single-park) and `config/park-factors.json` (multi-park).

## Validation

The script logs the **VOS_Score** range after writing. All scores are clamped to the 20–80 band by the normalization function; the log confirms they fall within it.

## Architecture

- **Data loading** — CSV and JSON configs; missing columns handled via alternatives (e.g. `Steal` vs `StealAbi`).
- **Hitter evaluation** — Batting (Gap/Pow/Eye/Ks), defense per position (with positional standards), baserunning; composite position scores and ideal position.
- **Pitcher evaluation** — Ability (Stuff/Movement/Control/HR_Avoid), arsenal (pitch type + slot weights, diversity bonuses/penalties), stamina penalty for SP; combined score.
- **Adjustments** — Development (current vs potential tiers + gap), age vs level (target age and tolerance from config), personality (trait modifiers).
- **Park factors (optional)** — Multiplicative tool adjustments by home park (batting, defense, baserunning, pitcher_ability) before weighting; applied only when `--park-factors` is set and application_rules/team mapping match.
- **Normalization** — `normalize_to_20_80()`: sigmoid-style compression so values near 50 stay similar and extremes map into 20–80.
- **Output** — Single CSV with one row per player (pitchers evaluated as SP), including Park_Name and Park_Applied when park factors are used.

No hardcoded values; weights, thresholds, and modifiers come from `weights_v2.json` (and park multipliers from park-factors.json when provided).
