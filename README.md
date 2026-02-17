# Ratings — VOS Player Evaluation

A Python tool for **baseball player evaluation** using a configurable weighted scoring system. It computes a single **VOS (VOS Optimized Score)** for each player, normalized to the familiar **20–80 scale**, from batting, defense, baserunning (hitters) or pitching ability and arsenal (pitchers).

Built for use with league exports (e.g. from OOTP or similar) and designed to be league-agnostic: you supply a league slug, player data CSV, and config files; the script produces an evaluation summary CSV.

---

## What it does

- **Hitters:** Combines batting tools (Gap, Power, Eye, Ks), position-specific defense, and baserunning into component scores, then applies development, age, and personality adjustments. Outputs a composite **VOS_Score** plus per-position scores and an **Ideal_Position**.
- **Pitchers:** Evaluates ability (Stuff, Movement, Control, HR Avoid) and arsenal (pitch mix, diversity), applies a stamina penalty for starters, then normalizes to 20–80.
- **Optional park factors:** Apply ballpark-specific tool adjustments (single-park or multi-park) so ratings reflect a chosen environment or each player’s home park.
- **Config-driven:** Weights, positional standards, age targets, and normalization parameters live in JSON; no hardcoded values in the script.

---

## Requirements

- **Python 3.7+** (standard library only; no external dependencies)

---

## Quick start

```bash
python vos_v2.py --league <league>
```

Example:

```bash
python vos_v2.py --league sky
```

Output is written to `evaluation_summary_sky_<timestamp>.csv` (or use `--output` to set the path).

---

## Usage

| Option | Description |
|--------|-------------|
| `--league` | **Required.** League slug (e.g. `sky`, `woba`, `sahl`). |
| `--output` | Output CSV path. Default: `evaluation_summary_{league}_{timestamp}.csv` |
| `--ids-file` | Optional file of player IDs (one per line or comma/semicolon separated) to evaluate only those players. |
| `--park-factors` | Optional path to a park-factors JSON file for ballpark-adjusted ratings. |
| `--data-dir` | Data directory (default: `data`). |
| `--config-dir` | Config directory (default: `config`). |

**Examples:**

```bash
python vos_v2.py --league sky
python vos_v2.py --league woba --output evaluation_summary_woba.csv
python vos_v2.py --league sky --ids-file filter.txt
python vos_v2.py --league sky --park-factors config/park-factors-example.json
```

---

## Inputs

| Input | Location | Purpose |
|-------|----------|---------|
| **PlayerData-{league}.csv** | `data/` | Player IDs, names, positions, age, team, org, level, and tool ratings (batting, defense, baserunning, pitching, personality). |
| **weights_v2.json** | `config/` | Tool weights, positional standards, age targets, normalization settings. |
| **teams-{league}.json** | `config/` | Team ID → name (and optional park) mapping. |
| **id_maps.json** | `config/` | League level labels → numeric IDs. |

Column names can vary (e.g. `Steal` vs `StealAbi`); the script uses built-in alternatives where needed.

---

## Output

**evaluation_summary_{league}_{timestamp}.csv** (or the path given by `--output`) includes:

- **Identity:** ID, Name, Pos, Age, Team, Org, League_Level  
- **VOS_Score** — Overall rating on a 20–80 scale (sigmoid-based normalization).  
- **Component scores:** Batting_Score, Defense_Score, Baserunning_Score (hitters); Pitching_Ability_Score, Pitching_Arsenal_Score (pitchers).  
- **Adjustments:** Development_Adj, Age_Adj, Personality_Adj.  
- **Position scores:** C_Score through DH_Score (hitters); N/A for pitchers.  
- **Ideal_Position / Ideal_Value** — Best defensive position and its composite score (or SP/RP and combined score for pitchers).  
- **Park_Name, Park_Applied** — Filled when `--park-factors` is used.

---

## Organizational Depth Analysis

After running VOS, you can analyze organizational depth with **org_depth_analysis.py**. It reads the evaluation summary CSV and identifies weak spots, stockpiles, and strategic opportunities (draft/acquisition targets, trade candidates).

```bash
python org_depth_analysis.py --league sky
python org_depth_analysis.py evaluation_summary_sky.csv -o "Atlanta Braves" --csv --html
```

- **Position strength scores** — Per-position depth, quality, and talent
- **Weak spots** — Positions needing draft/acquisition focus
- **Stockpiles** — Excess depth to consider for trades
- **Outputs** — Text report (default), plus optional CSV and HTML

For full options and workflow, see **[README_ORG_DEPTH_ANALYSIS.md](README_ORG_DEPTH_ANALYSIS.md)**.

---

## Park factors (optional)

With `--park-factors path/to/file.json` you can:

- **Single-park:** Compare all players as if they played in one reference park (e.g. a neutral or custom park).  
- **Multi-park:** Apply each player’s **home team** park via a team → park mapping.

Park multipliers adjust **raw tool values** before weighting. See `config/park-factors-lvk.json` (single-park) and the example multi-park format in the repo. Missing or invalid file results in a warning and no park adjustments.

---

## Project layout

```
ratings/
├── vos_v2.py              # Main evaluation script
├── org_depth_analysis.py  # Organizational depth analysis (consumes VOS output)
├── README.md              # This file (GitHub front page)
├── README_VOS_V2.md       # Detailed usage, options, and architecture
├── README_ORG_DEPTH_ANALYSIS.md  # Org depth tool usage and options
├── config/
│   ├── weights_v2.json    # Weights and normalization config
│   ├── id_maps.json       # League level ID mapping
│   ├── teams-{league}.json
│   └── park-factors*.json # Optional park factor files
├── data/
│   └── PlayerData-{league}.csv
└── old/
    └── analyze.py         # Legacy analyzer (replaced by vos_v2.py)
```

---

## More detail

- **VOS v2:** Full option descriptions, park factor formats, validation, and architecture — **[README_VOS_V2.md](README_VOS_V2.md)**
- **Org depth analysis:** Options, metrics, workflow — **[README_ORG_DEPTH_ANALYSIS.md](README_ORG_DEPTH_ANALYSIS.md)**
