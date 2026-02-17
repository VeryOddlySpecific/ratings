#!/usr/bin/env python3
"""
Draft Pool Analysis - Generates a comprehensive analysis package from VOS v2
evaluation summary CSV files. Produces 6 reports for podcast, article, or
post-draft analysis.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

# VOS v2 tier benchmarks (fixed cutoffs, not percentiles)
VOS_TIER_BENCHMARKS = {
    "elite_min": 62,   # Truly exceptional prospects (top ~2-5%)
    "plus_min": 54,    # Quality starters, above average
    "average_min": 48, # Solid contributors, near league average
}

# Standard position set
POSITION_GROUPS = {
    "Infield": ["C", "1B", "2B", "3B", "SS"],
    "Outfield": ["LF", "CF", "RF"],
    "Pitching": ["SP", "RP"],
}


def get_column_value(row: Dict[str, Any], *possible_names: str) -> Optional[Any]:
    """Try multiple column names in order, return first found non-empty value."""
    for name in possible_names:
        if name in row:
            val = row[name]
            if val is not None and str(val).strip() != "":
                return val
    return None


def load_draft_pool(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load draft pool CSV with flexible column mapping (VOS v2 vs legacy).
    Filters out rows with invalid position or value data.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Draft pool file not found: {csv_path}")

    players: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row")
            for row in reader:
                ideal_pos = get_column_value(row, "Ideal_Position", "Ideal Pos")
                raw_value = get_column_value(row, "VOS_Potential", "VOS Potential")
                if ideal_pos is None or raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                row["_ideal_position"] = str(ideal_pos).strip()
                row["_ideal_value"] = value  # used for ranking/tiers (VOS_Potential)
                players.append(row)
        except csv.Error as e:
            raise ValueError(f"Invalid CSV format: {e}") from e

    if not players:
        raise ValueError("No valid players found after filtering (need Ideal_Position and VOS_Potential)")

    return players


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics. Handles empty lists (zeros), single value (std=0),
    and proper percentile indexing.
    Returns: count, mean, median, min, max, std_dev, p5, p10, p25, p75, p90, p95
    """
    result: Dict[str, float] = {
        "count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0,
        "std_dev": 0.0, "p5": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0,
    }
    if not values:
        return result

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result["count"] = float(n)
    result["mean"] = mean(sorted_vals)
    result["median"] = median(sorted_vals)
    result["min"] = min(sorted_vals)
    result["max"] = max(sorted_vals)

    if n >= 2:
        result["std_dev"] = stdev(sorted_vals)

    def percentile_index(p: float) -> int:
        """Nearest-rank percentile index (1-based conceptually, 0-based for list)."""
        idx = max(0, min(n - 1, int(round(p / 100.0 * n)) - 1))
        return max(0, idx)

    result["p5"] = sorted_vals[percentile_index(5)]
    result["p10"] = sorted_vals[percentile_index(10)]
    result["p25"] = sorted_vals[percentile_index(25)]
    result["p75"] = sorted_vals[percentile_index(75)]
    result["p90"] = sorted_vals[percentile_index(90)]
    result["p95"] = sorted_vals[percentile_index(95)]

    return result


def analyze_position_distribution(players: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count players by ideal position."""
    counts: Dict[str, int] = defaultdict(int)
    for p in players:
        pos = p.get("_ideal_position", "")
        if pos:
            counts[pos] += 1
    return dict(counts)


def analyze_position_strength(players: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calculate position-level statistics (count, mean, median, min, max, percentiles)."""
    by_pos: Dict[str, List[float]] = defaultdict(list)
    for p in players:
        pos = p.get("_ideal_position", "")
        val = p.get("_ideal_value")
        if pos is not None and val is not None:
            by_pos[pos].append(float(val))

    result: Dict[str, Dict[str, Any]] = {}
    for pos, vals in by_pos.items():
        stats = calculate_statistics(vals)
        result[pos] = {
            "count": int(stats["count"]),
            "mean": stats["mean"],
            "median": stats["median"],
            "min": stats["min"],
            "max": stats["max"],
            "p5": stats["p5"],
            "p25": stats["p25"],
            "p75": stats["p75"],
            "p95": stats["p95"],
        }
    return result


def categorize_prospects(
    players: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Categorize into Elite/Plus/Average/Org Depth using fixed VOS benchmarks.
    Returns (tier -> list of players, tier -> aggregate stats).
    """
    elite_min = VOS_TIER_BENCHMARKS["elite_min"]
    plus_min = VOS_TIER_BENCHMARKS["plus_min"]
    average_min = VOS_TIER_BENCHMARKS["average_min"]

    categories: Dict[str, List[Dict[str, Any]]] = {
        "Elite": [],
        "Plus": [],
        "Average": [],
        "Org Depth": [],
    }

    for p in players:
        val = p.get("_ideal_value")
        if val is None:
            continue
        v = float(val)
        if v >= elite_min:
            categories["Elite"].append(p)
        elif v >= plus_min:
            categories["Plus"].append(p)
        elif v >= average_min:
            categories["Average"].append(p)
        else:
            categories["Org Depth"].append(p)

    tier_stats: Dict[str, Any] = {}
    for tier, tier_players in categories.items():
        vals = [p["_ideal_value"] for p in tier_players]
        tier_stats[tier] = calculate_statistics(vals) if vals else calculate_statistics([])

    return categories, tier_stats


def generate_summary_report(
    players: List[Dict[str, Any]],
    position_counts: Dict[str, int],
    position_strength: Dict[str, Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate 00_summary.txt."""
    total = len(players)
    all_vals = [p["_ideal_value"] for p in players]
    pool_stats = calculate_statistics(all_vals)

    # Strongest/weakest by mean
    pos_means = [(pos, data["mean"], data["count"]) for pos, data in position_strength.items()]
    pos_means.sort(key=lambda x: -x[1])
    strongest = pos_means[:5]
    weakest = pos_means[-5:] if len(pos_means) >= 5 else pos_means
    weakest.reverse()

    lines = [
        "=" * 80,
        "DRAFT POOL ANALYSIS SUMMARY",
        "=" * 80,
        "",
        "OVERVIEW",
        "-" * 80,
        f"Total Players: {total}",
        f"Mean VOS Potential: {pool_stats['mean']:.2f}",
        f"Median VOS Potential: {pool_stats['median']:.2f}",
        "",
        "POSITION DISTRIBUTION",
        "-" * 80,
    ]
    for pos in sorted(position_counts.keys()):
        c = position_counts[pos]
        pct = 100.0 * c / total if total else 0
        lines.append(f"{pos}: {c} ({pct:.1f}%)")

    lines.extend([
        "",
        "STRONGEST POSITIONS (by average VOS Potential)",
        "-" * 80,
    ])
    for i, (pos, m, c) in enumerate(strongest, 1):
        lines.append(f"{i}. {pos}: {m:.2f} (n={c})")

    lines.extend([
        "",
        "WEAKEST POSITIONS (by average VOS Potential)",
        "-" * 80,
    ])
    for i, (pos, m, c) in enumerate(weakest, 1):
        lines.append(f"{i}. {pos}: {m:.2f} (n={c})")

    lines.extend([
        "",
        "PROSPECT TIER BREAKDOWN (fixed VOS benchmarks)",
        "-" * 80,
    ])
    for tier in ["Elite", "Plus", "Average", "Org Depth"]:
        c = len(categories[tier])
        pct = 100.0 * c / total if total else 0
        lines.append(f"{tier}: {c} ({pct:.1f}%)")

    lines.extend([
        "",
        "KEY METRICS FOR REFERENCE (tier benchmarks)",
        "-" * 80,
        "Elite: VOS Potential >= 62 (Truly exceptional prospects)",
        "Plus: VOS Potential 54-61 (Quality starters, above average)",
        "Average: VOS Potential 48-53 (Solid contributors, league average)",
        "Org Depth: VOS Potential < 48 (Below average, organizational depth)",
        f"Average Draft Pool Quality (VOS Potential): {pool_stats['mean']:.2f}",
        f"Median Draft Pool Quality (VOS Potential): {pool_stats['median']:.2f}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_position_distribution_report(
    position_counts: Dict[str, int],
    total: int,
    output_path: Path,
) -> None:
    """Generate 01_position_distribution.txt."""
    lines = [
        "=" * 80,
        "POSITION DISTRIBUTION",
        "=" * 80,
        "",
        f"Total Players: {total}",
        "",
        "Position         Count      Percentage     ",
        "-" * 80,
    ]
    for pos in sorted(position_counts.keys()):
        c = position_counts[pos]
        pct = 100.0 * c / total if total else 0
        lines.append(f"{pos:<16} {c:>6}    {pct:>6.1f}%")

    # Group summary
    group_counts: Dict[str, int] = defaultdict(int)
    for pos_name, pos_list in POSITION_GROUPS.items():
        for p in pos_list:
            group_counts[pos_name] += position_counts.get(p, 0)
    if "DH" in position_counts:
        group_counts["DH"] = position_counts["DH"]

    lines.extend([
        "",
        "=" * 80,
        "POSITION GROUPING SUMMARY",
        "=" * 80,
        "",
        "Category             Count      Percentage     ",
        "-" * 80,
    ])
    for cat in ["Infield", "Outfield", "Pitching"]:
        c = group_counts.get(cat, 0)
        pct = 100.0 * c / total if total else 0
        lines.append(f"{cat:<20} {c:>6}    {pct:>6.1f}%")
    if group_counts.get("DH", 0):
        c = group_counts["DH"]
        pct = 100.0 * c / total if total else 0
        lines.append(f"{'DH':<20} {c:>6}    {pct:>6.1f}%")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_position_strength_report(
    position_strength: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate 02_position_strength.txt."""
    # Sort by mean descending
    rows = [(pos, data) for pos, data in position_strength.items()]
    rows.sort(key=lambda x: -x[1]["mean"])

    lines = [
        "=" * 80,
        "POSITION STRENGTH ANALYSIS",
        "=" * 80,
        "",
        "Average VOS Potential by Position",
        "",
        "Position         Count      Mean       Median     Min        Max       ",
        "-" * 80,
    ]
    for pos, data in rows:
        lines.append(
            f"{pos:<16} {data['count']:>6}    {data['mean']:>6.2f}     {data['median']:>6.2f}   "
            f"{data['min']:>6.2f}    {data['max']:>6.2f}"
        )

    lines.extend([
        "",
        "=" * 80,
        "POSITION STRENGTH RANKINGS",
        "=" * 80,
        "",
        "Strongest positions (by average VOS Potential):",
    ])
    for i, (pos, data) in enumerate(rows[:10], 1):
        lines.append(f"  {i}. {pos}: {data['mean']:.2f} (n={data['count']})")

    lines.extend([
        "",
        "Weakest positions (by average VOS Potential):",
    ])
    for i, (pos, data) in enumerate(reversed(rows[-10:]), 1):
        lines.append(f"  {i}. {pos}: {data['mean']:.2f} (n={data['count']})")

    lines.extend([
        "",
        "=" * 80,
        "POSITION PERCENTILES",
        "=" * 80,
        "",
        "Position         P25        P50        P75        P95       ",
        "-" * 80,
    ])
    for pos, data in rows:
        lines.append(
            f"{pos:<16} {data['p25']:>6.2f}     {data['median']:>6.2f}     "
            f"{data['p75']:>6.2f}     {data['p95']:>6.2f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_ideal_value_distribution_report(
    players: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate 03_vos_potential_distribution.txt (distribution of VOS_Potential)."""
    vals = [p["_ideal_value"] for p in players]
    stats = calculate_statistics(vals)
    total = len(vals)

    ranges = [
        (90, 100), (80, 89), (70, 79), (60, 69), (50, 59),
        (40, 49), (30, 39), (20, 29),
    ]

    range_counts: List[Tuple[str, int]] = []
    for low, high in ranges:
        c = sum(1 for v in vals if low <= v <= high)
        range_counts.append((f"{low}-{high}", c))
    c_under = sum(1 for v in vals if v < 20)
    range_counts.append(("<20", c_under))

    lines = [
        "=" * 80,
        "VOS POTENTIAL DISTRIBUTION",
        "=" * 80,
        "",
        "SUMMARY STATISTICS",
        "-" * 80,
        f"Count: {int(stats['count'])}",
        f"Mean: {stats['mean']:.2f}",
        f"Median: {stats['median']:.2f}",
        f"Std Dev: {stats['std_dev']:.2f}",
        f"Min: {stats['min']:.2f}",
        f"Max: {stats['max']:.2f}",
        "",
        "PERCENTILES",
        "-" * 80,
        f"5th:  {stats['p5']:.2f}",
        f"10th: {stats['p10']:.2f}",
        f"25th: {stats['p25']:.2f}",
        f"50th (Median): {stats['median']:.2f}",
        f"75th: {stats['p75']:.2f}",
        f"90th: {stats['p90']:.2f}",
        f"95th: {stats['p95']:.2f}",
        "",
        "DISTRIBUTION BY RANGE",
        "-" * 80,
        "Range            Count      Percentage",
    ]
    for label, c in range_counts:
        pct = 100.0 * c / total if total else 0
        lines.append(f"{label:<16} {c:>6}    {pct:>6.1f}%")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_prospect_tier_report(
    categories: Dict[str, List[Dict[str, Any]]],
    tier_stats: Dict[str, Any],
    total: int,
    output_path: Path,
) -> None:
    """Generate 04_prospect_tiers.txt."""
    lines = [
        "=" * 80,
        "PROSPECT TIER BREAKDOWN",
        "=" * 80,
        "",
        "Fixed VOS Potential Benchmarks (calibrated for VOS v2):",
        "- Elite: VOS Potential >= 62 (Truly exceptional prospects)",
        "- Plus: VOS Potential 54-61 (Quality starters, above average)",
        "- Average: VOS Potential 48-53 (Solid contributors, league average)",
        "- Org Depth: VOS Potential < 48 (Below average, organizational depth)",
        "",
        "TIER SUMMARY",
        "-" * 80,
        "Tier             Count      Percentage  Mean       Median     Range",
    ]

    for tier in ["Elite", "Plus", "Average", "Org Depth"]:
        pl = categories[tier]
        st = tier_stats[tier]
        c = len(pl)
        pct = 100.0 * c / total if total else 0
        mn = st["mean"]
        med = st["median"]
        rmin = st["min"]
        rmax = st["max"]
        lines.append(f"{tier:<16} {c:>6}    {pct:>6.1f}%      {mn:>6.2f}     {med:>6.2f}   {rmin:.2f}-{rmax:.2f}")

    lines.append("")
    lines.append("DETAILED BREAKDOWN")
    lines.append("-" * 80)

    for tier in ["Elite", "Plus", "Average", "Org Depth"]:
        pl = categories[tier]
        st = tier_stats[tier]
        c = len(pl)
        pct = 100.0 * c / total if total else 0
        lines.append("")
        if tier == "Elite":
            lines.append("ELITE TIER (VOS Potential >= 62)")
        elif tier == "Plus":
            lines.append("PLUS TIER (VOS Potential 54-61)")
        elif tier == "Average":
            lines.append("AVERAGE TIER (VOS Potential 48-53)")
        else:
            lines.append("ORG DEPTH TIER (VOS Potential < 48)")
        lines.append(f"Total: {c} players ({pct:.1f}% of pool)")
        lines.append(f"Average VOS Potential: {st['mean']:.2f}")
        lines.append(f"Median VOS Potential: {st['median']:.2f}")
        lines.append(f"Range: {st['min']:.2f} - {st['max']:.2f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_csv_summary(
    position_counts: Dict[str, int],
    position_strength: Dict[str, Dict[str, Any]],
    categories: Dict[str, List[Dict[str, Any]]],
    total: int,
    output_path: Path,
) -> None:
    """Generate summary_data.csv."""
    lines = [
        "Position Distribution",
        "Position,Count,Percentage",
    ]
    for pos in sorted(position_counts.keys()):
        c = position_counts[pos]
        pct = f"{100.0 * c / total:.2f}%" if total else "0%"
        lines.append(f"{pos},{c},{pct}")

    lines.extend([
        "",
        "Position Strength",
        "Position,Count,Mean,Median,Min,Max,P25,P75",
    ])
    for pos in sorted(position_strength.keys()):
        d = position_strength[pos]
        lines.append(f"{pos},{d['count']},{d['mean']:.2f},{d['median']:.2f},{d['min']:.2f},{d['max']:.2f},{d['p25']:.2f},{d['p75']:.2f}")

    def tier_pct(count: int) -> str:
        return f"{100.0 * count / total:.2f}%" if total else "0.00%"

    lines.extend([
        "",
        "Prospect Tiers",
        "Tier,Count,Percentage,Threshold",
        f"Elite,{len(categories['Elite'])},{tier_pct(len(categories['Elite']))},>= 62",
        f"Plus,{len(categories['Plus'])},{tier_pct(len(categories['Plus']))},54-61",
        f"Average,{len(categories['Average'])},{tier_pct(len(categories['Average']))},48-53",
        f"Org Depth,{len(categories['Org Depth'])},{tier_pct(len(categories['Org Depth']))},< 48",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _tier_for_value(value: float) -> str:
    """Return prospect tier label for a given VOS_Potential value."""
    if value >= VOS_TIER_BENCHMARKS["elite_min"]:
        return "Elite"
    if value >= VOS_TIER_BENCHMARKS["plus_min"]:
        return "Plus"
    if value >= VOS_TIER_BENCHMARKS["average_min"]:
        return "Average"
    return "Org Depth"


def _md_escape(s: str) -> str:
    """Escape pipe and newline for markdown table cells."""
    if not s:
        return ""
    return str(s).replace("|", "\\|").replace("\n", " ").replace("\r", "").strip()


def generate_draft_pool_markdown(
    players: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Convert the draft pool (eval CSV data) to a Markdown file in the output directory.
    Produces a table sorted by VOS_Potential descending, with key columns plus tier.
    """
    # Sort by VOS Potential descending (best first)
    sorted_players = sorted(players, key=lambda p: p.get("_ideal_value") or 0, reverse=True)

    headers = ["Name", "Pos", "Age", "Org", "Ideal Pos", "VOS Potential", "Tier"]
    rows = []
    for p in sorted_players:
        name = _md_escape(get_column_value(p, "Name") or "")
        pos = _md_escape(get_column_value(p, "Pos") or "")
        age_raw = get_column_value(p, "Age")
        age = ""
        if age_raw is not None and str(age_raw).strip():
            try:
                age = _md_escape(str(int(float(age_raw))))
            except (TypeError, ValueError):
                age = _md_escape(str(age_raw))
        org = _md_escape(get_column_value(p, "Org") or "")
        ideal_pos = _md_escape(p.get("_ideal_position") or "")
        vos_pot = p.get("_ideal_value")
        vos_pot_str = f"{vos_pot:.2f}" if vos_pot is not None else ""
        tier = _tier_for_value(float(vos_pot)) if vos_pot is not None else ""
        rows.append([name, pos, age, org, ideal_pos, vos_pot_str, tier])

    lines = [
        "# Draft Pool (Evaluation Summary)",
        "",
        f"Total players: {len(players)}. Sorted by VOS Potential (best first).",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_input_path(path_arg: str, script_dir: Path) -> Path:
    """Resolve input CSV path; if relative and not found, try parent directory."""
    p = Path(path_arg)
    if not p.is_absolute():
        if (script_dir / p).exists():
            return script_dir / p
        if (script_dir.parent / p).exists():
            return script_dir.parent / p
        return script_dir / p  # Return as-is so load_draft_pool can raise FileNotFoundError
    return p


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate draft pool analysis package from VOS v2 evaluation summary CSV.",
    )
    parser.add_argument(
        "draft_pool",
        type=str,
        help="Path to draft pool CSV file (e.g. draft_evaluation_summary_*_with_draft.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory path",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for folder (draft_pool_analysis_{name})",
    )
    args = parser.parse_args()

    csv_path = resolve_input_path(args.draft_pool, script_dir)
    print(f"Loading draft pool from {csv_path}...")

    try:
        players = load_draft_pool(csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(players)} players")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        if args.name:
            folder_name = f"draft_pool_analysis_{args.name}"
        else:
            folder_name = f"draft_pool_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = script_dir.parent / folder_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nAnalyzing position distribution...")
    position_counts = analyze_position_distribution(players)
    print("Analyzing position strength...")
    position_strength = analyze_position_strength(players)
    print("Categorizing prospects...")
    categories, tier_stats = categorize_prospects(players)

    total = len(players)

    print("\nGenerating reports...")
    generate_summary_report(players, position_counts, position_strength, categories, output_dir / "00_summary.txt")
    print("  [ok] Summary report")
    generate_position_distribution_report(position_counts, total, output_dir / "01_position_distribution.txt")
    print("  [ok] Position distribution report")
    generate_position_strength_report(position_strength, output_dir / "02_position_strength.txt")
    print("  [ok] Position strength report")
    generate_ideal_value_distribution_report(players, output_dir / "03_vos_potential_distribution.txt")
    print("  [ok] VOS Potential distribution report")
    generate_prospect_tier_report(categories, tier_stats, total, output_dir / "04_prospect_tiers.txt")
    print("  [ok] Prospect tier report")
    generate_csv_summary(position_counts, position_strength, categories, total, output_dir / "summary_data.csv")
    print("  [ok] CSV summary")
    generate_draft_pool_markdown(players, output_dir / "05_draft_pool.md")
    print("  [ok] Draft pool Markdown")

    print("\n" + "=" * 80)
    print("DRAFT POOL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAnalysis package saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - 00_summary.txt (Quick reference summary)")
    print("  - 01_position_distribution.txt")
    print("  - 02_position_strength.txt")
    print("  - 03_vos_potential_distribution.txt")
    print("  - 04_prospect_tiers.txt")
    print("  - summary_data.csv (Data for further analysis)")
    print("  - 05_draft_pool.md (Draft pool table from eval CSV)")


if __name__ == "__main__":
    main()
