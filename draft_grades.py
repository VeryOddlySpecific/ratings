#!/usr/bin/env python3
"""
Draft Grades - Compares draft results to VOS draft pool projections.
Reads draft_pool.md (or 05_draft_pool.md) from a directory, fetches current draft
status from the league API, and awards "VOS Stamps" when a player is drafted at
or after their projection. Top-100 projected players earn 3 points per stamp;
later projections earn 1 point each. A log-scaled bonus is added for how late
they were taken (delta = pick - projection), so steals add value without one
pick dominating. Reaches get 0 points (no penalty). Grades A–F are assigned by points range: range = max(points) − min(points);
the range is split into five equal bands, with A for the top band and F for the bottom.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request

# Default league draft API (override with --api-url)
DEFAULT_API_URL = "https://atl-01.statsplus.net/wwoba/api/draft/"

# Top 100: 3 points per stamp (drafted at or after projection). After top 100: 1 point each.
TOP_PROJECTION_CAP = 100
POINTS_TOP_100 = 3
POINTS_LATER = 1
# Log-scaled bonus for delta (pick - projection); only for stamps. Prevents one big steal from dominating.
DELTA_LOG_SCALE = 0.5

# Range-based grading: range = max(points) - min(points); five equal bands → A, B, C, D, F.
# Position within range (0 = min, 1 = max) maps to grade: [0, 0.2)=F, [0.2, 0.4)=D, [0.4, 0.6)=C, [0.6, 0.8)=B, [0.8, 1]=A.
GRADE_BANDS = [(0.2, "F"), (0.4, "D"), (0.6, "C"), (0.8, "B"), (1.0, "A")]


def _normalize_name(name: str) -> str:
    """Normalize name for matching: strip and collapse internal spaces."""
    if not name:
        return ""
    return " ".join(str(name).strip().split())


def find_draft_pool_md(directory: Path) -> Path:
    """Look for draft_pool.md or 05_draft_pool.md in directory."""
    for candidate in ("05_draft_pool.md", "draft_pool.md"):
        p = directory / candidate
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No draft pool file found in {directory}. Expected 05_draft_pool.md or draft_pool.md"
    )


def load_projections_from_md(md_path: Path) -> Dict[str, int]:
    """
    Parse draft pool markdown table (sorted by VOS, best first).
    Returns dict: normalized player name -> 1-based projection rank.
    Only includes first TOP_PROJECTION_CAP players for stamp eligibility.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name_to_rank: Dict[str, int] = {}
    rank = 0
    in_table = False
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|") if p.strip() != ""]
        if not parts:
            continue
        # Header row has "Name" in first column
        if parts[0].lower() == "name":
            in_table = True
            continue
        # Skip separator row (---)
        if in_table and "---" in line:
            continue
        if not in_table:
            continue
        name = _normalize_name(parts[0])
        if not name:
            continue
        rank += 1
        if rank <= TOP_PROJECTION_CAP:
            name_to_rank[name] = rank
        else:
            # Still record rank for delta/reference but no stamp eligibility
            name_to_rank[name] = rank
    return name_to_rank


def fetch_draft_csv(api_url: str) -> List[Dict[str, str]]:
    """Fetch draft status CSV from league API. Returns list of row dicts."""
    req = Request(api_url, headers={"User-Agent": "DraftGrades/1.0"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(
        (line for line in raw.splitlines() if line.strip()),
        quotechar='"',
        skipinitialspace=True,
    )
    rows = []
    for row in reader:
        # Normalize keys (API may use "Player Name" / "Overall" / "Team")
        rows.append({k.strip(): v for k, v in row.items()})
    return rows


def get_draft_value(row: Dict[str, str], *keys: str) -> Optional[str]:
    """Get first non-empty value from row for given keys."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def compare_draft_to_projections(
    draft_rows: List[Dict[str, str]],
    name_to_rank: Dict[str, int],
) -> List[Dict]:
    """
    For each drafted player, compute projection rank, delta, and VOS Stamp.
    Returns list of dicts for raw output CSV.
    """
    results = []
    for row in draft_rows:
        name = get_draft_value(row, "Player Name", "Player name", "Name")
        team = get_draft_value(row, "Team")
        overall_raw = get_draft_value(row, "Overall")
        if not name:
            continue
        overall = None
        if overall_raw is not None:
            try:
                overall = int(overall_raw)
            except ValueError:
                pass
        if overall is None:
            continue
        norm_name = _normalize_name(name)
        projection = name_to_rank.get(norm_name)
        if projection is None:
            # Try case-insensitive match as fallback
            for pool_name, r in name_to_rank.items():
                if pool_name.lower() == norm_name.lower():
                    projection = r
                    break
        delta = (overall - projection) if projection is not None else None
        # Stamps: at or after projection. Base pts + log-scaled delta bonus (reaches get 0, no penalty).
        points = 0.0
        stamp_type = ""
        if projection is not None and overall >= projection:
            safe_delta = max(0, delta if delta is not None else 0)
            log_bonus = DELTA_LOG_SCALE * math.log(1 + safe_delta)
            if projection <= TOP_PROJECTION_CAP:
                points = POINTS_TOP_100 + log_bonus
                stamp_type = "Top 100"
            else:
                points = POINTS_LATER + log_bonus
                stamp_type = "Later"
        results.append({
            "Player Name": name,
            "Team": team or "",
            "Overall Pick": overall,
            "Projection Rank": projection if projection is not None else "",
            "Delta": delta if delta is not None else "",
            "Stamp Type": stamp_type,
            "Points": points,
            "VOS Stamp": "Y" if points > 0 else "N",
        })
    return results


def aggregate_by_team(rows: List[Dict]) -> Dict[str, Dict]:
    """Per team: total points, top-100 stamp count, later stamp count. All teams that drafted appear."""
    # team -> {"points", "top_100", "later"} (points may be float due to log delta bonus)
    by_team: Dict[str, Dict] = {}
    for r in rows:
        team = (r.get("Team") or "").strip()
        if not team:
            continue
        if team not in by_team:
            by_team[team] = {"points": 0.0, "top_100": 0, "later": 0}
        pt = float(r.get("Points") or 0)
        by_team[team]["points"] += pt
        if r.get("Stamp Type") == "Top 100":
            by_team[team]["top_100"] += 1
        elif r.get("Stamp Type") == "Later":
            by_team[team]["later"] += 1
    return by_team


def compute_grades_by_range(team_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Assign grades based on points range: range = max(points) - min(points);
    five equal bands from min to max map to F, D, C, B, A.
    Returns team -> {grade, rank}. Rank is still by points desc (best = 1).
    """
    if not team_data:
        return {}
    points_list = [data["points"] for data in team_data.values()]
    min_pts = min(points_list)
    max_pts = max(points_list)
    span = max_pts - min_pts

    # Sort by points desc for rank; ties get best rank in group
    sorted_teams = sorted(
        team_data.items(),
        key=lambda x: (-x[1]["points"], x[0]),
    )
    result: Dict[str, Dict] = {}
    prev_pts = None
    for i, (team, data) in enumerate(sorted_teams):
        pts = data["points"]
        if pts != prev_pts:
            rank = i + 1
        prev_pts = pts
        if span == 0:
            grade = "C"
        else:
            # Position within range: 0 = min, 1 = max; five equal bands → F, D, C, B, A
            pos = (pts - min_pts) / span
            grade = "F"
            for cutoff, g in GRADE_BANDS:
                grade = g
                if pos < cutoff:
                    break
        result[team] = {"grade": grade, "rank": rank}
    return result


def write_raw_csv(rows: List[Dict], path: Path) -> None:
    """Write raw comparison data to CSV."""
    if not rows:
        return
    fieldnames = ["Player Name", "Team", "Overall Pick", "Projection Rank", "Delta", "Stamp Type", "Points", "VOS Stamp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_summary(team_data: Dict[str, Dict], path: Path) -> None:
    """Write summary: Team, Top 100 Stamps, Later Stamps, Total Points, Rank, Grade. Grades by points range (even bands)."""
    grades = compute_grades_by_range(team_data)
    rows = []
    for team, data in team_data.items():
        info = grades.get(team, {})
        rows.append({
            "Team": team,
            "Top 100 Stamps": data["top_100"],
            "Later Stamps": data["later"],
            "Total Points": round(data["points"], 1),
            "Rank": info.get("rank", ""),
            "Grade": info.get("grade", "F"),
        })
    rows.sort(key=lambda r: (r["Rank"] or 999, r["Team"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Team", "Top 100 Stamps", "Later Stamps", "Total Points", "Rank", "Grade"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade draft results vs VOS draft pool; output raw CSV and team summary."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing draft analysis output (e.g. 05_draft_pool.md)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"Draft API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (default: same as input directory)",
    )
    parser.add_argument(
        "--raw-name",
        type=str,
        default="draft_grades_raw.csv",
        help="Filename for raw comparison CSV",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="draft_grades_summary.csv",
        help="Filename for team summary CSV",
    )
    parser.add_argument(
        "--exclude-team",
        type=str,
        default=None,
        metavar="NAME",
        help="Exclude this team from all calculations and output (as if it did not exist)",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else directory

    try:
        pool_path = find_draft_pool_md(directory)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(f"Loading projections from {pool_path}...")
    name_to_rank = load_projections_from_md(pool_path)
    print(f"  Loaded {len(name_to_rank)} players (top {TOP_PROJECTION_CAP} eligible for VOS Stamp).")

    print(f"Fetching draft status from {args.api_url}...")
    try:
        draft_rows = fetch_draft_csv(args.api_url)
    except Exception as e:
        print(f"Error fetching draft API: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Fetched {len(draft_rows)} draft picks.")

    rows = compare_draft_to_projections(draft_rows, name_to_rank)

    if args.exclude_team:
        exclude_name = args.exclude_team.strip()
        orig_len = len(rows)
        rows = [r for r in rows if (r.get("Team") or "").strip().lower() != exclude_name.lower()]
        n_removed = orig_len - len(rows)
        print(f"  Excluding team {exclude_name!r}: removed {n_removed} picks from consideration.")

    top100_count = sum(1 for r in rows if r.get("Stamp Type") == "Top 100")
    later_count = sum(1 for r in rows if r.get("Stamp Type") == "Later")
    total_pts = sum(float(r.get("Points") or 0) for r in rows)
    print(f"  Stamps: {top100_count} top-100 (3 pts each), {later_count} later (1 pt each). Total points: {total_pts:.1f}.")

    team_data = aggregate_by_team(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / args.raw_name
    summary_path = output_dir / args.summary_name

    write_raw_csv(rows, raw_path)
    write_summary(team_data, summary_path)

    print(f"\nOutput written to {output_dir}:")
    print(f"  - {raw_path.name} (raw: delta, stamp type, points per pick)")
    print(f"  - {summary_path.name} (teams, top 100 / later stamps, total points, grade)")


if __name__ == "__main__":
    main()
