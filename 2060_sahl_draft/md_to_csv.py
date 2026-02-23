"""Convert 05_draft_pool.md table to CSV."""
import csv

input_path = "05_draft_pool.md"
output_path = "05_draft_pool.csv"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find table: skip title/blank, header row, separator row, then data
rows = []
for line in lines:
    line = line.rstrip("\n")
    if not line.startswith("|"):
        continue
    parts = [p.strip() for p in line.split("|")]
    # Split gives: ['', col1, col2, ..., col7, '']
    if len(parts) < 8:
        continue
    cells = [parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]]
    # Skip separator row (---)
    if cells[0].startswith("---") or cells[0] == "Name":
        continue
    rows.append(cells)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Pos", "Age", "Org", "Ideal Pos", "VOS Potential", "Tier"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {output_path}")
