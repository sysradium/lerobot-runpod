"""Copy a LeRobot v3 dataset with one camera key renamed.

Runs standalone (via `uv run --with pandas --with pyarrow`) so the app itself
doesn't need pandas.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

src, dst, old, new = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
old_key, new_key = f"observation.images.{old}", f"observation.images.{new}"

if dst.exists():
    sys.exit(f"{dst} already exists")
shutil.copytree(src, dst)

info_path = dst / "meta/info.json"
info = json.loads(info_path.read_text())
if old_key not in info["features"]:
    shutil.rmtree(dst)
    sys.exit(f"{old_key} is not a feature of {src}")
info["features"] = {new_key if k == old_key else k: v for k, v in info["features"].items()}
info_path.write_text(json.dumps(info, indent=4))

stats_path = dst / "meta/stats.json"
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    stats[new_key] = stats.pop(old_key)
    stats_path.write_text(json.dumps(stats, indent=4))

# per-episode stats live in parquet columns named like stats/<feature>/<stat>
for pq in (dst / "meta/episodes").rglob("*.parquet"):
    df = pd.read_parquet(pq)
    df.columns = [c.replace(f"/{old_key}/", f"/{new_key}/") for c in df.columns]
    df.to_parquet(pq)

if (dst / "videos" / old_key).exists():
    (dst / "videos" / old_key).rename(dst / "videos" / new_key)

leftovers = [p for p in dst.rglob("*") if old_key in str(p)]
assert not leftovers, leftovers
print(f"renamed {old_key} -> {new_key} in {dst}")
