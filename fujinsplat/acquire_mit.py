"""Deterministic, coverage-first acquisition of 1400 distinct MIT training ARWs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request

from .io import sha256, write_json

INDEX = "https://projects.csail.mit.edu/illumination/download_instructions.html"
RAW_URL = "https://data.csail.mit.edu/multilum/raw"
NONFRONTAL = [i for i in range(25) if i not in [2, 3, 19, 20, 21, 22, 24]]
SEED = 90202


def fetch(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    (a.output / "listings").mkdir(exist_ok=True)
    (a.output / "raw").mkdir(exist_ok=True)
    plan_path = a.output / "PLAN.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        page = fetch(INDEX).decode()
        section = page.split("Train Scenes", 1)[1].split("Extra Scenes", 1)[0]
        scenes = sorted(set(re.findall(r'href="download_instructions/([^"/]+)\.txt"', section)))
        if len(scenes) != 985 or any(s.startswith("everett") for s in scenes):
            raise ValueError(f"official training-scene population drift: {len(scenes)}")
        order = sorted(scenes, key=lambda s: hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest())
        selection = [(s, 0) for s in order] + [(s, 1) for s in order[:1400 - len(order)]]
        quotas = {s: [j for ss, j in selection if ss == s] for s in order}
        def listing(scene):
            data = fetch(f"{RAW_URL}/{scene}/listing.txt")
            path = a.output / "listings" / f"{scene}.txt"
            if not path.exists():
                with path.open("xb") as f:
                    f.write(data)
            entries = [line.split() for line in data.decode().splitlines()[1:] if line.strip()]
            by_dir = {int(r[0]): r for r in entries if int(r[1]) == 1 and int(r[0]) in NONFRONTAL}
            ranked = sorted(by_dir, key=lambda d: hashlib.sha256(f"{SEED}|{scene}|{d}".encode()).hexdigest())
            if len(ranked) < len(quotas[scene]):
                raise ValueError(f"insufficient nonfrontal exposures: {scene}")
            result = []
            for j in quotas[scene]:
                d = ranked[j]; row = by_dir[d]
                name = f"{scene}_d{d:02d}_e1"
                result.append({"capture_id": name, "scene": scene, "direction": d, "exposure": 1,
                    "url": f"{RAW_URL}/{scene}/{row[2]}.arw", "raw": str((a.output / "raw" / f"{name}.ARW").resolve()),
                    "listing_sha256": hashlib.sha256(data).hexdigest()})
            return result
        rows = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(listing, s) for s in order]
            for i, f in enumerate(as_completed(futures), 1):
                rows.extend(f.result())
                if i % 50 == 0:
                    print(json.dumps({"listings": i, "total": len(order)}), flush=True)
        rows.sort(key=lambda r: r["capture_id"])
        assert len(rows) == len({r["url"] for r in rows}) == 1400
        plan = {"schema": "fujinsplat.mit_acquisition.v1", "seed": SEED, "scene_count": len(scenes),
                "policy": "one capture per all 985 official train scenes; second distinct lighting for 415 hash-ranked scenes; exposure_id=1; no frontal/ambient/test captures",
                "index_url": INDEX, "index_sha256": hashlib.sha256(page.encode()).hexdigest(), "rows": rows}
        write_json(plan_path, plan)
    def download(row):
        path = Path(row["raw"])
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            info = json.loads(receipt.read_text())
            if sha256(path) != info["sha256"]:
                raise ValueError(f"existing RAW hash drift: {path}")
            return {**row, **info}
        partial = a.scratch / f"{row['capture_id']}.partial"
        for attempt in range(4):
            try:
                digest = hashlib.sha256(); size = 0
                with urllib.request.urlopen(row["url"], timeout=90) as r, partial.open("wb") as f:
                    expected = r.headers.get("Content-Length")
                    while True:
                        block = r.read(2 << 20)
                        if not block:
                            break
                        f.write(block); digest.update(block); size += len(block)
                if expected is not None and size != int(expected):
                    raise IOError("download length mismatch")
                if size < 10_000_000:
                    raise IOError("not a full Sony RAW payload")
                stage = path.with_suffix(".staging")
                shutil.copyfile(partial, stage)
                if sha256(stage) != digest.hexdigest():
                    raise IOError("canonical publication hash mismatch")
                stage.replace(path)
                info = {"sha256": digest.hexdigest(), "bytes": size, "url": row["url"]}
                if not receipt.exists():
                    write_json(receipt, info)
                partial.unlink()
                return {**row, **info}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    completed = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output / "DOWNLOAD.jsonl").open("a") as log:
        futures = [pool.submit(download, row) for row in plan["rows"]]
        for i, f in enumerate(as_completed(futures), 1):
            row = f.result(); completed.append(row)
            log.write(json.dumps(row) + "\n"); log.flush()
            if i % 10 == 0 or i == 1:
                print(json.dumps({"downloaded": i, "total": 1400, "elapsed": time.monotonic() - start}), flush=True)
    completed.sort(key=lambda r: r["capture_id"])
    manifest = a.output / "MANIFEST.json"
    if not manifest.exists():
        write_json(manifest, {"schema": "fujinsplat.external_raw.v1", "rows": completed, "plan_sha256": sha256(plan_path)})
    print("COMPLETE_1400_DISTINCT_ARWS", flush=True)


if __name__ == "__main__":
    main()
