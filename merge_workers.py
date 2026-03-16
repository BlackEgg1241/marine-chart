"""Merge worker backtest results into main backtest_results.json."""
import json
import sys
from pathlib import Path

MAIN = Path("data/backtest/backtest_results.json")
WORKERS = [f"data/backtest_w{i}/backtest_results.json" for i in range(1, 21)]

def merge():
    main = json.loads(MAIN.read_text())
    existing = {e["date"] for e in main["dates"]}

    added = 0
    updated = 0
    for wf in WORKERS:
        p = Path(wf)
        if not p.exists():
            print(f"  SKIP {wf} (not found)")
            continue
        worker = json.loads(p.read_text())
        entries = worker.get("dates", [])
        print(f"  {wf}: {len(entries)} entries")
        for e in entries:
            if e["date"] in existing:
                # Update if worker has sub-scores and existing doesn't
                for i, m in enumerate(main["dates"]):
                    if m["date"] == e["date"]:
                        if "s_sst" in e and "s_sst" not in m:
                            main["dates"][i] = e
                            updated += 1
                        break
            else:
                main["dates"].append(e)
                existing.add(e["date"])
                added += 1

    # Sort by date
    main["dates"].sort(key=lambda x: x["date"])

    # Write
    MAIN.write_text(json.dumps(main, indent=2))
    total = len(main["dates"])
    with_sub = sum(1 for e in main["dates"] if "s_sst" in e)
    print(f"\nMerged: +{added} new, {updated} updated -> {total} total ({with_sub} with sub-scores)")

if __name__ == "__main__":
    merge()
