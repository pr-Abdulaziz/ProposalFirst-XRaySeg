#!/usr/bin/env python
"""Campaign progress from the frozen manifest, the scheduler and the append-only records.

For each run: scheduler state of its training and evaluation jobs, last status
event, epochs completed, update opportunities, AMP-skipped updates, non-finite
losses, last learning rate, peak memory and seconds per epoch. Read-only.

    python scripts/campaign_multiseed/campaign_status.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.campaign_multiseed.campaign_common import MANIFEST, run_dir  # noqa: E402


def job_states(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "JobID,State,Elapsed,NodeList", "-j", ",".join(ids)],
                         capture_output=True, text=True).stdout
    states = {}
    for line in out.strip().splitlines():
        jid, state, elapsed, node = (line.split("|") + ["", "", "", ""])[:4]
        states[jid] = f"{state.split()[0]} {elapsed} {node}".strip()
    return states


def main() -> int:
    man = json.loads(MANIFEST.read_text())
    subs = {}
    for e in man["run_order"]:
        f = run_dir(e["method"], e["seed"]) / "submission.json"
        if f.is_file():
            subs[e["run_key"]] = json.loads(f.read_text())
    states = job_states([j for s in subs.values() for j in (s["train_job"], s["eval_job"])])
    print(f"{'#':>2} {'run':42s} {'train job':28s} {'eval job':24s} {'ep':>3} {'upd':>6} {'skip':>4} {'nan':>3} "
          f"{'lr':>9} {'GB':>5} {'s/ep':>6}  last event")
    done_train = done_eval = 0
    for e in man["run_order"]:
        key = e["run_key"]
        s = subs.get(key)
        wd = Path(man["runs"][key]["work_dir"])
        acct = wd / "update_accounting.jsonl"
        last = json.loads(acct.read_text().splitlines()[-1]) if acct.is_file() and acct.stat().st_size else {}
        cum = last.get("cumulative", {})
        st = run_dir(e["method"], e["seed"]) / "status.jsonl"
        ev = json.loads(st.read_text().splitlines()[-1])["event"] if st.is_file() else "-"
        done_train += ev in ("train_complete", "eval_started", "eval_complete")
        done_eval += ev == "eval_complete"
        tj = f"{s['train_job']} {states.get(s['train_job'], '?')}" if s else "not submitted"
        ej = f"{s['eval_job']} {states.get(s['eval_job'], '?').split()[0]}" if s else ""
        print(f"{e['index']:>2} {key:42s} {tj[:28]:28s} {ej[:24]:24s} {last.get('epoch', 0):>3} "
              f"{cum.get('update_opportunities', 0):>6} {cum.get('amp_skipped_updates', 0):>4} "
              f"{cum.get('nonfinite_loss_iterations', 0):>3} {last.get('lr_last_update') or 0:>9.2e} "
              f"{(last.get('peak_reserved_mb') or 0) / 1024:>5.1f} {last.get('seconds') or 0:>6.0f}  {ev}")
    print(f"\ntraining complete: {done_train}/{len(man['run_order'])}   evaluated: {done_eval}/{len(man['run_order'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
