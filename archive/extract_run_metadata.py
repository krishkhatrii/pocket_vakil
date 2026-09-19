#!/usr/bin/env python
"""Pull every manifest.yaml field that can be recovered from local files.

Reads the checkpoint's own optimizer_state_dict rather than train.py, because
param_groups records what ACTUALLY ran -- argparse defaults in the source can
differ from what the launch command passed.

Run from the folder holding your downloaded files. Adjust --ckpt / --runs if
your layout differs.

    python extract_run_metadata.py --ckpt step_10000.pt --runs runs_backup
"""
import argparse
import json
import re
from pathlib import Path


def dump_checkpoint(path):
    import torch
    print("=" * 68)
    print(f"CHECKPOINT: {path}")
    print("=" * 68)
    ck = torch.load(path, map_location="cpu", weights_only=False)

    print("\ntop-level keys:", list(ck.keys()))
    for k in ("step", "epoch", "loss", "total_tokens_processed", "run_id",
              "stage", "tokenizer_sha256"):
        if k in ck:
            print(f"  {k} = {ck[k]}")

    cfg = ck.get("config")
    if cfg is not None:
        print("\n--- config ---")
        d = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", cfg)
        print(json.dumps(d, indent=2, default=str))

    osd = ck.get("optimizer_state_dict")
    if osd and "param_groups" in osd:
        print("\n--- optimizer param_groups (AUTHORITATIVE) ---")
        for i, g in enumerate(osd["param_groups"]):
            slim = {k: v for k, v in g.items() if k != "params"}
            slim["n_params"] = len(g.get("params", []))
            print(f"  group {i}: {json.dumps(slim, default=str)}")

    ssd = ck.get("scheduler_state_dict")
    if ssd:
        print("\n--- scheduler_state_dict ---")
        print(json.dumps({k: v for k, v in ssd.items()
                          if not isinstance(v, (list, dict))},
                         indent=2, default=str))

    if "sampler_state" in ck:
        s = ck["sampler_state"]
        print("\n--- sampler_state (seed may live here) ---")
        print(json.dumps(s, indent=2, default=str)[:600] if isinstance(s, dict)
              else str(s)[:600])

    sc = ck.get("scaler_state_dict")
    if sc:
        print("\n--- scaler_state_dict (confirms fp16 vs bf16) ---")
        print(json.dumps(sc, indent=2, default=str)[:400])

    del ck


def scan_logs(runs):
    runs = Path(runs)
    print("\n" + "=" * 68)
    print("CHECKPOINT-SAVE LOG (step <-> file)")
    print("=" * 68)
    for log in sorted(runs.rglob("*.log")) + sorted(runs.rglob("*.md")):
        hits = []
        try:
            for line in log.open(encoding="utf-8", errors="ignore"):
                if "[ckpt]" in line or "[milestone]" in line:
                    hits.append(line.rstrip())
        except Exception:
            continue
        if hits:
            print(f"\n{log}")
            for h in hits:
                print("  ", h)

    print("\n" + "=" * 68)
    print("LAUNCH COMMANDS / SEED / LR (grep)")
    print("=" * 68)
    pat = re.compile(r"(train\.py|--lr|--seed|--warmup|--batch-size|"
                     r"--grad-accum|--restart|Initialized \w*Scheduler|"
                     r"precision:|OLM \d)", re.I)
    for f in sorted(runs.rglob("*")):
        if not f.is_file() or f.suffix not in (".log", ".md", ".txt"):
            continue
        hits = []
        try:
            for line in f.open(encoding="utf-8", errors="ignore"):
                if pat.search(line):
                    hits.append(line.rstrip()[:400])
        except Exception:
            continue
        if hits:
            print(f"\n{f}")
            for h in hits[:12]:
                print("  ", h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="step_10000.pt")
    ap.add_argument("--runs", default="runs_backup")
    args = ap.parse_args()

    if Path(args.ckpt).exists():
        dump_checkpoint(args.ckpt)
    else:
        print(f"[skip] {args.ckpt} not found -- pass --ckpt <path>")

    if Path(args.runs).exists():
        scan_logs(args.runs)
    else:
        print(f"[skip] {args.runs} not found -- pass --runs <path>")


if __name__ == "__main__":
    main()
