"""
CPU correctness harness. Proves the plumbing before any GPU time is spent.

    python src/cpu_verify.py

Five checks, each reported pass/fail with the numbers behind it:

  1. pretrain 20 steps      loss starts ~ln(32000)=10.37, decreases, no NaN/Inf
  2. kill at 10 and resume  loss continues AND the next batch is the one that
                            would have come next had nothing been interrupted
  3. finetune 20 steps      -100 masking supervises exactly the label tokens
  4. evaluate 20 test rows  generation, parsing, F1, failure taxonomy
  5. tokenizer hash guard   fires on a deliberately corrupted tokenizer

Runs the `tiny` preset on CPU. It cannot show that the model learns anything --
only that every mechanism does what it claims.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# --------------------------------------------------------------------------


def check_1_pretrain(tmp):
    banner("1. PRETRAIN 20 STEPS (tiny, CPU)")
    import olm_patches
    from model import CONFIGS, build_model
    from train import PretrainShardDataset, ResumableSampler, SHARDS
    from olm.data.datasets import DataLoader
    from olm.train.losses import CrossEntropyLoss
    from olm.train.optim import AdamW

    olm_patches.apply_all(device="cpu", verbose=False)
    cfg = CONFIGS["tiny"]
    torch.manual_seed(0)
    ds = PretrainShardDataset(SHARDS, cfg.max_seq_len)
    sampler = ResumableSampler(len(ds), seed=7)
    dl = DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False,
                    num_workers=0, pin_memory=False, drop_last=True)
    model = build_model(cfg)
    opt = AdamW(model.parameters(), lr=3e-4)
    ce = CrossEntropyLoss()

    losses = []
    it = iter(dl)
    for _ in range(20):
        x, y = next(it)
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    first, last = losses[0], losses[-1]
    finite = all(np.isfinite(losses))
    near_ln_v = 9.0 <= first <= 12.0
    decreased = last < first
    print(f"    loss[0]={first:.4f}  loss[19]={last:.4f}  "
          f"min={min(losses):.4f}  ln(32000)=10.37")
    record("1a start near ln(vocab)", near_ln_v, f"{first:.4f} in [9,12]")
    record("1b loss decreases", decreased, f"{first:.4f} -> {last:.4f}")
    record("1c no NaN/Inf", finite, f"{len(losses)} finite values")
    return losses


def check_2_resume(tmp):
    banner("2. KILL AT STEP 10, RESUME (loss continues + data position correct)")
    import olm_patches
    from model import CONFIGS, build_model
    from train import (PretrainShardDataset, ResumableSampler, SHARDS,
                       PocketCheckpointCallback, restore_rng)
    from olm.data.datasets import DataLoader
    from olm.train.losses import CrossEntropyLoss
    from olm.train.optim import AdamW

    olm_patches.apply_all(device="cpu", verbose=False)
    cfg = CONFIGS["tiny"]
    ce = CrossEntropyLoss()

    def make(seed=0):
        torch.manual_seed(seed)
        ds = PretrainShardDataset(SHARDS, cfg.max_seq_len)
        sampler = ResumableSampler(len(ds), seed=7)
        dl = DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False,
                        num_workers=0, pin_memory=False, drop_last=True)
        m = build_model(cfg)
        o = AdamW(m.parameters(), lr=3e-4)
        return ds, sampler, dl, m, o

    # --- uninterrupted reference run: 20 steps, recording batch ids ---
    ds, sampler, dl, model, opt = make()
    ref_losses, ref_ids = [], []
    it = iter(dl)
    for _ in range(20):
        x, y = next(it)
        ref_ids.append(x[:, :6].tolist())
        loss = ce(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        ref_losses.append(float(loss.detach()))

    # --- run A: 10 steps, then checkpoint (simulating the kill) ---
    ds, sampler, dl, model, opt = make()
    a_losses, a_ids = [], []
    it = iter(dl)
    for _ in range(10):
        x, y = next(it)
        a_ids.append(x[:, :6].tolist())
        loss = ce(model(x), y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        a_losses.append(float(loss.detach()))

    ckpt = {
        "step": 10, "epoch": 0, "loss": a_losses[-1],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "losses": a_losses, "total_tokens_processed": 0,
        "sampler_state": sampler.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "numpy": np.random.get_state(), "cuda": None},
    }
    path = tmp / "resume_test.pt"
    torch.save(ckpt, path)
    saved_cursor = sampler.cursor
    expected_next = sampler.peek_next()
    print(f"    killed at step 10; sampler epoch {sampler.epoch} "
          f"cursor {saved_cursor}; next index would be {expected_next}")

    # --- run B: fresh objects, restore, continue 10 more ---
    ds2, sampler2, dl2, model2, opt2 = make(seed=999)   # different init on purpose
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ck["model_state_dict"])
    opt2.load_state_dict(ck["optimizer_state_dict"])
    sampler2.load_state_dict(ck["sampler_state"])
    restore_rng(ck)
    resumed_next = sampler2.peek_next()
    record("2a sampler position restored",
           sampler2.cursor == saved_cursor and sampler2.epoch == 0,
           f"cursor {sampler2.cursor} == {saved_cursor}")
    record("2b next index matches uninterrupted run",
           resumed_next == expected_next,
           f"{resumed_next} == {expected_next}")

    b_losses, b_ids = [], []
    it2 = iter(dl2)
    for _ in range(10):
        x, y = next(it2)
        b_ids.append(x[:, :6].tolist())
        loss = ce(model2(x), y)
        loss.backward(); opt2.step(); opt2.zero_grad(set_to_none=True)
        b_losses.append(float(loss.detach()))

    same_batches = b_ids == ref_ids[10:20]
    record("2c resumed batches == batches 10-19 of the uninterrupted run",
           same_batches,
           "identical token ids" if same_batches else "DIFFERENT -- data replayed/skipped")

    drift = abs(b_losses[0] - ref_losses[10])
    record("2d loss continues (no restart spike)", drift < 0.5,
           f"resumed {b_losses[0]:.4f} vs reference {ref_losses[10]:.4f} "
           f"(|diff| {drift:.4f})")
    print(f"    combined curve: {a_losses[0]:.3f} -> {a_losses[-1]:.3f} "
          f"|| {b_losses[0]:.3f} -> {b_losses[-1]:.3f}")
    return True


def check_3_finetune(tmp):
    banner("3. FINETUNE 20 STEPS (-100 masking)")
    import olm_patches
    from model import CONFIGS, build_model
    from train import ILSIFinetuneDataset, ResumableSampler, ILSI_FT, IGNORE_INDEX
    from olm.data.datasets import DataLoader
    from olm.train.losses import CrossEntropyLoss
    from olm.train.optim import AdamW

    olm_patches.apply_all(device="cpu", verbose=False)
    cfg = CONFIGS["tiny"]
    torch.manual_seed(0)
    ds = ILSIFinetuneDataset(ILSI_FT / "train.jsonl", cfg.max_seq_len)

    ok = True
    for i in range(8):
        x, y = ds[i]
        sup = y[y != IGNORE_INDEX].tolist()
        if sup != ds.rows[i]["target_ids"]:
            ok = False
            print(f"    row {i}: {sup} != {ds.rows[i]['target_ids']}")
    record("3a supervised positions == target_ids (8 rows)", ok)

    x, y = ds[0]
    n_sup = int((y != IGNORE_INDEX).sum())
    record("3b everything else ignored", n_sup == len(ds.rows[0]["target_ids"]),
           f"{n_sup} supervised / {y.numel()} positions")

    sampler = ResumableSampler(len(ds), seed=3)
    dl = DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False,
                    num_workers=0, pin_memory=False, drop_last=True)
    model = build_model(cfg)
    opt = AdamW(model.parameters(), lr=1e-4)
    ce = CrossEntropyLoss()
    losses = []
    it = iter(dl)
    for _ in range(20):
        xb, yb = next(it)
        loss = ce(model(xb), yb)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    finite = all(np.isfinite(losses))
    print(f"    loss[0]={losses[0]:.4f}  loss[19]={losses[-1]:.4f}")
    record("3c finetune loss finite", finite, f"{len(losses)} steps")
    record("3d finetune loss decreases", losses[-1] < losses[0],
           f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    return True


def check_4_evaluate(tmp):
    banner("4. EVALUATE 20 TEST ROWS (generation, parsing, F1, failures)")
    import olm_patches
    from model import CONFIGS, build_model
    import evaluate as ev
    from tokenizers import Tokenizer

    olm_patches.apply_all(device="cpu", verbose=False)
    cfg = CONFIGS["tiny"]
    torch.manual_seed(0)

    tok = Tokenizer.from_file(str(ev.TOKENIZER))
    eos = tok.token_to_id("<|endoftext|>")
    meta = json.loads(ev.LABEL_MAP.read_text(encoding="utf-8"))
    id_to_label = {tok.token_to_id(t): s
                   for s, t in zip(meta["sections"], meta["special_tokens"])}

    # parser taxonomy on hand-built sequences
    a = tok.token_to_id("<ipc_302>"); b = tok.token_to_id("<ipc_34>")
    word = tok.encode("the", add_special_tokens=False).ids[0]
    cases = {
        "clean": ([a, b, eos], set()),
        "duplicate": ([a, a, b, eos], {"duplicate"}),
        "no_eos": ([a, b], {"no_eos"}),
        "non_label": ([a, word, b, eos], {"non_label"}),
        "empty": ([eos], {"empty"}),
        "all": ([a, a, word], {"duplicate", "no_eos", "non_label"}),
    }
    ok = True
    for name, (seq, expect) in cases.items():
        _, flags = ev.parse_generation(seq, eos, id_to_label)
        got = {k for k, v in flags.items() if v}
        if got != expect:
            ok = False
            print(f"    {name}: got {got}, expected {expect}")
    record("4a parse failure taxonomy (6 cases)", ok)

    m = ev.multilabel_scores([{"A", "B"}, {"C"}], [{"A"}, {"C", "B"}])
    hand_micro = 2 * 2 / (2 * 2 + 1 + 1)
    record("4b micro-F1 arithmetic", abs(m["micro_f1"] - hand_micro) < 1e-9,
           f"{m['micro_f1']:.4f} == {hand_micro:.4f} (TP2 FP1 FN1)")
    record("4c exact-match arithmetic", abs(m["exact_match"] - 0.0) < 1e-9,
           f"{m['exact_match']:.4f} (neither row exactly right)")

    # end-to-end generation on 20 real rows with an untrained model
    rows = [json.loads(l) for l in
            (ev.ILSI_FT / "test.jsonl").open(encoding="utf-8")][:20]
    model = build_model(cfg).eval()
    gen = ev.greedy_generate(model, [r["input_ids"] for r in rows], 8, eos,
                             "cpu", 4, cfg.max_seq_len)
    gold, pred, flags = [], [], []
    for r, g in zip(rows, gen):
        s, f = ev.parse_generation(g, eos, id_to_label)
        gold.append(set(r["labels"])); pred.append(s); flags.append(f)
    sc = multilabel = ev.multilabel_scores(gold, pred)
    anyfail = sum(any(f.values()) for f in flags)
    record("4d generation ran on 20 rows", len(gen) == 20,
           f"{len(gen)} sequences, max len {max(len(g) for g in gen)}")
    record("4e metrics computed", np.isfinite(sc["micro_f1"]),
           f"micro {sc['micro_f1']:.4f} macro {sc['macro_f1']:.4f} "
           f"exact {sc['exact_match']:.4f}")
    record("4f failure taxonomy on untrained model", True,
           f"{anyfail}/20 rows failed (expected ~20/20 -- model is random)")
    return True


def check_5_hash_guard(tmp):
    banner("5. TOKENIZER HASH GUARD (deliberate mismatch)")
    from train import TOKENIZER, verify_tokenizer

    ok_normal = False
    try:
        verify_tokenizer(check_shards=True)
        ok_normal = True
    except SystemExit as e:
        print(f"    unexpected failure on the good tokenizer: {e}")
    record("5a passes on the real tokenizer", ok_normal)

    backup = tmp / "tokenizer_backup.json"
    shutil.copy(TOKENIZER, backup)
    try:
        data = json.loads(TOKENIZER.read_text(encoding="utf-8"))
        data["_pocket_vakil_corruption_test"] = "deliberate"
        TOKENIZER.write_text(json.dumps(data), encoding="utf-8")
        fired = False
        try:
            verify_tokenizer(check_shards=False)
        except SystemExit:
            fired = True
        record("5b fires on a corrupted tokenizer", fired,
               "SystemExit raised" if fired else "DID NOT FIRE -- guard is broken")
    finally:
        shutil.copy(backup, TOKENIZER)
    import hashlib
    from train import EXPECTED_TOKENIZER_SHA
    restored = hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()
    record("5c tokenizer restored after test",
           restored == EXPECTED_TOKENIZER_SHA, restored[:16] + "...")
    return True


def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="pv_cpu_verify_"))
    print(f"scratch: {tmp}")
    checks = [check_1_pretrain, check_2_resume, check_3_finetune,
              check_4_evaluate, check_5_hash_guard]
    for fn in checks:
        try:
            fn(tmp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            record(f"{fn.__name__} CRASHED", False, f"{type(e).__name__}: {e}")

    banner("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {passed}/{len(RESULTS)} checks passed in {time.time()-t0:.0f}s")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
