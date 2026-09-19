#!/usr/bin/env python
"""Replace evaluate.py's greedy_generate with a right-padding implementation.

Why this change:

  The shipped version left-pads with eos_id and passes no attention mask. Under
  causal attention that means real tokens sit AFTER the pad block and therefore
  attend to it, and every row's absolute positions are shifted by however much
  padding its batch happened to need. Measured cost: micro-F1 0.4695 -> 0.2304,
  with a 37.5% parse-failure rate and 25% empty outputs at batch size 16.

  Right-padding is correct without needing a mask: pads sit after the real
  tokens, so causality alone guarantees no real token ever attends to one, and
  the first real token is always at absolute position 0 -- which is exactly the
  geometry fine-tuning trained under (ILSIFinetuneDataset right-pads to 2048).

  A mask is NOT passed here deliberately. OLM's Block.forward filters **kwargs
  per submodule and silently drops unknown ones, so a mask kwarg that fails to
  reach attention would fail invisibly. Right-padding needs no mask to be right.

Two bugs fixed along the way:

  1. Per-row next-token position. Under right-padding, rows have different true
     lengths, so logits[:, -1, :] reads a pad position for every row shorter
     than the batch max. This version indexes logits[b, true_len[b]-1, :].

  2. The x[:, -ctx:] re-slice. Under right-padding it silently drops leading
     REAL tokens from short rows once generation pushes past ctx. Prompts are
     now truncated to (ctx - max_new) so total width can never exceed ctx and
     the re-slice becomes unnecessary. Costs at most `max_new` tokens of
     context on the longest rows; eliminates the bug entirely. This is the
     most likely explanation for the unexplained 4/40 A-vs-C disagreement.

Usage:
    python patch_evaluate.py            # writes evaluate.py, keeps a .bak
    python patch_evaluate.py --dry-run  # print the result, change nothing
"""
import argparse
import re
import shutil
from pathlib import Path

TARGET = Path("src/evaluate.py")

NEW_FUNC = '''def greedy_generate(model, prompts, max_new, eos_id, device, batch_size, ctx):
    """Greedy decode with RIGHT padding, matching fine-tuning geometry.

    Pads sit after the real tokens, so under causal attention no real token
    ever attends to one and the first real token is always at position 0.
    Each row's next-token logits are read at its own true last position.
    """
    out = []
    # Reserve room for generation so total width never exceeds ctx; this
    # removes the need for an x[:, -ctx:] re-slice, which under right padding
    # would drop leading real tokens from short rows.
    prompt_ctx = max(1, ctx - max_new)

    for i in range(0, len(prompts), batch_size):
        chunk = [p[-prompt_ctx:] for p in prompts[i:i + batch_size]]
        n = len(chunk)
        lens = [len(p) for p in chunk]
        width = max(lens)

        X = np.full((n, width + max_new), eos_id, dtype=np.int64)
        for b, p in enumerate(chunk):
            X[b, :len(p)] = p
        x = torch.from_numpy(X).to(device)

        cur = torch.tensor(lens, dtype=torch.long, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        gen = [[] for _ in chunk]
        rows = torch.arange(n, device=device)

        for _ in range(max_new):
            upto = int(cur.max())
            logits = model(x[:, :upto])
            # each row reads its OWN last real position, not the batch's
            nxt = logits[rows, cur - 1, :].float().argmax(-1)

            for b in range(n):
                if not done[b]:
                    t = int(nxt[b])
                    gen[b].append(t)
                    if t == eos_id:
                        done[b] = True

            if bool(done.all()):
                break

            alive = ~done
            x[rows[alive], cur[alive]] = nxt[alive]
            cur = cur + alive.long()

        out.extend(gen)
        if (i // batch_size) % 20 == 0:
            print(f"   {min(i+batch_size, len(prompts)):,}/{len(prompts):,}", flush=True)
    return out
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", default=str(TARGET))
    args = ap.parse_args()

    path = Path(args.path)
    src = path.read_text(encoding="utf-8")

    start = src.find("def greedy_generate(")
    if start == -1:
        raise SystemExit("could not find 'def greedy_generate(' -- aborting")

    # find the next top-level def after it
    m = re.search(r"\n(?=def )", src[start + 1:])
    if not m:
        raise SystemExit("could not find the end of greedy_generate -- aborting")
    end = start + 1 + m.start() + 1

    old = src[start:end]
    print(f"replacing {old.count(chr(10))} lines "
          f"(chars {start}-{end})")

    if "left pad" in old or "left padding" in old:
        print("  confirmed: the existing function is the left-padding version")
    else:
        print("  WARNING: existing function does not mention left padding. "
              "Inspect before trusting this patch.")

    new_src = src[:start] + NEW_FUNC + "\n" + src[end:]

    # also record batch_size in the results JSON if we can find the dict
    if '"batch_size"' not in new_src:
        for anchor in ('"split":', "'split':"):
            idx = new_src.find(anchor)
            if idx != -1:
                line_start = new_src.rfind("\n", 0, idx) + 1
                indent = new_src[line_start:idx]
                ins = f'{indent}"batch_size": args.batch_size,\n'
                new_src = new_src[:line_start] + ins + new_src[line_start:]
                print("  inserted batch_size into the results dict")
                break
        else:
            print("  NOTE: could not locate the results dict. Add "
                  '\'"batch_size": args.batch_size\' by hand -- without it '
                  "eval JSONs are not reproducible from their own metadata.")

    if args.dry_run:
        print("\n--- dry run, nothing written ---\n")
        print(NEW_FUNC)
        return

    backup = path.with_suffix(".py.bak")
    shutil.copy2(path, backup)
    path.write_text(new_src, encoding="utf-8")
    print(f"\nwrote {path} (backup at {backup})")
    print("Verify with:  python -c \"import ast,sys; "
          "ast.parse(open('%s').read()); print('syntax ok')\"" % path)


if __name__ == "__main__":
    main()
