"""
Pocket Vakil — plain-language demo.

Takes a layman description of an incident and returns candidate IPC sections.

Run interactively:      python demo_layman.py
Run the test battery:   python demo_layman.py --test
"""

import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Pocket-Vakil/pocket-vakil-332m-ipc"

PROMPT_TEMPLATE = (
    "{passage}\n\n### TASK: identify the applicable IPC sections\n### ANSWER:\n"
)

# The model was pretrained on court judgments, which almost always open with a
# numbered recital and refer to people as anonymised entities. Wrapping plain
# language in that shell moves the input closer to the training distribution.
RECITAL_PREFIX = "2. Prosecution case in brief was that "

# Common layman words -> the register the corpus actually uses.
# SUBSTITUTIONS = [
#     (r"\bmy husband\b", "her husband <ENTITY>"),
#     (r"\bmy wife\b", "his wife <ENTITY>"),
#     (r"\bmy neighbour\b", "the accused <ENTITY>"),
#     (r"\bmy neighbor\b", "the accused <ENTITY>"),
#     (r"\bmy brother-in-law\b", "her brother-in-law <ENTITY>"),
#     (r"\bmy mother-in-law\b", "her mother-in-law <ENTITY>"),
#     (r"\bmy father-in-law\b", "her father-in-law <ENTITY>"),
#     (r"\bthey\b", "the accused persons"),
#     (r"\bhe\b", "the accused <ENTITY>"),
#     (r"\bstole\b", "committed theft of"),
#     (r"\bbeat me up\b", "caused hurt to the informant"),
#     (r"\bbeat me\b", "caused hurt to the informant"),
#     (r"\bhit me\b", "caused hurt to the informant"),
#     (r"\bkilled\b", "committed murder of"),
#     (r"\bthreatened\b", "criminally intimidated"),
#     (r"\bcheated me\b", "cheated the informant"),
#     (r"\bdowry\b", "demand of dowry"),
#     (r"\bI\b", "the informant"),
#     (r"\bmy\b", "the informant's"),
#     (r"\bme\b", "the informant"),
# ]

SUFFIX = (
    " After receiving knowledge of this incident, the informant <ENTITY> had given a "
    "written report to the police on the basis of which the case crime was registered."
)


# def formalize(plain: str) -> str:
#     """Rewrite a layman description into court-judgment register."""
#     text = plain.strip().rstrip(".")
#     for pattern, replacement in SUBSTITUTIONS:
#         text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
#     return RECITAL_PREFIX + text + "." + SUFFIX


def load():
    print("loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s\n")
    return tokenizer, model


def predict(tokenizer, model, passage: str, max_new_tokens: int = 40, show_raw: bool = True):
    prompt = PROMPT_TEMPLATE.format(passage=passage)
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=2000,
    )
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)

    if show_raw:
        print(f"  raw output ({elapsed:.1f}s): {decoded!r}")

    # Handles lettered sections such as 304B and 498A, which \d+ would drop.
    found = re.findall(r"<ipc_([0-9]+[A-Za-z]*)>", decoded)
    return sorted(set(found), key=lambda s: (int(re.match(r"\d+", s).group()), s))


def compare(tokenizer, model, plain: str):
    """Show raw plain language vs. the formalized version, side by side."""
    print("=" * 70)
    print(f"INPUT: {plain}")
    print("=" * 70)

    print("\n[A] raw plain language, straight into the prompt frame:")
    raw_result = predict(tokenizer, model, plain)
    print(f"  sections: {raw_result if raw_result else '(none)'}")

    # formal = formalize(plain)
    # print(f"\n[B] rewritten into court register:")
    # print(f"  {formal[:150]}...")
    # formal_result = predict(tokenizer, model, formal)
    # print(f"  sections: {formal_result if formal_result else '(none)'}")
    # print()
    return raw_result #, formal_result


TEST_CASES = [
    "My neighbour broke into my house at night and stole my gold jewellery.",
    "My husband and his mother have been beating me and demanding money from my "
    "family since our wedding two years ago.",
    "Three men attacked my brother with knives outside the market and he died "
    "in the hospital.",
    "A man took ten lakh rupees from me saying he would buy me land, but he never "
    "bought anything and stopped answering my calls.",
    "My landlord threatened to kill me when I asked for my deposit back.",
]


def main():
    tokenizer, model = load()

    if "--test" in sys.argv:
        for case in TEST_CASES:
            compare(tokenizer, model, case)
        return

    print("Describe an incident in plain language. Blank line or Ctrl-C to quit.\n")
    while True:
        try:
            plain = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not plain:
            break
        compare(tokenizer, model, plain)


if __name__ == "__main__":
    main()
