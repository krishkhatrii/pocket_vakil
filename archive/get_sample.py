# get_sample.py
import json
from pathlib import Path
from tokenizers import Tokenizer

tok = Tokenizer.from_file("data/tokenizer/bpe_32000_final.json")
rows = [json.loads(l) for l in open("data/ilsi_ft/test.jsonl", encoding="utf-8")]
short = [r for r in rows if len(r["input_ids"]) < 300]
r = short[0]
print("gold labels:", r["labels"])
print(repr(tok.decode(r["input_ids"], skip_special_tokens=False)))