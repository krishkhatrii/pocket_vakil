# Pocket Vakil — reproduction

## Install
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers

## Load the model
from_pretrained("ORG/pocket-vakil-332m-ipc", trust_remote_code=True)

trust_remote_code is REQUIRED — the architecture is custom and modelling
code ships with the repo. First run may prompt for interactive confirmation.

## Reproduce results
python inference_hf.py --model ORG/pocket-vakil-332m-ipc \
    --data data/ilsi_ft_v2/test.jsonl

Expected: micro-F1 0.4791, macro-F1 0.3585, 0/100 labels with F1 = 0.

REQUIRES THE v2 REFRAMED SPLIT. The 23-token task frame is part of the
model's expected input. Running against the original ILSI split produces
empty output on every row and scores 0.0 with no error raised.

## Single prediction
python inference_hf.py --model ORG/pocket-vakil-332m-ipc \
    --predict "<case facts>"

## Note on evaluate.py
Included as the script the reported numbers were originally produced with.
It imports OLM 2.2.0, supplied inside the bootcamp compute environment and
not publicly installable. Use inference_hf.py instead — the converted model
is numerically identical (max |delta logit| = 0.000e+00).