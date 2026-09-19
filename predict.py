import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Pocket-Vakil/pocket-vakil-332m-ipc"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()

PROMPT_TEMPLATE = "{passage}\n\n### TASK: identify the applicable IPC sections\n### ANSWER:\n"

def predict_ipc_sections(passage: str, max_new_tokens: int = 40, verbose: bool = True):
    prompt = PROMPT_TEMPLATE.format(passage=passage)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if verbose:
        print("--- raw generated text ---")
        print(repr(decoded))
        print("--------------------------")
    return sorted(set(int(x) for x in re.findall(r"<ipc_(\d+)>", decoded)))

REAL_SAMPLE_PASSAGE = (
    "2.Prosecution case in brief was that <ENTITY>, daughter of informant <ENTITY> (PW-1) "
    "was married to <ENTITY> 5 years ago. After marriage <ENTITY> was treated with cruelty "
    "by her husband <ENTITY>, <ENTITY>-in-law <ENTITY> and mother-in-law <ENTITY> <ENTITY> "
    "demand of dowry. For this reason these three accused had committed murder of <ENTITY> "
    "on 8.5.2010 by inflicting injuries on her body at their house situate in village "
    "Chaubepur, police station- Mardah, district Ghazipur and tried to dispose of her dead "
    "body. After receiving knowledge of this incident, victim's <ENTITY> <ENTITY> (informant) "
    "had given a written report (Ex-Ka-1) to the police on the basis of which case crime no. "
    "689 of 2010 was registered."
)

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else REAL_SAMPLE_PASSAGE
    print("gold labels: Section 34, 300, 302, 304, 304B, 498A")
    print(predict_ipc_sections(text))