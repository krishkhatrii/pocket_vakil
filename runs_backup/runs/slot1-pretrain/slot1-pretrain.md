# Run: slot1-pretrain

| | |
|---|---|
| Stage | pretrain |
| Date | 2026-09-08 12:12 |
| Preset | prod_300m (1024d/24L/16H) |
| Context | 2048 |
| Precision | bf16 (cuda bf16 supported) |
| OLM | 2.2.0 |
| Patches | autocast->torch.bfloat16, Trainer.load_state, scaler_enabled=False |
| Tokenizer sha256 | b2de5ea09885462611a73045cc98d88f670f0893ce7518bc6e36d1895dde8203 |
| Examples | 976,379 |
| Batch / accum | 12 / 16 |
| LR / warmup / clip | 0.0006 / 500 / 1.0 |
| Checkpoint dir | checkpoints/slot1-pretrain |

## Loss curve

| Step | Tokens seen | Train loss | LR | Wall clock | Checkpoint |
|---|---|---|---|---|---|

## What broke

| What | Symptom | Cause | Fix |
|---|---|---|---|
