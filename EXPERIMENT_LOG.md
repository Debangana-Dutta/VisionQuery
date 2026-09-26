# Experiment Log

Record every real training/evaluation run here. Do not fill in a field until
you have actually measured it — use `[TO BE MEASURED]` until then. This file
is the source of truth for any numbers you quote in the main README or in
interviews.

---

## Run #1

| Field | Value |
|---|---|
| Date | [TO BE MEASURED] |
| Model | Salesforce/blip-vqa-base |
| Dataset / subset size (train / val) | [TO BE MEASURED] |
| Epochs | [TO BE MEASURED] |
| Batch size | [TO BE MEASURED] |
| Gradient accumulation steps | [TO BE MEASURED] |
| Learning rate | [TO BE MEASURED] |
| Final train loss | [TO BE MEASURED] |
| Final validation loss | [TO BE MEASURED] |
| VQA accuracy (validation subset) | [TO BE MEASURED] |
| GPU used | [TO BE MEASURED] |
| Total training time | [TO BE MEASURED] |
| Observations | [TO BE MEASURED] |

Per-epoch losses (paste from `models/train_log.csv` after training):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | [TO BE MEASURED] | [TO BE MEASURED] |
| 2 | [TO BE MEASURED] | [TO BE MEASURED] |
| 3 | [TO BE MEASURED] | [TO BE MEASURED] |

---

## Run #2 (e.g. after scaling up subset size or tuning hyperparameters)

| Field | Value |
|---|---|
| Date | [TO BE MEASURED] |
| Model | |
| Dataset / subset size (train / val) | |
| Epochs | |
| Batch size | |
| Gradient accumulation steps | |
| Learning rate | |
| Final train loss | |
| Final validation loss | |
| VQA accuracy (validation subset) | |
| GPU used | |
| Total training time | |
| Observations | |

---

*Add a new `## Run #N` section for every training run. Never overwrite a
previous run's real numbers — keeping the history is part of what makes this
a credible, reproducible project.*
