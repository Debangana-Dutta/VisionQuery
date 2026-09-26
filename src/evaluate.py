"""
src/evaluate.py
----------------
Loads a checkpoint (fine-tuned, or falls back to the base pretrained model if
none exists yet) and measures it on a VQAv2 validation subset using the
OFFICIAL VQA accuracy metric:

    acc(answer) = min( (# humans who gave this exact answer) / 3, 1 )

averaged over all questions. This is the standard metric used by the VQA
challenge (Antol et al., 2015 / Goyal et al., 2017) - we do not invent our
own scoring rule.

No number produced here is fabricated: run this script/notebook to get a
real measurement. If you have not run it yet, do not report a number -
use the `[TO BE MEASURED]` placeholder in your README / EXPERIMENT_LOG.
"""

import json
import os
import re
import string

from src.model import load_model
from src.dataset import load_vqa_subset, VQADataset
from src.inference import answer_question
from src.train import load_config


def normalize_answer(ans: str) -> str:
    """Lightly normalize an answer string for comparison (lowercase, strip
    punctuation/articles/extra whitespace) - mirrors the normalization used
    by the official VQA evaluation code.
    """
    ans = ans.lower().strip()
    ans = "".join(ch for ch in ans if ch not in string.punctuation)
    ans = re.sub(r"\b(a|an|the)\b", " ", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


def vqa_accuracy_for_one(pred: str, human_answers: list) -> float:
    """Official per-question VQA accuracy: for the predicted answer, look at
    how many of the (up to 10) human answers match it, divide by 3, cap at 1.
    """
    pred_norm = normalize_answer(pred)
    human_norm = []
    for a in human_answers:
        if isinstance(a, dict):
            human_norm.append(normalize_answer(str(a.get("answer", ""))))
        else:
            human_norm.append(normalize_answer(str(a)))

    matches = sum(1 for h in human_norm if h == pred_norm)
    return min(matches / 3.0, 1.0)


def run_evaluation(cfg: dict):
    processor, model, device, used_finetuned = load_model(
        model_name=cfg["model"]["name"],
        checkpoint_dir=cfg["evaluation"]["checkpoint_dir"],
    )
    print(f"[evaluate.py] Using {'FINE-TUNED' if used_finetuned else 'BASE PRETRAINED (no checkpoint found)'} model.")

    val_split = load_vqa_subset(
        cfg["dataset"]["hf_dataset_name"], "validation",
        cfg["dataset"]["val_subset_size"], cfg["dataset"]["seed"],
    )
    val_ds = VQADataset(val_split, keep_all_answers=True)

    accuracies = []
    qualitative = []

    for i in range(len(val_ds)):
        item = val_ds[i]
        pred = answer_question(
            item["image"], item["question"], processor, model, device,
            max_answer_length=cfg["model"]["max_answer_length"],
        )
        acc = vqa_accuracy_for_one(pred, item["all_answers"])
        accuracies.append(acc)

        if len(qualitative) < cfg["evaluation"]["num_qualitative_examples"]:
            qualitative.append({
                "question": item["question"],
                "predicted_answer": pred,
                "human_answers_sample": item["all_answers"][:3],
                "per_example_accuracy": acc,
            })

    overall_accuracy = sum(accuracies) / max(len(accuracies), 1)
    print(f"[evaluate.py] VQA accuracy on {len(accuracies)} validation examples: {overall_accuracy:.4f}")

    out_dir = cfg["evaluation"]["qualitative_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "num_examples": len(accuracies),
        "vqa_accuracy": overall_accuracy,
        "used_finetuned_checkpoint": used_finetuned,
        "qualitative_examples": qualitative,
    }
    out_path = os.path.join(out_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate.py] Full results written to {out_path}")

    return results


if __name__ == "__main__":
    cfg = load_config()
    run_evaluation(cfg)
