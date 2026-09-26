"""
src/dataset.py
---------------
Wraps a small, configurable SUBSET of VQAv2 (loaded via the HuggingFace
`datasets` library) into a PyTorch Dataset that src/train.py and
src/evaluate.py can use directly.

Design choices (see README for the full reasoning):
  - We deliberately take only `train_subset_size` / `val_subset_size` examples
    (from configs/config.yaml) instead of the full ~443K/~214K VQAv2 splits,
    because full VQAv2 training is not realistic on a single Colab T4.
  - Each VQAv2 example has ~10 human answers. For TRAINING we use the single
    most common ("majority") answer as the training target. For EVALUATION
    (src/evaluate.py) we keep ALL human answers, because the official VQA
    accuracy metric needs them.
"""

from collections import Counter
from typing import Optional

from torch.utils.data import Dataset
from datasets import load_dataset


def load_vqa_subset(hf_dataset_name: str, split: str, subset_size: int, seed: int = 42):
    """Download (or use HF's cache of) a VQAv2 split and return a small,
    deterministic, shuffled subset of it.

    Args:
        hf_dataset_name: e.g. "HuggingFaceM4/VQAv2"
        split: "train" or "validation" (dataset-dependent naming)
        subset_size: how many examples to keep
        seed: for reproducible shuffling
    """
    full_split = load_dataset(hf_dataset_name, split=split)
    shuffled = full_split.shuffle(seed=seed)
    n = min(subset_size, len(shuffled))
    return shuffled.select(range(n))


def majority_answer(answers) -> str:
    """Given a list of human answer strings (or dicts with an 'answer' key,
    depending on the dataset schema), return the most frequent one.
    """
    cleaned = []
    for a in answers:
        if isinstance(a, dict):
            cleaned.append(str(a.get("answer", "")).strip().lower())
        else:
            cleaned.append(str(a).strip().lower())
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


class VQADataset(Dataset):
    """PyTorch Dataset over a VQAv2 subset.

    __getitem__ returns a dict with the raw PIL image, the question string,
    the training-target answer (majority vote), and (for eval use) the full
    list of human answers.
    """

    def __init__(self, hf_split, keep_all_answers: bool = False):
        self.data = hf_split
        self.keep_all_answers = keep_all_answers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row["image"]  # PIL.Image, provided directly by this HF dataset
        question = row["question"]
        raw_answers = row.get("answers", [])

        item = {
            "image": image,
            "question": question,
            "answer": majority_answer(raw_answers),
        }
        if self.keep_all_answers:
            item["all_answers"] = raw_answers
        return item


def collate_fn(batch, processor, max_question_length: int, max_answer_length: int):
    """Turn a list of dataset items into model-ready tensors.

    Kept as a standalone function (rather than a method) so it can be passed
    to a DataLoader with functools.partial, and so it's easy to unit-test.
    """
    images = [b["image"].convert("RGB") for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]

    inputs = processor(
        images=images,
        text=questions,
        padding="max_length",
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt",
    )

    labels = processor(
        text=answers,
        padding="max_length",
        truncation=True,
        max_length=max_answer_length,
        return_tensors="pt",
    ).input_ids

    inputs["labels"] = labels
    return inputs
