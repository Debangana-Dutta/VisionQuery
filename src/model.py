"""
src/model.py
------------
Single place responsible for loading and saving the BLIP VQA model + processor.
Every other file (train.py, evaluate.py, inference.py, app.py) imports from here
so there is exactly one code path for "how do we get the model".

Pretrained component: 100% of the weights loaded by load_model() come from the
Hugging Face Hub checkpoint (Salesforce/blip-vqa-base by default). Nothing here
is trained from scratch.
"""

import os
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise fall back to CPU.

    This is the one CPU-fallback switch used everywhere in the project so
    the code also runs (slowly) on a machine with no GPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str = "Salesforce/blip-vqa-base", checkpoint_dir: str = None):
    """Load the BLIP processor + model.

    Args:
        model_name: HF Hub id of the base pretrained model. Used when no
            fine-tuned checkpoint is available yet.
        checkpoint_dir: path to a local fine-tuned checkpoint (produced by
            src/train.py). If it exists, it is loaded INSTEAD of the base
            model. If it doesn't exist, we fall back to the pretrained base
            model and the caller is responsible for telling the user that
            they're seeing base-model (not fine-tuned) predictions.

    Returns:
        (processor, model, device, used_finetuned: bool)
    """
    device = get_device()

    if checkpoint_dir and os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
        # A previously fine-tuned checkpoint exists locally -> use the ACTUAL
        # fine-tuned weights, not just a claim of fine-tuning.
        processor = BlipProcessor.from_pretrained(checkpoint_dir)
        model = BlipForQuestionAnswering.from_pretrained(checkpoint_dir)
        used_finetuned = True
    else:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        used_finetuned = False

    model.to(device)
    model.eval()
    return processor, model, device, used_finetuned


def save_model(model, processor, output_dir: str):
    """Save a (fine-tuned) model + processor to disk in HF format so that
    load_model() can later load it back via from_pretrained(output_dir).
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[model.py] Saved model + processor to: {output_dir}")
