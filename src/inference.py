"""
src/inference.py
-----------------
One function that turns (image, question) into a predicted answer string.
Used by BOTH src/evaluate.py (batch evaluation) and app/app.py (the Streamlit
UI), so there's a single, tested code path for "ask the model a question".
"""

from PIL import Image
import torch


def answer_question(image: Image.Image, question: str, processor, model, device,
                     max_answer_length: int = 8) -> str:
    """Run BLIP VQA inference on a single (image, question) pair.

    Args:
        image: a PIL Image (RGB). Callers should convert before calling this.
        question: natural-language question string.
        processor: BlipProcessor (from src/model.py load_model()).
        model: BlipForQuestionAnswering (from src/model.py load_model()).
        device: torch.device the model lives on.
        max_answer_length: generation cap; VQA answers are almost always short.

    Returns:
        The decoded answer string.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        # generate() runs the decoder autoregressively conditioned on the
        # image + question encodings - this is standard seq2seq generation,
        # nothing custom.
        output_ids = model.generate(**inputs, max_new_tokens=max_answer_length)

    answer = processor.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip()
