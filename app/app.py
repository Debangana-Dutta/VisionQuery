"""
app/app.py
----------
Streamlit application for VisionQuery.

Loads the ACTUAL fine-tuned checkpoint from configs/config.yaml's
`training.output_dir` if it exists (i.e. only after you've run
notebooks/03_finetuning.ipynb or src/train.py). If no checkpoint exists yet,
it transparently falls back to the base pretrained model and tells the user
so in the UI - it never silently pretends to be fine-tuned.

Run with:  streamlit run app/app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

# Allow running `streamlit run app/app.py` from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_model
from src.inference import answer_question
from src.train import load_config


st.set_page_config(page_title="VisionQuery — Visual Question Answering", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", cfg["training"]["output_dir"])
    processor, model, device, used_finetuned = load_model(
        model_name=cfg["model"]["name"], checkpoint_dir=checkpoint_dir
    )
    return processor, model, device, used_finetuned, cfg


def main():
    st.title("🖼️ VisionQuery")
    st.caption("Visual Question Answering — upload an image, ask a question about it.")

    with st.spinner("Loading model... (first load can take a minute)"):
        try:
            processor, model, device, used_finetuned, cfg = get_model_bundle()
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            st.stop()

    if used_finetuned:
        st.success("Using your fine-tuned checkpoint.", icon="✅")
    else:
        st.warning(
            "No fine-tuned checkpoint found — using the base pretrained "
            "Salesforce/blip-vqa-base model. Run the fine-tuning notebook "
            "to use your own trained checkpoint here.",
            icon="⚠️",
        )

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    question = st.text_input("Ask a question about the image", placeholder="e.g. What color is the car?")

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not read that image file: {e}")

    ask_clicked = st.button("Ask Question", type="primary", disabled=(image is None or not question.strip()))

    if ask_clicked:
        if image is None:
            st.error("Please upload an image first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(
                        image, question, processor, model, device,
                        max_answer_length=cfg["model"]["max_answer_length"],
                    )
                    st.markdown("### Predicted answer")
                    st.markdown(f"> {answer}")
                except Exception as e:
                    st.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
