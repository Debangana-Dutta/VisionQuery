# models/

This folder is intentionally empty in the ZIP — trained model weights are too
large to ship as source code and are excluded via `.gitignore`.

## Where the real checkpoint goes

After running `notebooks/03_finetuning.ipynb` (or `python src/train.py`), the
fine-tuned checkpoint is saved to:

```
models/visionquery-finetuned/
├── config.json
├── pytorch_model.bin  (or model.safetensors)
├── preprocessor_config.json
├── tokenizer_config.json
├── vocab.txt
└── ... (other HF processor/tokenizer files)
```

This is exactly what `transformers`' `save_pretrained()` writes, and it's
exactly what `src/model.py`'s `load_model(checkpoint_dir=...)` and
`app/app.py` expect to load back.

## If you trained in Colab and are working locally

1. After training completes in Colab, zip the checkpoint folder:
   ```python
   import shutil
   shutil.make_archive("visionquery-finetuned", "zip", "models/visionquery-finetuned")
   ```
2. Download `visionquery-finetuned.zip` from the Colab file browser.
3. Unzip it locally into `VisionQuery/models/visionquery-finetuned/`.
4. Run `streamlit run app/app.py` — it will detect the checkpoint automatically.

## If no checkpoint is present

Both `src/evaluate.py` and `app/app.py` fall back gracefully to the base
pretrained `Salesforce/blip-vqa-base` model, and clearly report that they are
doing so (see `used_finetuned` / the Streamlit warning banner). Nothing in
this project silently pretends a checkpoint exists when it doesn't.

`models/qualitative_examples/` and `models/train_log.csv` are also generated
here by `src/evaluate.py` / `src/train.py` respectively, and are also
git-ignored for the same reason (they're run artifacts, not source).
