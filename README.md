# VisionQuery — Visual Question Answering System

VisionQuery is a portfolio project that fine-tunes a pretrained vision-language
model (BLIP) to answer natural-language questions about images, wrapped in a
simple Streamlit application.

> **Attribution note:** the general VQA project workflow was informed by
> reviewing [varadbhogayata/visual-question-answering](https://github.com/varadbhogayata/visual-question-answering)
> as a reference for how a VQA pipeline is typically structured. No code,
> README text, or reported results from that repository are copied or reused
> here — this is an independent implementation, and any numbers reported
> below are this project's own measurements (or explicitly marked as not yet
> measured).

## 1. Architecture

```
Image ──┐
        ├─► BlipProcessor ─► BlipForQuestionAnswering ─► generated answer tokens ─► decoded string
Question┘        (Salesforce/blip-vqa-base, from Hugging Face Hub)
```

- **Vision encoder + text encoder/decoder**: both come from the pretrained
  `Salesforce/blip-vqa-base` checkpoint. BLIP encodes the image with a ViT,
  encodes the question with a text encoder, fuses the two via cross-attention,
  and a text decoder generates the answer autoregressively.
- **What's pretrained**: the entire backbone. We never train a vision or
  language model from scratch.
- **What's actually fine-tuned**: the same model's weights, continued via
  standard supervised fine-tuning (cross-entropy on answer tokens) on a
  configurable subset of VQAv2. This is what `src/train.py` does.

## 2. Project structure

```
VisionQuery/
├── notebooks/
│   ├── 01_baseline_inference.ipynb   # verify the pretrained model works
│   ├── 02_dataset_preparation.ipynb  # download + inspect a VQAv2 subset
│   ├── 03_finetuning.ipynb           # actually fine-tune, save checkpoint
│   └── 04_evaluation.ipynb           # measure real VQA accuracy
├── src/
│   ├── dataset.py     # VQAv2 subset loading + preprocessing
│   ├── model.py       # load/save the BLIP model + processor
│   ├── train.py        # fine-tuning loop
│   ├── evaluate.py    # official VQA accuracy metric + qualitative examples
│   └── inference.py   # single (image, question) -> answer
├── app/
│   └── app.py          # Streamlit application
├── configs/
│   └── config.yaml    # every hyperparameter lives here
├── models/
│   └── README.md      # where the real trained checkpoint goes (not shipped)
├── screenshots/
│   └── README.md      # where to add your own app screenshots
├── requirements.txt
├── EXPERIMENT_LOG.md   # template for recording real training/eval results
├── LICENSE
└── .gitignore
```

## 3. Dataset

We use [VQAv2](https://visualqa.org/) via the Hugging Face `datasets` library
(`HuggingFaceM4/VQAv2`), which provides images, questions, and ~10 human
answers per question.

**We use a small, configurable subset, not the full dataset**, because the
full VQAv2 train split (~443K questions) is not practical to train on
repeatedly within a single Colab T4 session. Subset sizes are set in
`configs/config.yaml`:

```yaml
dataset:
  train_subset_size: 3000
  val_subset_size: 500
```

Raise these numbers if you have more compute / longer sessions available —
everything else in the pipeline (dataloaders, training loop, evaluation)
works unchanged at any subset size.

## 4. Training

`src/train.py` (invoked from `notebooks/03_finetuning.ipynb`) fine-tunes BLIP
with:
- `AdamW` optimizer, linear warmup + decay schedule
- Mixed precision (fp16) to fit comfortably on a T4
- Gradient accumulation to reach a reasonable effective batch size on limited
  GPU memory
- Per-epoch checkpoint saving to `models/visionquery-finetuned/`
- A CSV loss log (`models/train_log.csv`) with real train/val loss per epoch

All of these are configurable in `configs/config.yaml` — nothing is
hard-coded in `train.py` itself.

## 5. Evaluation

`src/evaluate.py` implements the **official VQA accuracy metric**:

```
accuracy(prediction) = min( (# human annotators who gave this exact answer) / 3, 1 )
```

averaged over the validation subset. This requires the multiple human
answers VQAv2 provides per question, which `src/dataset.py` preserves for
evaluation (`keep_all_answers=True`). It also saves a handful of qualitative
examples (question, predicted answer, sample human answers) to
`models/qualitative_examples/eval_results.json`.

**No accuracy number in this repository is fabricated.** Until you run
`notebooks/04_evaluation.ipynb` yourself, treat any accuracy figure as
`[TO BE MEASURED]` — see `EXPERIMENT_LOG.md`.

## 6. Application

`app/app.py` is a Streamlit app: upload an image, type a question, click
"Ask Question", see the predicted answer. It automatically uses your
fine-tuned checkpoint from `models/visionquery-finetuned/` if present;
otherwise it falls back to the base pretrained model and says so clearly in
the UI (it never silently pretends to be using a fine-tuned model).

## 7. Setup

```bash
git clone <your-repo-url>
cd VisionQuery
python -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt
```

### Run baseline inference locally (no training needed)
```bash
python -c "
from src.model import load_model
from src.inference import answer_question
from PIL import Image

processor, model, device, _ = load_model()
image = Image.open('path/to/your/image.jpg')
print(answer_question(image, 'what is in this picture?', processor, model, device))
"
```

### Run the app
```bash
streamlit run app/app.py
```

## 8. Reproducing on Google Colab

1. Upload the `VisionQuery/` folder to Google Drive, or `git clone` your repo
   into a Colab notebook.
2. Open `notebooks/01_baseline_inference.ipynb` in Colab.
3. **Runtime → Change runtime type → T4 GPU.**
4. Run all cells top to bottom. Repeat for notebooks 02, 03, 04 **in that
   order** (02 needs no GPU but caches the dataset; 03 needs the GPU for
   fine-tuning; 04 needs the checkpoint from 03).
5. After 03 finishes, download the checkpoint (see `models/README.md`) if you
   want to run the Streamlit app locally.

## 9. Reproducibility

- Random seed fixed via `configs/config.yaml` (`reproducibility.seed`, default 42).
- Exact `torch`/`transformers`/CUDA versions and GPU model should be recorded
  by running notebook 01's environment-check cell and pasted into
  `configs/config.yaml`'s `reproducibility` section and `EXPERIMENT_LOG.md`.

## 10. Limitations

- The model is fine-tuned on a small subset of VQAv2 (thousands, not
  hundreds of thousands, of examples) due to Colab T4 constraints, so
  accuracy will be meaningfully below published full-dataset VQAv2 numbers.
- BLIP-VQA generates short, extractive-style answers; it is not designed for
  long, explanatory answers.
- No hyperparameter search was performed beyond the defaults in
  `configs/config.yaml` — see `EXPERIMENT_LOG.md` for what was actually tried.
- Evaluation is run on a validation subset, not the full official VQAv2
  validation/test-dev split, so reported accuracy is only directly comparable
  to other runs of this same project, not to leaderboard numbers.

## 11. Future improvements

- Scale `train_subset_size` up (more compute / longer sessions / a paid GPU tier).
- Try partial fine-tuning (freeze the vision encoder, only tune the text
  decoder) to fit larger batch sizes.
- Add a simple hyperparameter sweep (learning rate, epochs) and log each run
  in `EXPERIMENT_LOG.md`.
- Deploy the Streamlit app (e.g. Streamlit Community Cloud or Hugging Face
  Spaces) once a checkpoint is trained and its size is confirmed to fit the
  target platform's limits.
- Add basic answer-type breakdown (yes/no vs. number vs. other) to the
  evaluation report for more informative error analysis.

## 12. Credits

- Model: [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)
  (Li et al., *BLIP: Bootstrapping Language-Image Pre-training*).
- Dataset: [VQAv2](https://visualqa.org/) (Goyal et al., 2017), accessed via
  the `HuggingFaceM4/VQAv2` Hugging Face dataset.
- Workflow reference (not code/results): [varadbhogayata/visual-question-answering](https://github.com/varadbhogayata/visual-question-answering).
