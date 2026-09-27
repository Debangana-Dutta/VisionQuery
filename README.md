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

```text
VisionQuery/
│
├── notebooks/
│   ├── 01_baseline_inference.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_finetuning.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── app/
│   ├── app.py
│   └── main.py
│
├── frontend/
│   └── Next.js application
│
├── configs/
│   └── config.yaml
│
├── models/
│   ├── README.md
│   └── visionquery_blip_vqa_3000_lr1e6/
│
├── screenshots/
│   └── README.md      # where to add your own app screenshots
├── requirements.txt
├── EXPERIMENT_LOG.md
├── LICENSE
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Debangana-Dutta/VisionQuery.git
cd VisionQuery
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the FastAPI Backend

From the project root:

```bash
uvicorn app.main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Open Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
POST http://127.0.0.1:8000/predict
```

---

# 🌐 Running the Next.js Frontend

Navigate to the frontend:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

---

# 📱 Streamlit Prototype

The project also includes a Streamlit prototype.

Run:

```bash
streamlit run app/app.py
```

This provides a lightweight interface for testing VQA inference.

---

# 🔁 Reproducibility

Experiments use:

```text
Random Seed = 42
```

The project records:

* Dataset configuration
* Training subset size
* Validation subset size
* Learning rate
* Number of epochs
* Model configuration
* Evaluation results
* Experiment details

This makes the experiments easier to reproduce and compare.

---

# 📈 Evaluation Philosophy

The project focuses on **controlled experimentation rather than claiming benchmark-level improvements**.

The evaluation compares:

```text
Pretrained BLIP
       vs
Fine-Tuned BLIP
       |
       +--> Different Learning Rates
```

This allows the effect of fine-tuning configuration to be observed under the same controlled validation setup.

---

# ⚠️ Limitations

The current implementation has several limitations:

* Training uses a relatively small subset of VQAv2.
* Fine-tuning was limited to one epoch.
* Experiments were performed on a fixed validation subset.
* The reported accuracy is not an official full VQAv2 benchmark score.
* Only a limited learning-rate sweep was performed.
* Generative VQA evaluation can be sensitive to answer normalization.
* CPU inference is slower than GPU inference.
* The current model does not provide calibrated confidence scores.
* The system may struggle with complex reasoning, counting, spatial relationships, or uncommon objects.
* Large model checkpoints are not stored directly in the Git repository.

---

# 🔮 Future Improvements

### Model & Training

* Train on a larger portion of VQAv2.
* Increase the number of training epochs.
* Perform broader hyperparameter sweeps.
* Experiment with learning-rate schedules.
* Explore parameter-efficient fine-tuning such as LoRA/PEFT.
* Compare different BLIP checkpoints.
* Experiment with larger vision-language models.

### Evaluation

* Evaluate on the complete VQAv2 validation set.
* Implement the official VQAv2 evaluation procedure.
* Perform question-category analysis.
* Analyze common failure cases.
* Measure inference latency.
* Add confidence estimation.

### Deployment

* GPU-based inference deployment.
* Model quantization.
* ONNX optimization where appropriate.
* Batch inference support.
* Better API monitoring.
* Experiment tracking and model versioning.

### Frontend

* Improved visual design.
* Answer history.
* Multiple image support.
* Question suggestions.
* Confidence visualization.
* Detailed prediction metadata.
* Better error and loading states.

---

# 🛠️ Tech Stack

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* BLIP
* VQAv2

### Backend

* FastAPI
* Uvicorn
* REST API
* CORS

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS

### Development & Deployment

* Google Colab
* NVIDIA T4
* Hugging Face Hub
* Git
* GitHub

---

# 📚 Notebooks

The project is organized into separate notebooks for reproducibility and experimentation.

### 01 — Baseline Inference

Tests the pretrained BLIP model before fine-tuning.

### 02 — Dataset Preparation

Loads VQAv2 and creates the controlled training and validation subsets.

### 03 — Fine-Tuning

Performs supervised fine-tuning with the selected configuration.

### 04 — Evaluation

Evaluates the pretrained and fine-tuned configurations and records the results.

---

# 📝 Experiment Log

Experiment details are maintained in:

```text
EXPERIMENT_LOG.md
```

The log records the configurations and results of the experiments so that different training runs can be compared systematically.

---

# 🔗 Resources

GitHub Repository:

https://github.com/Debangana-Dutta/VisionQuery

Hugging Face Model:

https://huggingface.co/Debangana-Dutta/visionquery-blip-vqa-3000-lr1e6

VQAv2:

https://visualqa.org/

---

# 📖 Reference & Attribution

This project builds upon established research and open-source technologies in visual question answering and multimodal learning:

- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**
- **VQAv2: Visual Question Answering v2 Dataset**
- **Hugging Face Transformers** for model loading, fine-tuning, and inference
- **PyTorch** for deep learning and model training
- **FastAPI** for backend inference serving
- **Next.js** for the web application frontend

The implementation, fine-tuning experiments, evaluation pipeline, model checkpoint, and application architecture were independently developed for VisionQuery.

---

# 📜 Credits

### BLIP

The project uses the BLIP vision-language architecture and the pretrained:

```text
Salesforce/blip-vqa-base
```

### VQAv2

The project uses the Visual Question Answering v2 dataset for training and evaluation.

---

# 👩‍💻 Author

**Debangana Dutta**

B.Tech (Hons.) Computer Science & Engineering
XIM University, Bhubaneswar

GitHub:

https://github.com/Debangana-Dutta

---

# ⭐ Project Summary

VisionQuery demonstrates a complete **multimodal AI application pipeline**:

```text
VQAv2 Dataset
      ↓
3,000 Training Examples
      ↓
BLIP Pretrained Model
      ↓
Supervised Fine-Tuning
      ↓
Learning-Rate Experiments
      ↓
Model Evaluation
      ↓
Checkpointing
      ↓
Hugging Face
      ↓
FastAPI Backend
      ↓
Next.js Frontend
      ↓
Interactive Visual Question Answering
```

The project brings together **computer vision, natural language processing, multimodal learning, model fine-tuning, experimentation, evaluation, API development, and frontend engineering** into one end-to-end VQA system.
