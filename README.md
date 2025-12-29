# StoryReasoning — Grounded Toy Storyteller

This repository is a modular, lightweight **toy implementation** inspired by the *StoryReasoning* paper.  
It is **not** a re-implementation of Qwen Storyteller; instead, it demonstrates (at small scale):

- Understanding the dataset fields (frames, GDI story tags, chain-of-thought grounding tables)
- A reasonable grounded architecture using these signals
- Qualitative evaluation + simple “toy” quantitative signals
- Discussion gaps vs the full paper system

---

## Dataset

HuggingFace: `daniel3303/StoryReasoning`

Each sample includes:
- 5+ image frames (`images`)
- frame-level descriptions inside HTML-like GDI tags (`story`)
- chain-of-thought markdown containing character/object tables + bounding boxes (`chain_of_thought`)

---

## Key Idea (Toy Implementation)

The baseline notebook architecture uses only global frame embeddings + global text embeddings, which can cause referential drift (e.g., wrong entity continuity).

This repo adds **toy grounding signals** using the dataset’s `chain_of_thought` markdown tables:

1. **CoT grounding parser**
   - Extracts per-frame **entity IDs** and **bounding boxes** from markdown tables in `chain_of_thought`.

2. **Toy cross-frame object re-identification loss**
   - If an entity ID appears in ≥2 frames, we crop two ROIs using the bounding boxes,
     encode both with the visual encoder, and enforce similarity (MSE).
   - This is a lightweight proxy for the paper’s re-ID concept.

3. **Latent alignment loss (image ↔ text)**
   - Encourages the **mean visual latent** of a sequence to align with the **mean text latent**
     from the corresponding descriptions.

4. **Sequence predictor**
   - Encodes 4 frames + 4 descriptions
   - Fuses per-step latents and runs a GRU
   - Decodes:
     - the **next frame** (image reconstruction)
     - the **next description** (teacher-forced text decoding)

---

## Results (Qualitative Summary)

After training:
- Visual AE reconstructs frames with correct global layout (blurred but coherent).
- Text generation follows dataset style but remains abstract (expected in a small LSTM).
- Sequence predictor produces structurally plausible next images.
- ROI re-ID loss stabilizes embeddings for the same character/object across frames when CoT IDs exist.

> Blurry/gray reconstructions are expected because the visual AE is trained at **60×125** with limited capacity (toy setup for free Colab).

---

## Limitations vs Paper

The paper’s Qwen Storyteller uses a large pretrained VLM with full detection + tracking.
This repo implements a small educational system:

- CNN autoencoder (not ViT)
- LSTM/GRU (not multimodal transformers)
- Toy ROI grounding (not full detection/tracking)

| This Project | Qwen Storyteller |
|-------------|------------------|
| CNN autoencoder | Vision Transformer |
| LSTM / GRU | Multimodal Transformer |
| Toy ROI loss | Full object detection + Re-ID |
| No VLM | Qwen2.5-VL 7B |
| Small-scale training | Large-scale fine-tuning |

This is intentional for feasibility on free Colab and interpretability.

---

## Project Structure

- `src/model.py` : model classes (Text AE, Visual AE, SequencePredictor)
- `src/train.py` : config-driven training pipeline with validation plots
- `src/utils.py` : parsing (GDI, CoT), ROI sampling, checkpoints, plotting
- `config.yaml` : all experimental settings (epochs, LR, sizes, loss weights, paths)
- `results/` : saved training curves + validation samples (auto-created)

**All hyperparameters are controlled by `config.yaml`.**

---

## Checkpoints & Outputs

### Where checkpoints go
Checkpoints are saved to the folder defined in `config.yaml`:

```yaml
paths:
  ckpt_dir: /content/gdrive/MyDrive/dnnls/models
````

Typical saved checkpoints:

* `text_autoencoder.pth`
* `visual_autoencoder.pth`
* `sequence_predictor_grounded.pth`

### Where results go

Figures and validation panels are saved to:

```yaml
paths:
  results_dir: /content/gdrive/MyDrive/dnnls/results
```

Example outputs:

* `results/training_curves/*.png`
* `results/validation_samples/epoch_*.png`
* `results/visual_ae_sanity/epoch_*.png` *(if enabled in training)*

---

## How to Run (Google Colab / Drive Workflow)

This repo is designed to run from **Google Colab** with the project stored in **Google Drive**.

### 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/gdrive")
```

### 2. Clone the repository into Drive (first time only)

```python
import os

repo_path = "/content/gdrive/MyDrive/dnnls"

if not os.path.exists(repo_path):
    !git clone https://github.com/codenameberyl/dnnls.git "$repo_path"
else:
    print(f"Repository already exists at {repo_path}.")
    print("Optional: pull latest changes with `!git pull` from inside the repo.")
```

### 3. Change directory into the repo

```python
%cd "/content/gdrive/MyDrive/dnnls"
```

### 4. Add project paths to Python

```python
import sys, os

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
```

### 5. Install dependencies

```python
!pip -q install -r requirements.txt
```

### 6. Quick sanity check (parsers)

```python
import yaml
from datasets import load_dataset
from src.utils import parse_gdi_text, parse_cot_grounding

with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

train_ds = load_dataset(cfg["hf_dataset"], split=cfg["hf_split_train"])
sample = train_ds[0]

gdi = parse_gdi_text(sample["story"])
cot = parse_cot_grounding(sample["chain_of_thought"])

print("GDI frames:", len(gdi))
print("CoT frame indices:", list(cot.keys())[:10])
print("Frame0 characters:", cot.get(0,{}).get("characters", [])[:2])
```

### 7. Run training (all enabled phases)

```python
from src.train import run
run("config.yaml")
```

> The pipeline will skip phases if checkpoints already exist (controlled by `skip_if_ckpt_exists` in `config.yaml`).

---

## Conclusion

This implementation demonstrates:

* Understanding of the StoryReasoning dataset
* The role of grounding and temporal modeling
* Why large pretrained VLMs are necessary for high-quality grounded storytelling

It satisfies the assessment goal of **reasoned architectural design and analysis** under realistic compute constraints.

---

## Author

**Abiola Oluwaseun Onasanya**
<br>
MSc Artificial Intelligence – Deep Neural Networks & Learning Systems
<br>
Sheffield Hallam University

