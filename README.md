# DNNLS Assessment — Grounded Toy Model for StoryReasoning

This repository contains a modular, reproducible implementation of a **toy grounded multimodal sequence predictor** for the **StoryReasoning** dataset.

It demonstrates:
- Understanding of the dataset fields (frames, GDI story, CoT grounding),
- A reasonable architecture that incorporates grounding ideas,
- Qualitative + lightweight quantitative evaluation,
- A clear discussion of limitations vs the full paper system.

---

## Dataset
HuggingFace: [`daniel3303/StoryReasoning`]("https://huggingface.co/datasets/daniel3303/StoryReasoning")

Each sample includes:
- 5+ image frames (`images`)
- frame-level descriptions inside HTML-like GDI tags (`story`)
- chain-of-thought markdown containing character/object tables + bounding boxes (`chain_of_thought`)

This project uses:
- GDI descriptions for frame text
- CoT bounding boxes for **toy ROI grounding** and **re-identification**

---

## Model Design (Toy Architecture)
Phases:
1. **Text Autoencoder** (LSTM seq2seq reconstruction)
2. **Visual Autoencoder** (CNN reconstruction)
3. **Sequence Predictor** (fuse image + text latents per frame, temporal GRU)
4. **Toy Grounding** (alignment + ROI re-ID losses)

---

## Main Innovations (relative to baseline)
1. **Explicit Chain-of-Thought parsing** for grounding signals  
2. **ROI-based entity re-identification loss** from CoT character IDs  
3. **Latent alignment loss** between image and text embeddings  
4. A clean 4-stage experimental pipeline (Text AE → Image AE → Frozen fusion → Grounded training)

---

## Results (Qualitative)
After training:
- Visual AE reconstructs frames with correct global layout (blurred but coherent).
- Sequence predictor produces structurally plausible next images.
- Text generation follows dataset style but remains abstract (expected in a small LSTM).
- ROI re-ID loss stabilizes embeddings for the same character across frames when CoT IDs exist.

Outputs saved under:
- `results/figures/` (loss curves)
- `results/tables/` (training summary CSV)

---

## Limitations vs Paper

The paper’s Qwen Storyteller uses a large pretrained VLM with full detection + tracking.
This repo implements a small educational system:

* CNN autoencoder (not ViT)
* LSTM/GRU (not multimodal transformers)
* Toy ROI grounding (not full detection/tracking)

| This Project | Qwen Storyteller |
|-------------|----------------|
CNN autoencoder | Vision Transformer
LSTM / GRU | Multimodal Transformer
Toy ROI loss | Full object detection + Re-ID
No VLM | Qwen2.5-VL 7B
Small-scale training | Large-scale fine-tuning

This is intentional for feasibility on free Colab and for interpretability.

---

## Conclusion
The implementation successfully demonstrates:
- Understanding of the StoryReasoning dataset
- The role of grounding and temporal modeling
- Why large pretrained VLMs are necessary for high-quality storytelling

It satisfies the assessment goal of **reasoned architectural design and analysis**.

---

## Files

* `src/model.py` : model classes (Text AE, Visual AE, SequencePredictor)
* `src/utils.py` : parsing + datasets + checkpointing + metrics
* `src/train.py` : config-driven training pipeline
* `config.yaml` : all experimental settings
* `results/` : curves + tables
* `models/` : saved checkpoints

All hyperparameters are controlled by `config.yaml`.

Checkpoints are saved to `models/`:

* `text_autoencoder.pth`
* `visual_autoencoder.pth`
* `sequence_predictor_grounded.pth`

---

## How to Run
From inside `dnnls/`:

```bash
pip install -r requirements.txt
python -m src.train
```

---

## Author
Abiola Oluwaseun Onasanya
<br>
MSc Artificial Intelligence – Deep Neural Networks & Learning Systems
<br>
Sheffield Hallam University
