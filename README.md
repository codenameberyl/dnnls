# StoryReasoning — Grounded Toy Storyteller

This repository is a modular, lightweight **toy implementation** inspired by the *StoryReasoning* paper.  
It is **not** a re-implementation of Qwen Storyteller; instead, it demonstrates (at small scale):

- Understanding the dataset fields (frames, GDI story tags, chain-of-thought grounding tables)
- A reasonable grounded architecture using these signals
- Qualitative evaluation + simple “toy” quantitative signals
- Discussion gaps vs the full paper system

---

## Dataset
HuggingFace: [`daniel3303/StoryReasoning`]("https://huggingface.co/datasets/daniel3303/StoryReasoning")

Each sample includes:
- 5+ image frames (`images`)
- frame-level descriptions inside HTML-like GDI tags (`story`)
- chain-of-thought markdown containing character/object tables + bounding boxes (`chain_of_thought`)

---

## Key Idea (Toy Implementation)

The baseline notebook architecture uses only global frame embeddings and global text embeddings, which can cause referential drift (e.g., wrong entity continuity).

This repo adds **toy grounding signals** using the dataset’s `chain_of_thought` markdown tables:

1. **CoT grounding parser**
   - Extracts per-frame **entity IDs** and **bounding boxes** from the chain-of-thought tables.

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

## Results
After training:
- Visual AE reconstructs frames with correct global layout (blurred but coherent).
- Text generation follows dataset style but remains abstract (expected in a small LSTM).
- Sequence predictor produces structurally plausible next images.
- ROI re-ID loss stabilizes embeddings for the same character across frames when CoT IDs exist.

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

## Project Structure

* `src/model.py` : model classes (Text AE, Visual AE, SequencePredictor)
* `src/train.py` : config-driven training pipeline with validation plots
* `src/utils.py` : parsing (GDI, CoT), ROI sampling, checkpoints, plotting
* `config.yaml` : all experimental settings
* `results/` : training_curves, validation_samples
* `models/` : saved checkpoints

All hyperparameters are controlled by `config.yaml`.

Checkpoints are saved to `models/`:

* `text_autoencoder.pth`
* `visual_autoencoder.pth`
* `sequence_predictor_grounded.pth`

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. In Google Colab:
   * Mount Drive
   * Ensure `config.yaml` points `paths.ckpt_dir` to your Drive checkpoint folder.
3. Run:
   * `final_notebook.ipynb` (Run All)
   * It calls `src.train.run("config.yaml")`, which:
      * trains phases enabled in config (or skips if checkpoints exist)
      * saves checkpoints into `paths.ckpt_dir`
      * saves figures into `paths.results_dir`

---

## Author
Abiola Oluwaseun Onasanya
<br>
MSc Artificial Intelligence – Deep Neural Networks & Learning Systems
<br>
Sheffield Hallam University
