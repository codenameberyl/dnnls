# Grounded StoryReasoning: A Toy Multimodal Sequence Predictor

## Overview
This project implements a **toy grounded multimodal sequence prediction model**
for the **StoryReasoning dataset**. The goal is to explore **scene understanding,
entity consistency, and grounded story continuation** using a simplified,
interpretable architecture suitable for an MSc-level deep learning assessment.

The work is **inspired by**, but does **not re-implement**, the full
Qwen Storyteller model described in:
> Daniel et al., *StoryReasoning: Chain-of-Thought for Scene Understanding and Grounded Story Generation* (2025).

---

## Dataset Understanding
Each story consists of:
- **5+ image frames**
- A **GDI-formatted story** with descriptions, objects, actions, and locations
- A **Chain-of-Thought (CoT)** markdown block containing:
  - Per-frame character/object tables
  - Explicit bounding boxes
  - Narrative reasoning

This project explicitly uses:
- Frame-level descriptions (GDI)
- CoT bounding boxes for **toy ROI grounding**
- Story-level temporal structure

---

## Model Design (Toy Architecture)

### Phase 1 – Text Autoencoder
- LSTM encoder–decoder
- Learns a compact latent representation of frame descriptions

### Phase 2 – Visual Autoencoder
- CNN encoder–decoder
- Reconstructs 60×125 RGB frames
- Produces a 128-D visual latent

### Phase 3 – Sequence Predictor
- Encodes 4 frames + descriptions
- Fuses visual + text latents per timestep
- Temporal GRU models narrative progression
- Predicts:
  - Next frame (image)
  - Next description (text)

### Phase 4 – Grounding & Consistency (Toy)
- **Image–Text latent alignment loss**
- **ROI re-identification loss**
  - Uses CoT bounding boxes
  - Encourages identity consistency across frames

---

## Main Innovations (Relative to Baseline)
- Explicit use of **Chain-of-Thought annotations**
- ROI-based **entity re-identification signal**
- Joint visual–text latent alignment
- Fully interpretable, modular architecture

This design demonstrates **how grounding improves narrative coherence**, even in a simplified model.

---

## Results (Qualitative)
- Visual autoencoder reconstructs scene layout and color distribution
- Sequence predictor produces **structurally plausible next frames**
- Text generation reflects scene continuity but remains abstract
- ROI loss stabilises identity representations across frames

> This is expected: the model is intentionally small and trained without
large-scale pretraining.

---

## Limitations vs the Paper
| This Project | Qwen Storyteller |
|-------------|----------------|
CNN autoencoder | Vision Transformer
LSTM / GRU | Multimodal Transformer
Toy ROI loss | Full object detection + Re-ID
No VLM | Qwen2.5-VL 7B
Small-scale training | Large-scale fine-tuning

This project is **conceptual and educational**, not state-of-the-art.

---

## Conclusion
The implementation successfully demonstrates:
- Understanding of the StoryReasoning dataset
- The role of grounding and temporal modeling
- Why large pretrained VLMs are necessary for high-quality storytelling

It satisfies the assessment goal of **reasoned architectural design and analysis**.

---

## How to Run
1. Open `final_notebook.ipynb`
2. Ensure checkpoints exist in `models/`
3. Run cells sequentially for evaluation and visualization

---

## Author
Abiola Oluwaseun Onasanya
<br>
MSc Artificial Intelligence – Deep Neural Networks & Learning Systems
<br>
Sheffield Hallam University
