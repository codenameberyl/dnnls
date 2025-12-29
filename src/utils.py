import os
import re
import random
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


# --------------------------
# Reproducibility helpers
# --------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Filesystem helpers
# --------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# --------------------------
# Checkpointing
# --------------------------
def ckpt_path(ckpt_dir: str, filename: str) -> str:
    return os.path.join(ckpt_dir, filename)


def ckpt_exists(ckpt_dir: str, filename: str) -> bool:
    return os.path.exists(ckpt_path(ckpt_dir, filename))


def save_checkpoint(model, optimizer, epoch: int, loss, ckpt_dir: str, filename: str):
    ensure_dir(ckpt_dir)
    path = ckpt_path(ckpt_dir, filename)
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss),
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved to: {path} (epoch {epoch})")


def load_checkpoint(model, ckpt_dir: str, filename: str, device: torch.device, strict: bool = True):
    path = ckpt_path(ckpt_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    print(f"Loaded checkpoint from: {path}")
    return model


# --------------------------
# Visualization
# --------------------------
def show_image(ax, image_tensor: torch.Tensor, title: Optional[str] = None):
    img = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)


def wrap(text: str, width=40) -> str:
    return textwrap.fill(text, width=width)


def save_curve(values: List[float], out_path: str, title: str, xlabel="Epoch", ylabel="Loss"):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(6, 4))
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


# --------------------------
# Parsing: GDI story XML-like
# --------------------------
def parse_gdi_text(text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(text, "html.parser")
    images = []

    for gdi in soup.find_all("gdi"):
        image_id = None
        if gdi.attrs:
            for attr_name in gdi.attrs.keys():
                if "image" in attr_name.lower():
                    image_id = attr_name.replace("image", "")
                    break

        if not image_id:
            tag_str = str(gdi)
            m = re.search(r"<gdi\s+image(\d+)", tag_str)
            if m:
                image_id = m.group(1)

        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()
        objects = [obj.get_text().strip() for obj in gdi.find_all("gdo")]
        actions = [act.get_text().strip() for act in gdi.find_all("gda")]
        locations = [loc.get_text().strip() for loc in gdi.find_all("gdl")]

        images.append({
            "image_id": image_id,
            "description": content,
            "objects": objects,
            "actions": actions,
            "locations": locations,
            "raw_text": str(gdi),
        })

    return images


# --------------------------
# Parsing: CoT markdown tables
# --------------------------
def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []

    header_line = table_lines[0]
    data_lines = table_lines[2:]
    headers = [h.strip() for h in header_line.strip("|").split("|")]

    rows = []
    for line in data_lines:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        rows.append(dict(zip(headers, cols)))
    return rows


def parse_cot_grounding(chain_of_thought: str) -> Dict[int, Dict[str, Any]]:
    frames: Dict[int, Dict[str, Any]] = {}
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))

    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
        section = chain_of_thought[start:end]

        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(r"###\s*Characters(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid = row.get("Character ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({"id": cid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

        obj_match = re.search(r"###\s*Objects(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid = row.get("Object ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({"id": oid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

    return frames


# --------------------------
# ROI helpers (toy re-id)
# --------------------------
def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_and_resize(pil_img, bbox, out_hw=(60, 125)):
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = transforms.Resize(out_hw)(crop)
    crop = transforms.ToTensor()(crop)
    return crop


def pick_reid_pair(frames_cot: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int, List[int], List[int]]]:
    id_to_dets = {}
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            ent_id = det.get("id")
            bbox = det.get("bbox")
            if ent_id and bbox:
                id_to_dets.setdefault(ent_id, []).append((f_idx, bbox))

    candidates = [ent_id for ent_id, dets in id_to_dets.items() if len(dets) >= 2]
    if not candidates:
        return None

    ent_id = random.choice(candidates)
    dets = id_to_dets[ent_id]
    (f1, b1), (f2, b2) = random.sample(dets, 2)
    return f1, f2, b1, b2


# --------------------------
# Simple toy metric
# --------------------------
def token_acc_at_1(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> float:
    """
    logits: [N,V]
    targets:[N]
    """
    with torch.no_grad():
        mask = targets != ignore_index
        if mask.sum() == 0:
            return 0.0
        pred = logits.argmax(dim=-1)
        correct = (pred[mask] == targets[mask]).float().mean().item()
        return float(correct)
