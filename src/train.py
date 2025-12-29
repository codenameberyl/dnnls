import os
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

from datasets import load_dataset
from transformers import BertTokenizer

from src.model import (
    EncoderLSTM, DecoderLSTM, Seq2SeqLSTM,
    VisualAutoencoder, SequencePredictor
)
from src.utils import (
    seed_everything, ensure_dir,
    parse_gdi_text, parse_cot_grounding,
    crop_and_resize, pick_reid_pair,
    save_checkpoint, load_checkpoint, ckpt_exists,
    show_image, wrap, save_curve, token_acc_at_1
)
import matplotlib.pyplot as plt


# --------------------------
# Datasets
# --------------------------
class TextTaskDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        attrs = parse_gdi_text(self.dataset[idx]["story"])
        f = np.random.randint(0, min(5, len(attrs)))
        return attrs[f]["description"]


class AutoEncoderTaskDataset(Dataset):
    def __init__(self, hf_dataset, image_hw=(60, 125)):
        self.dataset = hf_dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        f = np.random.randint(0, min(5, len(frames)))
        return self.transform(frames[f])


class SequencePredictionDataset(Dataset):
    """
    Returns:
      seq_imgs:   [K,3,60,125]
      seq_desc:   [K,T]
      target_img: [3,60,125]
      target_ids: [1,T]
      roi1, roi2, roi_valid: ROI pair for toy re-id loss
    """
    def __init__(self, hf_dataset, tokenizer, image_hw=(60,125), max_len=120, K=4):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
        ])
        self.image_hw = image_hw

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        frames = sample["images"]

        gdi_frames = parse_gdi_text(sample["story"])
        cot_frames = parse_cot_grounding(sample["chain_of_thought"])

        # inputs
        frame_tensors = []
        desc_tensors = []

        for t in range(self.K):
            img = FT.equalize(frames[t])
            img = self.transform(img)
            frame_tensors.append(img)

            desc = gdi_frames[t]["description"]
            ids = self.tokenizer(
                desc,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).input_ids.squeeze(0)
            desc_tensors.append(ids)

        # targets
        target_img = FT.equalize(frames[self.K])
        target_img = self.transform(target_img)

        target_desc = gdi_frames[self.K]["description"]
        target_ids = self.tokenizer(
            target_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).input_ids  # [1,T]

        # toy ROI re-id pair
        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2 = pair
            try:
                roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                roi_valid = torch.tensor(1, dtype=torch.long)
            except Exception:
                pass

        return (
            torch.stack(frame_tensors),             # [K,3,H,W]
            torch.stack(desc_tensors),              # [K,T]
            target_img,                             # [3,H,W]
            target_ids,                             # [1,T]
            roi1, roi2, roi_valid
        )


# --------------------------
# Qualitative validation plot
# --------------------------
def validation(sequence_predictor: SequencePredictor, data_loader, tokenizer, device, out_path=None):
    sequence_predictor.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        seq_imgs, seq_desc, target_img, target_ids, roi1, roi2, roi_valid = batch

        seq_imgs = seq_imgs.to(device)         # [B,K,3,H,W]
        seq_desc = seq_desc.to(device)         # [B,K,T]
        target_img = target_img.to(device)     # [B,3,H,W]
        target_ids = target_ids.to(device)     # [B,1,T]

        pred_img, text_logits, *_ = sequence_predictor(seq_imgs, seq_desc, target_ids)

        # toy token acc@1 on this batch
        B, Tm1, V = text_logits.shape
        logits_flat = text_logits.reshape(-1, V)
        targets_flat = target_ids[:, 0, 1:].reshape(-1)
        acc1 = token_acc_at_1(logits_flat, targets_flat, ignore_index=tokenizer.pad_token_id)

    fig, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={"height_ratios": [2, 1.5]})

    # 4 inputs
    for i in range(4):
        show_image(ax[0, i], seq_imgs[0, i])
        txt = tokenizer.decode(seq_desc[0, i], skip_special_tokens=True)
        ax[1, i].text(0.5, 0.99, wrap(txt, 40), ha="center", va="top", fontsize=9)
        ax[1, i].axis("off")

    # target
    show_image(ax[0, 4], target_img[0], title="Target")
    target_txt = tokenizer.decode(target_ids[0, 0], skip_special_tokens=True)
    ax[1, 4].text(0.5, 0.99, wrap(target_txt, 40), ha="center", va="top", fontsize=9)
    ax[1, 4].axis("off")

    # predicted
    show_image(ax[0, 5], pred_img[0], title=f"Predicted (acc@1={acc1:.2f})")
    pred_ids = torch.argmax(text_logits[0], dim=-1).detach().cpu().tolist()
    pred_txt = tokenizer.decode(pred_ids, skip_special_tokens=True)
    ax[1, 5].text(0.5, 0.99, wrap(pred_txt, 40), ha="center", va="top", fontsize=9)
    ax[1, 5].axis("off")

    plt.tight_layout()
    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=150)
    plt.show()


# --------------------------
# Training loops
# --------------------------
def train_text_ae(cfg, train_dataset, tokenizer, device, ckpt_dir):
    ds = TextTaskDataset(train_dataset)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    enc = EncoderLSTM(tokenizer.vocab_size, cfg["emb_dim"], cfg["latent_dim"],
                      num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    dec = DecoderLSTM(tokenizer.vocab_size, cfg["emb_dim"], cfg["latent_dim"],
                      num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    model = Seq2SeqLSTM(enc, dec).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    losses = []
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for descriptions in dl:
            input_ids = tokenizer(
                list(descriptions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg["max_len"]
            ).input_ids.to(device)

            opt.zero_grad()
            logits = model(input_ids, input_ids)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), input_ids[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(dl)
        losses.append(avg)
        print(f"[Text AE] Epoch {epoch+1}/{cfg['epochs']} Avg loss: {avg:.4f}")

    save_checkpoint(model, opt, cfg["epochs"], torch.tensor(losses[-1]), ckpt_dir, cfg["ckpt_name"])
    return model, losses


def visual_sanity_check(model, dl, device, max_cols=6, title=None, out_path=None):
    model.eval()
    with torch.no_grad():
        imgs = next(iter(dl)).to(device)
        recon = model(imgs)

    cols = min(max_cols, imgs.size(0))
    fig, ax = plt.subplots(2, cols, figsize=(2*cols, 4))

    for i in range(cols):
        show_image(ax[0, i], imgs[i], title=None)
        show_image(ax[1, i], recon[i], title=None)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    if out_path:
        ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=150)
    plt.show()


def train_visual_ae(cfg, train_dataset, device, ckpt_dir, results_dir=None):
    ds = AutoEncoderTaskDataset(train_dataset, image_hw=tuple(cfg["image_hw"]))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

    model = VisualAutoencoder(latent_dim=cfg["latent_dim"]).to(device)
    criterion = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    losses = []

    # where to save sanity-check panels
    sanity_dir = None
    if results_dir is not None:
        sanity_dir = os.path.join(results_dir, "visual_ae_sanity")
        ensure_dir(sanity_dir)

    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0

        for imgs in dl:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(dl)
        losses.append(avg)
        print(f"[Image AE] Epoch {epoch+1}/{cfg['epochs']} Avg loss: {avg:.4f}")

        # ✅ Visual sanity check EVERY epoch (originals top, recon bottom)
        out_path = None
        if sanity_dir is not None:
            out_path = os.path.join(sanity_dir, f"epoch_{epoch+1}.png")

        visual_sanity_check(
            model, dl, device,
            max_cols=6,
            title=f"Visual AE — Epoch {epoch+1}/{cfg['epochs']} (avg L1={avg:.4f})",
            out_path=out_path
        )

    save_checkpoint(model, opt, cfg["epochs"], torch.tensor(losses[-1]), ckpt_dir, cfg["ckpt_name"])
    return model, losses


def train_sequence_predictor(cfg, train_dataset, tokenizer, text_ae, visual_ae, device, ckpt_dir, results_dir):
    sp_ds = SequencePredictionDataset(
        train_dataset, tokenizer,
        image_hw=tuple(cfg["image_hw"]),
        max_len=cfg["max_len"],
        K=cfg["K"]
    )

    train_size = int(cfg["train_split"] * len(sp_ds))
    val_size = len(sp_ds) - train_size
    train_subset, val_subset = random_split(sp_ds, [train_size, val_size])

    train_dl = DataLoader(train_subset, batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=True)

    ensure_dir(os.path.join(results_dir, "validation_samples"))
    ensure_dir(os.path.join(results_dir, "training_curves"))

    model = SequencePredictor(visual_ae, text_ae, latent_dim=cfg["latent_dim"]).to(device)

    criterion_img = nn.L1Loss()
    criterion_txt = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_mse = nn.MSELoss()

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    lam_align = cfg["lambda_align"]
    lam_reid = cfg["lambda_reid"]
    lam_ground = cfg.get("lambda_ground", 0.0)

    losses = []

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0

        for seq_imgs, seq_desc, target_img, target_ids, roi1, roi2, roi_valid in train_dl:
            seq_imgs = seq_imgs.to(device)
            seq_desc = seq_desc.to(device)
            target_img = target_img.to(device)
            target_ids = target_ids.to(device)
            roi1 = roi1.to(device)
            roi2 = roi2.to(device)
            roi_valid = roi_valid.to(device)

            opt.zero_grad()

            pred_img, text_logits, z_v, z_t, _ = model(seq_imgs, seq_desc, target_ids)

            loss_im = criterion_img(pred_img, target_img)

            B, Tm1, V = text_logits.shape
            loss_txt = criterion_txt(text_logits.reshape(-1, V), target_ids[:, 0, 1:].reshape(-1))

            loss_align = criterion_mse(z_v.mean(dim=1), z_t.mean(dim=1))

            # toy ROI re-id
            loss_reid = torch.tensor(0.0, device=device)
            if roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 0:
                    z_r1 = model.image_encoder(roi1[mask])
                    z_r2 = model.image_encoder(roi2[mask])
                    loss_reid = criterion_mse(z_r1, z_r2)

            # toy grounding (ROI1 aligned to first text embedding)
            loss_ground = torch.tensor(0.0, device=device)
            if lam_ground > 0 and roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 0:
                    z_r1 = model.image_encoder(roi1[mask])
                    z_t0 = z_t[mask, 0, :]
                    loss_ground = criterion_mse(z_r1, z_t0)

            loss = loss_im + loss_txt + lam_align * loss_align + lam_reid * loss_reid + lam_ground * loss_ground
            loss.backward()
            opt.step()

            running += loss.item() * seq_imgs.size(0)

        epoch_loss = running / len(train_dl.dataset)
        losses.append(epoch_loss)

        print(
            f"[Seq] Epoch {epoch+1}/{cfg['epochs']} Loss: {epoch_loss:.4f}  "
            f"(im={loss_im.item():.3f}, txt={loss_txt.item():.3f}, "
            f"align={loss_align.item():.3f}, reid={float(loss_reid):.3f})"
        )

        print("Validation sample:")
        validation(
            model, val_dl, tokenizer, device,
            out_path=os.path.join(results_dir, "validation_samples", f"epoch_{epoch+1}.png")
        )

    save_curve(
        losses,
        os.path.join(results_dir, "training_curves", "seq_loss.png"),
        "SequencePredictor (grounded toy model) training loss"
    )

    save_checkpoint(model, opt, cfg["epochs"], torch.tensor(losses[-1]), ckpt_dir, cfg["ckpt_name"])
    return model, losses


# --------------------------
# Main entry used by notebook
# --------------------------
def run(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    results_dir = cfg["paths"]["results_dir"]
    ensure_dir(ckpt_dir)
    ensure_dir(results_dir)

    # dataset (config-driven)
    dataset_name = cfg.get("hf_dataset", "daniel3303/StoryReasoning")
    train_split = cfg.get("hf_split_train", "train")
    train_dataset = load_dataset(dataset_name, split=train_split)
    print("Train sample keys:", train_dataset[0].keys())

    tokenizer = BertTokenizer.from_pretrained(cfg["tokenizer_name"])

    # Phase 1: text AE
    tcfg = cfg["phase1_text_ae"]
    if not tcfg["enabled"]:
        raise ValueError("phase1_text_ae must be enabled")
    if cfg["skip_if_ckpt_exists"] and ckpt_exists(ckpt_dir, tcfg["ckpt_name"]):
        print("[Text AE] Skipping (checkpoint exists)")
        enc = EncoderLSTM(tokenizer.vocab_size, tcfg["emb_dim"], tcfg["latent_dim"],
                          num_layers=tcfg["num_layers"], dropout=tcfg["dropout"]).to(device)
        dec = DecoderLSTM(tokenizer.vocab_size, tcfg["emb_dim"], tcfg["latent_dim"],
                          num_layers=tcfg["num_layers"], dropout=tcfg["dropout"]).to(device)
        text_ae = Seq2SeqLSTM(enc, dec).to(device)
        text_ae = load_checkpoint(text_ae, ckpt_dir, tcfg["ckpt_name"], device)
    else:
        text_ae, _ = train_text_ae(tcfg, train_dataset, tokenizer, device, ckpt_dir)

    # Phase 2: visual AE
    vcfg = cfg["phase2_visual_ae"]
    if not vcfg["enabled"]:
        raise ValueError("phase2_visual_ae must be enabled")
    if cfg["skip_if_ckpt_exists"] and ckpt_exists(ckpt_dir, vcfg["ckpt_name"]):
        print("[Image AE] Skipping (checkpoint exists)")
        visual_ae = VisualAutoencoder(latent_dim=vcfg["latent_dim"]).to(device)
        visual_ae = load_checkpoint(visual_ae, ckpt_dir, vcfg["ckpt_name"], device)
    else:
        visual_ae, _ = train_visual_ae(vcfg, train_dataset, device, ckpt_dir, results_dir)


    # Freeze for speed
    if cfg["freeze_pretrained"]:
        for p in text_ae.parameters():
            p.requires_grad = False
        for p in visual_ae.parameters():
            p.requires_grad = False
        text_ae.eval()
        visual_ae.eval()

    # Phase 3: sequence predictor
    scfg = cfg["phase3_sequence"]
    if scfg["enabled"]:
        if cfg["skip_if_ckpt_exists"] and ckpt_exists(ckpt_dir, scfg["ckpt_name"]):
            print("[Seq] Skipping (checkpoint exists)")
        else:
            _seq, _ = train_sequence_predictor(
                scfg, train_dataset, tokenizer, text_ae, visual_ae, device, ckpt_dir, results_dir
            )

    print("Done.")
    print("Checkpoints:", ckpt_dir)
    print("Results:", results_dir)


if __name__ == "__main__":
    run("config.yaml")
