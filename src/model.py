import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Text Autoencoder (LSTM Seq2Seq)
# =========================
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        out, (h, c) = self.lstm(x)
        return out, h, c


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input_ids: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        x = self.embedding(decoder_input_ids)
        out, (h, c) = self.lstm(x, (h0, c0))
        logits = self.out(out)
        return logits, h, c


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor):
        _, h, c = self.encoder(input_ids)
        decoder_in = target_ids[:, :-1]
        logits, _, _ = self.decoder(decoder_in, h, c)
        return logits


# =========================
# Visual Autoencoder (conv AE for 60x125)
# =========================
class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 60x125 -> 30x63
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 30x63 -> 15x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), # 15x32 -> 8x16
            nn.ReLU(inplace=True),
        )
        self.flatten_dim = 128 * 8 * 16
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.h, self.w = 8, 16
        self.fc = nn.Linear(latent_dim, 128 * self.h * self.w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=(1, 0)), # 8x16 -> 15x31
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                         # 15x31 -> 30x62
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),                          # 30x62 -> 60x124
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        x = self.fc(z)
        x = x.view(-1, 128, self.h, self.w)
        x = self.deconv(x)

        # Fix: always return 60x125 exactly
        _, _, H, W = x.shape

        # height -> 60
        if H > 60:
            excess = H - 60
            top = excess // 2
            bottom = excess - top
            x = x[:, :, top:H-bottom, :]
        elif H < 60:
            pad = 60 - H
            x = F.pad(x, (0, 0, pad // 2, pad - pad // 2))

        # width -> 125
        if W > 125:
            excess = W - 125
            left = excess // 2
            right = excess - left
            x = x[:, :, :, left:W-right]
        elif W < 125:
            pad = 125 - W
            x = F.pad(x, (pad // 2, pad - pad // 2, 0, 0))

        return x


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# =========================
# Sequence Predictor (grounded toy model)
# =========================
class SequencePredictor(nn.Module):
    """
    Your toy grounded predictor:
    - encodes 4 frames (z_v[t]) + 4 descriptions (z_t[t])
    - fuses per-step and passes through GRU
    - decodes next image from h_last
    - decodes next text with teacher forcing from h_last
    Returns z_v/z_t for toy losses (align, ROI re-id, grounding).
    """
    def __init__(self, visual_ae: VisualAutoencoder, text_ae: Seq2SeqLSTM, latent_dim: int = 128):
        super().__init__()
        self.image_encoder = visual_ae.encoder
        self.image_decoder = visual_ae.decoder

        self.text_encoder = text_ae.encoder
        self.text_decoder = text_ae.decoder

        self.latent_dim = latent_dim

        self.fuse = nn.Linear(latent_dim * 2, latent_dim)
        self.temporal = nn.GRU(latent_dim, latent_dim, batch_first=True)

        self.to_img_latent = nn.Linear(latent_dim, latent_dim)
        self.to_text_h0 = nn.Linear(latent_dim, latent_dim)

    def forward(self, seq_imgs: torch.Tensor, seq_desc_ids: torch.Tensor, target_ids: torch.Tensor):
        """
        seq_imgs:     [B,K,3,60,125] (K=4)
        seq_desc_ids: [B,K,T]
        target_ids:   [B,1,T]
        """
        B, K, C, H, W = seq_imgs.shape
        _, _, T = seq_desc_ids.shape

        # encode images
        imgs_flat = seq_imgs.view(B * K, C, H, W)
        z_v_flat = self.image_encoder(imgs_flat)     # [B*K, D]
        z_v = z_v_flat.view(B, K, -1)                # [B,K,D]

        # encode text (use last layer hidden state)
        desc_flat = seq_desc_ids.view(B * K, T)
        _, h_txt, _ = self.text_encoder(desc_flat)
        z_t_flat = h_txt[-1]                         # [B*K,D]
        z_t = z_t_flat.view(B, K, -1)                # [B,K,D]

        # fuse + temporal GRU
        fused = self.fuse(torch.cat([z_v, z_t], dim=-1))  # [B,K,D]
        _, h_last = self.temporal(fused)                  # [1,B,D]
        h_last = h_last[-1]                               # [B,D]

        # next-image prediction
        z_next = self.to_img_latent(h_last)
        pred_img = self.image_decoder(z_next)             # [B,3,60,125]

        # next-text prediction (teacher forcing)
        decoder_in = target_ids[:, 0, :-1]                # [B,T-1]
        h0 = self.to_text_h0(h_last).unsqueeze(0)         # [1,B,D]
        c0 = torch.zeros_like(h0)                         # [1,B,D]
        text_logits, _, _ = self.text_decoder(decoder_in, h0, c0)

        return pred_img, text_logits, z_v, z_t, h_last
