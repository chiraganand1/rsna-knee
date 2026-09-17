"""
RSNA Knee — Upgraded model: DINOv2 backbone + learned attention pooling.

"""

import torch
import torch.nn as nn
import timm


class CFG:

    MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
    IMG_SIZE = 224
    NUM_CLASSES = 12
    ENCODER_CHUNK_SIZE = 16


# ═══════════════════════════════════════════════════════════════════
# TODO: Attention pooling -- the real exercise
# ═══════════════════════════════════════════════════════════════════
class AttentionPool(nn.Module):
    """
    Takes N slice-level feature vectors, learns which ones matter most for
    THIS study, and returns one weighted-combination feature vector.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        slice_features: shape (B, N, feat_dim) -- N slice features per study in the batch.
        Returns: shape (B, feat_dim) -- one pooled feature vector per study.
   
        """
        raw_scores = self.attention(slice_features)  # (B, N, 1)
        scores = raw_scores.squeeze(-1)
        weights = torch.softmax(scores, dim=1)  # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), slice_features).squeeze(1)  # (B, feat_dim)
        return pooled
        raise NotImplementedError("Implement AttentionPool.forward")


class RSNAModelV2(nn.Module):
    """
    RSNA Knee model with DINOv2 backbone and attention-based slice pooling.
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            CFG.MODEL_NAME, pretrained=True, num_classes=0
        )
        feat_dim = self.encoder.num_features

        self.pool = AttentionPool(feat_dim)  # <- your implementation above, used here

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, CFG.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        flat = x.view(B * N, C, H, W)
        feats = torch.cat(
            [self.encoder(chunk) for chunk in flat.split(CFG.ENCODER_CHUNK_SIZE, dim=0)],
            dim=0,
        )
        feats = feats.view(B, N, -1)          # (B, N, feat_dim) -- NOT mean-pooled yet
        pooled = self.pool(feats)               # <- attention pooling replaces .mean(dim=1)
        return self.head(pooled)                # (B, num_classes)


if __name__ == "__main__":
    # Quick sanity test with fake data, no real DICOM files needed --
    # confirms shapes flow correctly through the whole model before you
    # wire this into your real training script.
    model = RSNAModelV2()
    fake_batch = torch.randn(2, 12, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)  # (B=2, N=12 slices, C=3, H, W)
    out = model(fake_batch)
    print(f"Output shape: {out.shape}")  # should be (2, 12) -- 2 studies, 12 target predictions each
    assert out.shape == (2, CFG.NUM_CLASSES), "Shape mismatch -- check AttentionPool implementation"
    print("Shape check passed.")
