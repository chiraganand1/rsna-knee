"""
RSNA Knee Abnormality Detection — Starter Training Pipeline
Targets: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
         PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
Metric:  Macro-averaged AUC-ROC
"""

import os
import random
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ── Config ─────────────────────────────────────────────────────────────────────

class CFG:
    # Paths — update DATA_DIR for local use; /kaggle/input/... on Kaggle
    DATA_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
    OUTPUT_DIR = Path("/kaggle/working")

    TARGETS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    NUM_CLASSES = len(TARGETS)  # 12

    # Slices uniformly sampled per series
    N_SLICES = 12

    # Model
    MODEL_NAME = "efficientnet_b3"
    IMG_SIZE = 224

    # Training
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    FOLDS = 5
    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


# ── DICOM Utilities ────────────────────────────────────────────────────────────

def load_dicom_volume(series_dir: Path) -> np.ndarray:
    """Load all slices in a series; sort by InstanceNumber when available."""
    dcm_files = list(series_dir.glob("*.dcm"))

    def sort_key(f):
        try:
            return int(pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber)
        except Exception:
            return f.name

    dcm_files.sort(key=sort_key)

    slices = []
    for f in dcm_files:
        dcm = pydicom.dcmread(str(f))
        arr = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1))
        intercept = float(getattr(dcm, "RescaleIntercept", 0))
        slices.append(arr * slope + intercept)

    return np.stack(slices, axis=0)  # (D, H, W)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Clip to 1st/99th percentile then scale to [0, 1]."""
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-6)
    return volume.astype(np.float32)


def sample_slices(volume: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample n slices along the depth axis."""
    indices = np.linspace(0, volume.shape[0] - 1, n, dtype=int)
    return volume[indices]  # (n, H, W)


def pick_best_series(study_uid: str, series_df: pd.DataFrame) -> str:
    """Prefer sagittal fluid-sensitive series; fall back to first available."""
    rows = series_df[series_df["StudyInstanceUID"] == study_uid]
    preferred = rows[
        (rows["Anatomical_Plane"] == "Sagittal") & (rows["Fluid_Sensitive"] == 1)
    ]
    chosen = preferred if len(preferred) > 0 else rows
    return chosen.iloc[0]["SeriesInstanceUID"]


# ── Dataset ────────────────────────────────────────────────────────────────────

class RSNADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        split: str = "train",    # "train" | "test"
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.split = split
        self.transform = transform
        self.series_root = CFG.DATA_DIR / f"{split}_series"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        series_uid = pick_best_series(study_uid, self.series_df)

        series_dir = self.series_root / study_uid / series_uid
        volume = load_dicom_volume(series_dir)
        volume = normalize_volume(volume)
        slices = sample_slices(volume, CFG.N_SLICES)  # (N, H, W)

        images = []
        for s in slices:
            img = np.stack([s, s, s], axis=-1)  # gray → 3-channel (H, W, 3)
            if self.transform:
                img = self.transform(image=img)["image"]
            images.append(img)
        images = torch.stack(images, dim=0)  # (N, C, H, W)

        if self.split == "train":
            labels = torch.tensor(
                row[CFG.TARGETS].values.astype(np.float32), dtype=torch.float32
            )
            return images, labels

        return images, study_uid  # return uid for submission building


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


# ── Model ──────────────────────────────────────────────────────────────────────

class RSNAModel(nn.Module):
    """
    2.5D model: encode each slice independently with a CNN backbone,
    then mean-pool slice features before the classification head.
    """

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            CFG.MODEL_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, CFG.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        feats = self.encoder(x.view(B * N, C, H, W))  # (B*N, feat_dim)
        feats = feats.view(B, N, -1).mean(dim=1)       # mean over slices → (B, feat_dim)
        return self.head(feats)                         # (B, num_classes)


# ── Metrics ────────────────────────────────────────────────────────────────────

def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro AUC-ROC; skips any target with only one unique class in the batch."""
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


# ── Train / Validate ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            total_loss += criterion(logits, labels).item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader), macro_auc(labels, preds)


# ── Main ───────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(CFG.SEED)
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(CFG.DATA_DIR / "train.csv")
    series_df = pd.read_csv(CFG.DATA_DIR / "train_series.csv")

    # Keep only rows that have hard labels (not NaN)
    labeled_df = train_df.dropna(subset=CFG.TARGETS).copy()
    for col in CFG.TARGETS:
        labeled_df[col] = labeled_df[col].astype(int)

    print(f"Labeled studies : {len(labeled_df)}")
    print(f"Label prevalence:\n{labeled_df[CFG.TARGETS].mean().round(3).to_string()}\n")

    # Stratified K-Fold on ACL as the representative label
    skf = StratifiedKFold(n_splits=CFG.FOLDS, shuffle=True, random_state=CFG.SEED)
    labeled_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(labeled_df, labeled_df["ACL"])):
        labeled_df.loc[labeled_df.index[val_idx], "fold"] = fold

    device = torch.device(CFG.DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    for fold in range(CFG.FOLDS):
        print(f"\n{'─'*45}")
        print(f"  Fold {fold + 1}/{CFG.FOLDS}")
        print(f"{'─'*45}")

        train_fold = labeled_df[labeled_df["fold"] != fold]
        val_fold = labeled_df[labeled_df["fold"] == fold]

        train_loader = DataLoader(
            RSNADataset(train_fold, series_df, "train", get_transforms(True)),
            batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=CFG.NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            RSNADataset(val_fold, series_df, "train", get_transforms(False)),
            batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
            num_workers=CFG.NUM_WORKERS, pin_memory=True,
        )

        model = RSNAModel().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.EPOCHS
        )
        scaler = torch.amp.GradScaler()

        best_auc = 0.0
        for epoch in range(CFG.EPOCHS):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            flag = "  ←" if val_auc > best_auc else ""
            print(
                f"  Epoch {epoch+1:02d}  "
                f"train_loss={tr_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}{flag}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(
                    model.state_dict(),
                    CFG.OUTPUT_DIR / f"fold{fold}_best.pth",
                )

        print(f"\n  Best AUC fold {fold+1}: {best_auc:.4f}")
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
