"""
RSNA Knee Abnormality Detection — Inference & Submission Builder
Loads all fold checkpoints, averages predictions, writes submission.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Import shared config and classes from train.py
from train import CFG, RSNADataset, RSNAModel, get_transforms


@torch.no_grad()
def run_inference() -> None:
    device = torch.device(CFG.DEVICE)

    test_df = pd.read_csv(CFG.DATA_DIR / "test.csv")
    test_series_df = pd.read_csv(CFG.DATA_DIR / "test_series.csv")

    test_loader = DataLoader(
        RSNADataset(test_df, test_series_df, split="test", transform=get_transforms(False)),
        batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
    )

    fold_weights = sorted(CFG.OUTPUT_DIR.glob("fold*_best.pth"))
    if not fold_weights:
        raise FileNotFoundError(f"No fold checkpoints found in {CFG.OUTPUT_DIR}")

    print(f"Found {len(fold_weights)} fold checkpoint(s): {[f.name for f in fold_weights]}")

    all_fold_preds: list[np.ndarray] = []
    study_uids: list[str] = []

    for i, weight_path in enumerate(fold_weights):
        print(f"  Running fold {i+1}/{len(fold_weights)} ({weight_path.name}) ...")
        model = RSNAModel().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        fold_preds: list[np.ndarray] = []
        fold_uids: list[str] = []

        for images, uids in test_loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            fold_preds.append(probs)
            fold_uids.extend(uids)

        all_fold_preds.append(np.concatenate(fold_preds))  # (N, 12)
        if i == 0:
            study_uids = fold_uids  # same order every fold

        del model
        torch.cuda.empty_cache()

    avg_preds = np.mean(all_fold_preds, axis=0)  # (N, 12)

    sub = pd.DataFrame(avg_preds, columns=CFG.TARGETS)
    sub.insert(0, "StudyInstanceUID", study_uids)

    out_path = CFG.OUTPUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved submission.csv  shape={sub.shape}  path={out_path}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    run_inference()
