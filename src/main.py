"""Entry point: python src/main.py --stage <N>.

Dispatches to the right stage's model class, builds train/val/test loaders,
runs train.py's unified loop, prints metrics, and writes them to
results/stage<N>_metrics.json for later compilation into the ablation table.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SeqRecDataset, eval_collate, load_user_sequences
from models.stage1_bag_of_items import BagOfItemsModel
from models.stage2_attention_no_pos import AttentionNoPositionModel
from train import train

ROOT = Path(__file__).resolve().parents[1]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(stage, n_items):
    if stage == 1:
        return BagOfItemsModel(n_items, hidden_dim=64)
    if stage == 2:
        return AttentionNoPositionModel(n_items, hidden_dim=64, n_blocks=2,
                                        n_heads=1, dropout=0.5)
    raise NotImplementedError(f"stage {stage} not implemented yet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")

    n_items = len(json.loads((ROOT / "data/processed/item_id_to_idx.json").read_text()))
    seqs = load_user_sequences()
    print(f"n_users={len(seqs):,}  n_items={n_items:,}")

    train_ds = SeqRecDataset(seqs, n_items, "train")
    val_ds = SeqRecDataset(seqs, n_items, "val")
    test_ds = SeqRecDataset(seqs, n_items, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=eval_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=eval_collate)

    model = build_model(args.stage, n_items)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}  params={n_params:,}")

    ckpt_dir = ROOT / "saved_models"
    ckpt_dir.mkdir(exist_ok=True)
    config = {
        "lr": 1e-3,
        "n_epochs": args.epochs,
        "val_every": 5,
        "patience": 4,                  # 4 val checks * 5 epochs = 20 epochs plateau
        "device": device,
        "n_items": n_items,
        "checkpoint_path": ckpt_dir / f"stage{args.stage}_best.pth",
    }

    result = train(model, train_loader, val_loader, test_loader, config)

    print("\n=== final ===")
    print(f"best val NDCG@10 (sampled): {result['best_val_ndcg10_sampled']:.4f} "
          f"at epoch {result['best_epoch']}")
    print("test metrics:")
    for k, v in result["test"].items():
        print(f"  {k:<16} {v:.4f}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"stage{args.stage}_metrics.json").write_text(
        json.dumps({
            "stage": args.stage,
            "model": type(model).__name__,
            "n_params": n_params,
            "best_val_ndcg10_sampled": result["best_val_ndcg10_sampled"],
            "best_epoch": result["best_epoch"],
            "test": result["test"],
        }, indent=2)
    )


if __name__ == "__main__":
    main()
