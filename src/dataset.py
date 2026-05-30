"""Leave-one-out sequential dataset for Stages 1-3.

For each user with chronological items [i_1, ..., i_T]:
    train: shifted-pair next-item learning over the prefix [i_1, ..., i_{T-2}]
           (input = prefix[:-1], pos = prefix[1:])
    val:   predict i_{T-1} given [i_1, ..., i_{T-2}]
    test:  predict i_T     given [i_1, ..., i_{T-1}]

Sequences are LEFT-padded with 0 to MAX_SEQ_LEN; idx 0 is the padding token.
Training negatives are resampled fresh on every __getitem__ call, one per
non-pad position, drawn from items the user has never interacted with.
Val/test also return the user's full seen-item tensor so evaluate.py can
sample its 100 eval negatives and mask seen items out of the full-corpus
ranking — use the provided eval_collate as the DataLoader's collate_fn.

Usage smoke test: python3 src/dataset.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MAX_SEQ_LEN = 50
DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_user_sequences(path=DATA / "interactions.parquet"):
    """{user_idx: [item_idx, ...]} in chronological order."""
    # preprocess.py already sorted by (user_idx, timestamp, asin).
    df = pd.read_parquet(path)
    return df.groupby("user_idx", sort=True)["item_idx"].apply(list).to_dict()


def _left_pad(seq, max_len, pad=0):
    seq = seq[-max_len:]
    return [pad] * (max_len - len(seq)) + list(seq)


def eval_collate(batch):
    """Default collate doesn't handle the variable-length seen tensor; this does."""
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    seen = [b[2] for b in batch]
    return inputs, targets, seen


class SeqRecDataset(Dataset):
    def __init__(self, user_sequences, n_items, split):
        assert split in ("train", "val", "test")
        self.split = split
        self.n_items = n_items

        inputs, targets, seen = [], [], []
        for _, seq in user_sequences.items():
            # 5-core guarantees len(seq) >= 5, so train/val/test always exist.
            if split == "train":
                prefix = seq[:-2]
                inputs.append(_left_pad(prefix[:-1], MAX_SEQ_LEN))
                targets.append(_left_pad(prefix[1:], MAX_SEQ_LEN))
                seen.append(set(seq))
            elif split == "val":
                inputs.append(_left_pad(seq[:-2], MAX_SEQ_LEN))
                targets.append(seq[-2])
                seen.append(torch.tensor(list(set(seq)), dtype=torch.long))
            else:
                inputs.append(_left_pad(seq[:-1], MAX_SEQ_LEN))
                targets.append(seq[-1])
                seen.append(torch.tensor(list(set(seq)), dtype=torch.long))

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        # Train: list of sets (used for fast `in` lookup during negative sampling)
        # Val/test: list of 1D long tensors (used by evaluate.py for masking)
        self.seen = seen

    def __len__(self):
        return len(self.inputs)

    def _sample_negatives(self, pos_np, seen):
        """One unseen item per non-pad position; zeros at pad positions."""
        L = len(pos_np)
        neg = np.random.randint(1, self.n_items + 1, size=L)
        bad = np.array([n in seen for n in neg])
        while bad.any():
            neg[bad] = np.random.randint(1, self.n_items + 1, size=bad.sum())
            bad = np.array([n in seen for n in neg])
        neg[pos_np == 0] = 0  # keep padding columns at 0
        return neg

    def __getitem__(self, idx):
        if self.split == "train":
            inp = self.inputs[idx]
            pos = self.targets[idx]
            neg = self._sample_negatives(pos.numpy(), self.seen[idx])
            return inp, pos, torch.from_numpy(neg).long()
        return self.inputs[idx], self.targets[idx], self.seen[idx]


# --------------------------- smoke test ------------------------------------

def _smoke_test():
    seqs = load_user_sequences()
    n_items = len(json.loads((DATA / "item_id_to_idx.json").read_text()))
    print(f"loaded {len(seqs):,} users, n_items={n_items:,}")

    for split in ("train", "val", "test"):
        ds = SeqRecDataset(seqs, n_items, split=split)
        if split == "train":
            loader = DataLoader(ds, batch_size=128, shuffle=True)
            inp, pos, neg = next(iter(loader))
            assert inp.shape == pos.shape == neg.shape == (128, MAX_SEQ_LEN)
            assert (inp[:, -1] != 0).all(), "last input position should be real"
            assert ((pos == 0) == (neg == 0)).all(), "neg pad mask must match pos"
            assert (neg[pos != 0] >= 1).all() and (neg[pos != 0] <= n_items).all()
            assert (neg[pos != 0] != pos[pos != 0]).all()
            print(f"  train  len={len(ds):,}  shapes={[tuple(t.shape) for t in (inp,pos,neg)]}")
            print(f"    sample 0: input[-5:]={inp[0, -5:].tolist()}  "
                  f"pos[-5:]={pos[0, -5:].tolist()}  neg[-5:]={neg[0, -5:].tolist()}")
        else:
            loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=eval_collate)
            inp, tgt, seen = next(iter(loader))
            assert inp.shape == (128, MAX_SEQ_LEN)
            assert tgt.shape == (128,)
            assert (tgt >= 1).all() and (tgt <= n_items).all()
            assert len(seen) == 128 and all(s.ndim == 1 for s in seen)
            print(f"  {split:<5}  len={len(ds):,}  shapes=[{tuple(inp.shape)}, {tuple(tgt.shape)}, "
                  f"list[{len(seen)}] of seen tensors (lens {min(s.numel() for s in seen)}-"
                  f"{max(s.numel() for s in seen)})]")


if __name__ == "__main__":
    _smoke_test()
