"""Unified training loop for Stages 1-3 (Stage 4 BERT4Rec has its own loop).

BCE-with-logits with one sampled negative per non-pad position (per SASRec
§III-E), Adam(lr=1e-3, betas=(0.9, 0.98)), grad clip max_norm=1.0. Loss is
position-averaged over non-pad positions only.

Validates every `val_every` epochs on sampled NDCG@10 (primary selection
metric per CLAUDE.md). Saves the best checkpoint and early-stops after
`patience` consecutive non-improving val checks. Final report uses the
best checkpoint on the held-out test split.

The eval-time NaN trap (left-pad + causal mask + key_padding_mask -> padding
query softmax over all -inf -> NaN) is handled per-stage inside each model's
forward. Train still asserts loss is finite on the first batch so a regression
fails loud instead of silently.
"""

import time

import torch
import torch.nn.functional as F

from src.evaluate import evaluate


def train(model, train_loader, val_loader, test_loader, config):
    device = config["device"]
    n_items = config["n_items"]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))

    best_ndcg, best_epoch, patience = 0.0, -1, 0
    asserted_finite = False

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        t0 = time.time()
        total, batches = 0.0, 0
        for input_seq, pos_seq, neg_seq in train_loader:
            input_seq = input_seq.to(device)
            pos_seq = pos_seq.to(device)
            neg_seq = neg_seq.to(device)

            seq_out = model(input_seq)
            pos_emb = model.item_embedding(pos_seq)
            neg_emb = model.item_embedding(neg_seq)

            pos_logits = (seq_out * pos_emb).sum(-1)
            neg_logits = (seq_out * neg_emb).sum(-1)

            mask = (pos_seq != 0).float()
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none")
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none")
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum()

            if not asserted_finite:
                assert torch.isfinite(loss), f"non-finite loss on first batch: {loss.item()}"
                asserted_finite = True

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
            batches += 1

        dt = time.time() - t0
        if epoch % config["val_every"] == 0:
            m = evaluate(model, val_loader, n_items, device)
            print(f"epoch {epoch:3d}  loss={total/batches:.4f}  "
                  f"val Hit@10={m['hit10_sampled']:.4f}  "
                  f"val NDCG@10={m['ndcg10_sampled']:.4f}  "
                  f"({dt:.1f}s/epoch)")

            if m["ndcg10_sampled"] > best_ndcg:
                best_ndcg, best_epoch, patience = m["ndcg10_sampled"], epoch, 0
                torch.save(model.state_dict(), config["checkpoint_path"])
            else:
                patience += 1
                if patience >= config["patience"]:
                    print(f"early stop: no val improvement for "
                          f"{patience * config['val_every']} epochs")
                    break
        else:
            print(f"epoch {epoch:3d}  loss={total/batches:.4f}  ({dt:.1f}s/epoch)")

    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    test_m = evaluate(model, test_loader, n_items, device)
    return {"best_val_ndcg10_sampled": best_ndcg, "best_epoch": best_epoch, "test": test_m}
