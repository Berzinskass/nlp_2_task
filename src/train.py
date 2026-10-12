"""Step 3 - train the CNN on clean training speakers.

The validation speakers (disjoint from train and test) are only used to keep
the checkpoint of the best epoch. No noise or augmentation is applied to the
training data.
"""
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .features import load_clean_features
from .model import KeywordCNN

MODEL_PATH = C.OUT_DIR / "model.pt"


def set_seed(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(X):
    return torch.from_numpy(X).unsqueeze(1)  # add channel dim


@torch.no_grad()
def predict(model, X, batch_size=512):
    model.eval()
    preds = [model(to_tensor(X[i:i + batch_size])).argmax(1).numpy()
             for i in range(0, len(X), batch_size)]
    return np.concatenate(preds)


def main():
    set_seed()
    df, X = load_clean_features()
    y = df["label"].to_numpy()
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()

    model = KeywordCNN()
    model.feat_mean.copy_(torch.from_numpy(X[tr].mean(axis=(0, 2))).view(1, 1, -1, 1))
    model.feat_std.copy_(torch.from_numpy(X[tr].std(axis=(0, 2))).view(1, 1, -1, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(model, f"\ntrainable parameters: {n_params}")

    loader = DataLoader(TensorDataset(to_tensor(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=C.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(C.SEED))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE,
                           weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)

    history, best = [], -1.0
    for epoch in range(1, C.EPOCHS + 1):
        t0 = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        sched.step()
        val_acc = float((predict(model, X[va]) == y[va]).mean())
        rec = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_acc": correct / total, "val_acc": val_acc}
        history.append(rec)
        print(f"epoch {epoch:2d}  loss {rec['train_loss']:.4f}  "
              f"train acc {rec['train_acc']:.4f}  val acc {val_acc:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    summary = {
        "trainable_parameters": n_params, "epochs": C.EPOCHS,
        "batch_size": C.BATCH_SIZE, "optimizer": "Adam",
        "learning_rate": C.LEARNING_RATE, "weight_decay": C.WEIGHT_DECAY,
        "lr_schedule": "cosine annealing", "loss": "cross-entropy",
        "dropout": C.DROPOUT, "best_val_acc": best,
        "best_epoch": max(history, key=lambda r: r["val_acc"])["epoch"],
        "history": history,
    }
    (C.OUT_DIR / "training.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
