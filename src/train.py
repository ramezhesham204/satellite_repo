import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dataset import DEFAULT_DATA, ROOT, get_loaders
from model import build_model

MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl, classes = get_loaders(a.data, a.batch_size)
    model = build_model(a.model, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / f"best_{a.model}.pt"
    (MODELS_DIR / "classes.json").write_text(json.dumps(classes))

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, wait = float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"epoch {epoch:02d}  train {tl:.4f}/{ta:.3f}  val {vl:.4f}/{va:.3f}")
        if vl < best:
            best, wait = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= a.patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    y_true, y_pred = predict(model, test_dl, device)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"confusion_matrix_{a.model}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"curves_{a.model}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
