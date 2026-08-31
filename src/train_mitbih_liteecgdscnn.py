import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mitbih_loader import load_all_segments_labels
from dataset_mitbih import MitbihDataset
from models_pytorch import LiteECGDSCNN


# =============================================================================
# Configuration
# =============================================================================

# Fixed project root for your Windows machine
PROJECT_ROOT = r"F:\ecg_thesis"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Training / evaluation helpers
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = outputs.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / total
    avg_acc = correct / total
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    return avg_loss, avg_acc, all_targets, all_preds


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RESULTS_DIR:", RESULTS_DIR)
    print("PLOTS_DIR:", PLOTS_DIR)

    window_size = 200
    pre_samples = 80

    # 1. Load all MIT-BIH beats and AAMI labels
    segments, labels, record_ids = load_all_segments_labels(
        window_size=window_size,
        pre_samples=pre_samples
    )
    print("Segments shape (ALL records):", segments.shape)
    print("Labels shape:", labels.shape)
    print("Unique labels and counts:", np.unique(labels, return_counts=True))

    input_length = segments.shape[1]
    num_classes = int(labels.max()) + 1

    # 2. Train/validation split (use stratify if possible)
    class_counts = np.bincount(labels)
    print("Class counts:", class_counts)

    use_stratify = np.all(class_counts >= 2)

    if use_stratify:
        print("Using stratified train/val split.")
        x_train, x_val, y_train, y_val = train_test_split(
            segments,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    else:
        print("Not enough samples in at least one class; using non-stratified split.")
        x_train, x_val, y_train, y_val = train_test_split(
            segments,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=None
        )

    # 3. Dataset and DataLoader
    train_ds = MitbihDataset(x_train, y_train)
    val_ds = MitbihDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 4. Model, loss, optimizer
    model = LiteECGDSCNN(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    n_params = count_parameters(model)
    print(f"LiteECGDSCNN parameters: {n_params}")

    # 5. Training loop with history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, targets, preds = eval_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[LiteECGDSCNN] Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    end_time = time.time()
    total_train_time = end_time - start_time
    print(f"\n[LiteECGDSCNN] Total training time: {total_train_time:.2f} seconds")

    # 6. Final evaluation
    val_loss, val_acc, targets, preds = eval_one_epoch(model, val_loader, criterion, device)
    print("\n[LiteECGDSCNN] Final validation loss:", val_loss)
    print("[LiteECGDSCNN] Final validation accuracy:", val_acc)

    unique_labels = np.unique(targets)
    all_class_names = ['N', 'S', 'V', 'F', 'Q']
    class_names = [all_class_names[i] for i in unique_labels]

    cm = confusion_matrix(targets, preds, labels=unique_labels)
    print("\n[LiteECGDSCNN] Confusion matrix:\n", cm)

    report = classification_report(targets, preds, output_dict=True)
    print("\n[LiteECGDSCNN] Classification report:\n")
    print(classification_report(targets, preds))

    # 7. Save tables (CSV) – LiteECGDSCNN only
    prefix = "mitbih_all_liteecgdscnn"

    history_df = pd.DataFrame({
        'epoch': np.arange(1, num_epochs + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    history_csv = os.path.join(RESULTS_DIR, f"{prefix}_history.csv")
    history_df.to_csv(history_csv, index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = os.path.join(RESULTS_DIR, f"{prefix}_confusion.csv")
    cm_df.to_csv(cm_csv)

    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(RESULTS_DIR, f"{prefix}_classification_report.csv")
    report_df.to_csv(report_csv)

    summary_df = pd.DataFrame({
        'model': ['LiteECGDSCNN'],
        'dataset': ['MIT-BIH all records'],
        'input_length': [input_length],
        'num_classes': [num_classes],
        'num_params': [n_params],
        'val_loss': [val_loss],
        'val_acc': [val_acc],
        'train_time_sec': [total_train_time]
    })
    summary_csv = os.path.join(RESULTS_DIR, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # 8. Save plots (PNG) – LiteECGDSCNN only[web:675][web:676]
    epochs = np.arange(1, num_epochs + 1)

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label='Train loss')
    ax.plot(epochs, val_losses, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MIT-BIH all – LiteECGDSCNN loss')
    ax.legend()
    fig.tight_layout()
    loss_png = os.path.join(PLOTS_DIR, f"{prefix}_loss.png")
    fig.savefig(loss_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Accuracy curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_accs, label='Train accuracy')
    ax.plot(epochs, val_accs, label='Val accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('MIT-BIH all – LiteECGDSCNN accuracy')
    ax.legend()
    fig.tight_layout()
    acc_png = os.path.join(PLOTS_DIR, f"{prefix}_accuracy.png")
    fig.savefig(acc_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('MIT-BIH all – LiteECGDSCNN confusion matrix')
    fig.tight_layout()
    cm_png = os.path.join(PLOTS_DIR, f"{prefix}_confusion.png")
    fig.savefig(cm_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\n[LiteECGDSCNN] Saved CSV tables in:", RESULTS_DIR)
    print("[LiteECGDSCNN] Saved plots in:", PLOTS_DIR)


if __name__ == "__main__":
    main()