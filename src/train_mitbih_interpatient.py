import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from patient_split import load_ds1_ds2_segments_labels
from dataset_mitbih import MitbihDataset
from models_pytorch import BaselineCNN, LiteECGCNN, LiteECGDSCNN, ResCNN


# =============================================================================
# Configuration
# =============================================================================

# PROJECT_ROOT = one level above the src/ folder that contains this script,
# regardless of current working directory or OS.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODELS = [
    ("baseline", BaselineCNN),
    ("liteecgcnn", LiteECGCNN),
    ("liteecgdscnn", LiteECGDSCNN),
    ("rescnn", ResCNN),
]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Training / evaluation helpers (same loop as the intra-patient scripts)
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
# Per-model inter-patient run: train on DS1, evaluate on DS2
# =============================================================================

def run_model(name, model_cls, x_train, y_train, x_test, y_test,
               input_length, num_classes, device):
    print(f"\n{'=' * 70}\n[Interpatient] Training {name} on DS1, evaluating on DS2\n{'=' * 70}")

    train_ds = MitbihDataset(x_train, y_train)
    test_ds = MitbihDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = model_cls(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    n_params = count_parameters(model)
    print(f"[{name}] parameters: {n_params}")

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, targets, preds = eval_one_epoch(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"[{name}] Epoch {epoch:02d} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"DS2_loss={val_loss:.4f}, DS2_acc={val_acc:.4f}")

    total_train_time = time.time() - start_time
    print(f"[{name}] Total training time: {total_train_time:.2f} seconds")

    val_loss, val_acc, targets, preds = eval_one_epoch(model, test_loader, criterion, device)
    print(f"[{name}] Final DS2 (inter-patient) loss: {val_loss}")
    print(f"[{name}] Final DS2 (inter-patient) accuracy: {val_acc}")

    unique_labels = np.unique(targets)
    all_class_names = ['N', 'S', 'V', 'F', 'Q']
    class_names = [all_class_names[i] for i in unique_labels]

    cm = confusion_matrix(targets, preds, labels=unique_labels)
    print(f"[{name}] Confusion matrix (DS2):\n", cm)

    report = classification_report(targets, preds, output_dict=True)
    print(f"[{name}] Classification report (DS2):\n", classification_report(targets, preds))

    prefix = f"interpatient_{name}"

    pred_df = pd.DataFrame({
        'index': np.arange(len(targets)),
        'y_true': targets,
        'y_pred': preds
    })
    pred_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_predictions.csv"), index=False)

    torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f"{prefix}.pt"))

    history_df = pd.DataFrame({
        'epoch': np.arange(1, num_epochs + 1),
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    })
    history_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_history.csv"), index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_confusion.csv"))

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_classification_report.csv"))

    summary_df = pd.DataFrame({
        'model': [name],
        'dataset': ['MIT-BIH DS2 (inter-patient)'],
        'input_length': [input_length],
        'num_classes': [num_classes],
        'num_params': [n_params],
        'val_loss': [val_loss],
        'val_acc': [val_acc],
        'train_time_sec': [total_train_time]
    })
    summary_df.to_csv(os.path.join(RESULTS_DIR, f"{prefix}_summary.csv"), index=False)

    epochs = np.arange(1, num_epochs + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_losses, label='Train loss (DS1)')
    ax.plot(epochs, val_losses, label='Test loss (DS2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Inter-patient DS1\u2192DS2 \u2013 {name} loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_accs, label='Train accuracy (DS1)')
    ax.plot(epochs, val_accs, label='Test accuracy (DS2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Inter-patient DS1\u2192DS2 \u2013 {name} accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Inter-patient DS1\u2192DS2 \u2013 {name} confusion matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{prefix}_confusion.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[{name}] Saved inter-patient results/plots/checkpoint.")

    return summary_df


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("PROJECT_ROOT:", PROJECT_ROOT)

    window_size = 200
    pre_samples = 80

    x_train, y_train, train_record_ids, x_test, y_test, test_record_ids = load_ds1_ds2_segments_labels(
        window_size=window_size,
        pre_samples=pre_samples
    )

    input_length = x_train.shape[1]
    num_classes = int(max(y_train.max(), y_test.max())) + 1

    print("\nDS1 (train) shape:", x_train.shape)
    print("DS2 (test) shape:", x_test.shape)
    print("num_classes:", num_classes)

    all_summaries = []
    for name, model_cls in MODELS:
        summary_df = run_model(
            name, model_cls, x_train, y_train, x_test, y_test,
            input_length, num_classes, device
        )
        all_summaries.append(summary_df)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "interpatient_all_summary.csv"), index=False)
    print("\n=== All inter-patient models done ===")
    print(combined)


if __name__ == "__main__":
    main()
