import torch
from torch.utils.data import Dataset
import numpy as np


class MitbihDataset(Dataset):
    """
    PyTorch Dataset for beat-centered MIT-BIH segments and AAMI labels.
    segments: np.ndarray [N, L]
    labels:   np.ndarray [N]
    """

    def __init__(self, segments: np.ndarray, labels: np.ndarray):
        self.segments = torch.from_numpy(segments).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        x = self.segments[idx]   # [L]
        y = self.labels[idx]
        # Add channel dimension for Conv1d → [1, L]
        x = x.unsqueeze(0)
        return x, y