import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. BaselineCNN
# =============================================================================

class BaselineCNN(nn.Module):
    """
    Baseline 1D CNN for ECG beat classification.

    Design goals:
      - Simple, fully-convolutional feature extractor.
      - Serves as a reference model for all other variants.
      - 3 convolutional blocks with pooling, then 2 FC layers.

    Input:  [batch, 1, L]
    Output: [batch, num_classes]
    """

    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)

        # Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # After 3 pools, effective length is input_length / 8
        reduced_length = input_length // 8
        flattened = reduced_length * 128

        self.fc1 = nn.Linear(flattened, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# 2. ResCNN – residual CNN tailored for 1D ECG
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Basic 1D residual block:
      Conv1d -> BN -> ReLU -> Conv1d -> BN
      + optional 1x1 Conv+BN shortcut
      -> ReLU

    This is a compact residual unit adapted to 1D ECG,
    not a direct copy of any ResNet-18/34 blueprint.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=5, padding=2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)
        return out


class ResCNN(nn.Module):
    """
    Residual 1D CNN for ECG beat classification.

    Architecture:
      - ResidualBlock(1,   32) + MaxPool1d(2)
      - ResidualBlock(32,  64) + MaxPool1d(2)
      - ResidualBlock(64, 128) + MaxPool1d(2)
      - Flatten -> FC(128 * (L/8)) -> FC(num_classes)

    Compared to BaselineCNN, this model:
      - Adds skip connections to ease optimization on noisy ECG.
      - Keeps depth moderate to control parameters and avoid overfitting.
    """

    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.block1 = ResidualBlock(1, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        reduced_length = input_length // 8
        flattened = reduced_length * 128

        self.fc1 = nn.Linear(flattened, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# 3. LiteECGCNN – lightweight CNN for efficiency
# =============================================================================

class LiteECGCNN(nn.Module):
    """
    Lightweight 1D CNN for ECG beat classification.

    Design goals:
      - Fewer channels than BaselineCNN.
      - Use global average pooling (GAP) instead of large FC layers
        to drastically reduce parameter count.
      - Suitable for edge / wearable deployment and faster training.

    Architecture:
      - Conv1d(1->16) + BN + ReLU + MaxPool
      - Conv1d(16->32) + BN + ReLU + MaxPool
      - Conv1d(32->64) + BN + ReLU + MaxPool
      - GlobalAvgPool1d (over time)
      - FC(64 -> num_classes)
    """

    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global average pooling reduces [B, C, T] -> [B, C]
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)           # [B, C, 1]
        x = x.squeeze(-1)         # [B, C]
        x = self.fc(x)            # [B, num_classes]
        return x


# =============================================================================
# 4. LiteECGDSCNN – depthwise separable CNN for maximal efficiency
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for 1D signals:
      - Depthwise: grouped Conv1d (groups=in_channels)
      - Pointwise: 1x1 Conv1d to mix channels

    This reduces parameters and FLOPs significantly,
    a well-known strategy in lightweight CNNs.[web:695][web:696][web:703]
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=1):
        super().__init__()

        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm1d(in_channels)

        # Pointwise: 1x1 conv to change channel dimension
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn_pw = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_dw(self.depthwise(x)))
        x = F.relu(self.bn_pw(self.pointwise(x)))
        return x


class LiteECGDSCNN(nn.Module):
    """
    Lightweight depthwise-separable CNN for ECG beat classification.

    Design goals:
      - Use DepthwiseSeparableConv1d blocks to drastically reduce
        parameter count and computation while keeping accuracy.
      - Explicitly designed for low-power / wearable ECG devices.

    Architecture:
      - Initial Conv1d(1->16)
      - DSConv(16->32) + MaxPool
      - DSConv(32->64) + MaxPool
      - DSConv(64->96) + MaxPool
      - GlobalAvgPool1d
      - FC(96 -> num_classes)
    """

    def __init__(self, input_length: int, num_classes: int):
        super().__init__()

        self.init_conv = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.init_bn = nn.BatchNorm1d(16)

        self.ds1 = DepthwiseSeparableConv1d(16, 32, kernel_size=5, padding=2)
        self.ds2 = DepthwiseSeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.ds3 = DepthwiseSeparableConv1d(64, 96, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = F.relu(self.init_bn(self.init_conv(x)))

        x = self.pool(self.ds1(x))
        x = self.pool(self.ds2(x))
        x = self.pool(self.ds3(x))

        x = self.gap(x)        # [B, 96, 1]
        x = x.squeeze(-1)      # [B, 96]
        x = self.fc(x)
        return x


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    batch_size = 4
    seq_len = 200
    num_classes = 5

    x = torch.randn(batch_size, 1, seq_len)

    for Model in [BaselineCNN, ResCNN, LiteECGCNN, LiteECGDSCNN]:
        model = Model(input_length=seq_len, num_classes=num_classes)
        y = model(x)
        print(Model.__name__, "output shape:", y.shape)