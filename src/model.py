"""Step 3 - the single, fixed CNN used in every experiment."""
import torch
from torch import nn

from . import config as C


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KeywordCNN(nn.Module):
    """Input (B, 1, 40, 101) log-mel -> logits (B, 10).

    4 conv blocks (3x3 conv, BatchNorm, ReLU, 2x2 max-pool) with 16/32/64/128
    channels, global average pooling, dropout and one fully-connected layer.
    Mel-band normalisation statistics (from the training set) are stored as
    buffers so the saved model is self-contained.
    """

    def __init__(self, n_classes=len(C.WORDS), n_mels=C.N_MELS):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(1, 1, n_mels, 1))
        self.register_buffer("feat_std", torch.ones(1, 1, n_mels, 1))
        self.features = nn.Sequential(
            conv_block(1, 16),    # 40x101 -> 20x50
            conv_block(16, 32),   # -> 10x25
            conv_block(32, 64),   # -> 5x12
            conv_block(64, 128),  # -> 2x6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(C.DROPOUT)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = (x - self.feat_mean) / self.feat_std
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))  # softmax is applied inside the loss
