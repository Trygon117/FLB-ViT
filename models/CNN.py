import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depth: int = 3,
        base_channels: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert depth >= 3, "depth must be at least 3 (at least 1 layer per stage)"

        # Distribute total conv layers across 3 spatial stages (32x32, 16x16, 8x8)
        base_count = depth // 3
        rem = depth % 3
        stage_Counts = [
            base_count + (1 if i < rem else 0) for i in range(3)
        ]
        stage_channels = [base_channels, base_channels * 2, base_channels * 4]

        stages = []
        curr_ch = in_channels
        for stage_idx, (num_layers, out_ch) in enumerate(zip(stage_Counts, stage_channels)):
            if stage_idx > 0:
                stages.append(nn.MaxPool2d(kernel_size=2, stride=2))
            for _ in range(num_layers):
                stages.append(ConvBlock(curr_ch, out_ch, dropout=dropout))
                curr_ch = out_ch

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(curr_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)