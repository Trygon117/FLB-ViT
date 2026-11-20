import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + self.shortcut(x))
        return out


class BaselineResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depth: int = 3,
        base_channels: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        assert depth >= 3, "depth must be at least 3 (at least 1 residual block per stage)"

        # CIFAR-10 stem: 3x3 conv at stride 1 preserves 32x32 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        base_count = depth // 3
        rem = depth % 3
        stage_counts = [
            base_count + (1 if i < rem else 0) for i in range(3)
        ]
        stage_channels = [base_channels, base_channels * 2, base_channels * 4]

        stages = []
        curr_ch = base_channels
        for stage_idx, (num_blocks, out_ch) in enumerate(zip(stage_counts, stage_channels)):
            first_stride = 1 if stage_idx == 0 else 2
            stages.append(
                ResidualBlock(curr_ch, out_ch, stride=first_stride, dropout=dropout)
            )
            curr_ch = out_ch
            for _ in range(num_blocks - 1):
                stages.append(
                    ResidualBlock(curr_ch, out_ch, stride=1, dropout=dropout)
                )

        self.blocks = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(curr_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)