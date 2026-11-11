import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.num_tokens = self.num_patches

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project (B, C, H, W) -> (B, D, grid, grid) -> (B, num_patches, D)
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        return self.norm(tokens)