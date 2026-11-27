"""AdaBins-style mViT/range-attention head, adapted to relative z-depth.

Based on the project's historical miniViT. Method: https://github.com/shariqfarooq123/AdaBins
Adaptations: patch size 4, normalized depth bounds, final depth interpolation.
No CLS bin MLP, inverse-depth bins, instance masks or RGB guidance.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AdaBinsHead(nn.Module):
    def __init__(self, channels=256, n_bins=100, rho_min=.001, embedding_dim=128,
                 n_queries=128, patch_size=4, num_layers=4, max_tokens=1024):
        super().__init__()
        if not 0 < rho_min < 1 or n_bins < 2 or embedding_dim % 4:
            raise ValueError("invalid AdaBins dimensions or depth bounds")
        self.rho_min, self.n_queries = rho_min, n_queries
        self.embedding = nn.Conv2d(channels, embedding_dim, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.rand(max_tokens, embedding_dim))
        layer = nn.TransformerEncoderLayer(embedding_dim, 4, dim_feedforward=1024,
                                           batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # Independently initialize cloned encoder layers.
        for block in self.transformer.layers:
            for p in block.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.pixel_embedding = nn.Conv2d(channels, embedding_dim, 3, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256), nn.LeakyReLU(),
                                       nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, n_bins))
        self.classifier = nn.Conv2d(n_queries, n_bins, 1)

    def forward(self, x, output_size):
        tokens = self.embedding(x).flatten(2).transpose(1, 2)
        length = tokens.shape[1]
        if not self.n_queries + 1 <= length <= len(self.position):
            raise ValueError(f"AdaBins token count {length}; expected [{self.n_queries+1}, {len(self.position)}]. Adjust input size/patch size.")
        tokens = self.transformer(tokens + self.position[:length])
        widths = F.relu(self.regressor(tokens[:, 0]).float()) + .1
        widths = widths / widths.sum(dim=1, keepdim=True)
        edges = F.pad((1 - self.rho_min)*widths, (1, 0), value=self.rho_min).cumsum(dim=1)
        queries = tokens[:, 1:self.n_queries+1]
        attention = torch.einsum("bchw,bqc->bqhw", self.pixel_embedding(x), queries)
        probability = self.classifier(attention).float().softmax(dim=1)
        centers = (edges[:, :-1] + edges[:, 1:]) / 2
        depth = (probability * centers[:, :, None, None]).sum(dim=1, keepdim=True)
        depth = F.interpolate(depth, size=output_size, mode="bilinear", align_corners=False)
        return {"prediction": depth, "bin_edges": edges}
