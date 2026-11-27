"""LP, fixed-scope DySample adapted from the project's previous implementation.

Method attribution: https://github.com/tiny-smart/dysample
The progressive decoder arrangement follows AnyDepth/SDT.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DySample(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by sampling groups")
        self.groups = groups
        self.offset = nn.Conv2d(channels, 8 * groups, 1)
        nn.init.normal_(self.offset.weight, std=.001)
        nn.init.zeros_(self.offset.bias)
        axis = torch.tensor([-.25, .25])
        pos = torch.stack(torch.meshgrid(axis, axis, indexing="ij")).transpose(1, 2)
        self.register_buffer("init_pos", pos.repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        offset = (self.offset(x) * .25 + self.init_pos).reshape(b, 2, -1, h, w)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype) + .5,
                                torch.arange(w, device=x.device, dtype=x.dtype) + .5, indexing="ij")
        coords = torch.stack((xx, yy))[None, :, None]
        normalizer = x.new_tensor([w, h]).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), 2)
        coords = coords.reshape(b, 2, self.groups, 2*h, 2*w).permute(0, 2, 3, 4, 1).flatten(0, 1)
        return F.grid_sample(x.reshape(b*self.groups, c//self.groups, h, w), coords,
                             mode="bilinear", padding_mode="border", align_corners=False).reshape(b, c, 2*h, 2*w)


def stage(channels, sampling):
    up = DySample(channels) if sampling == "dysample" else nn.Upsample(
        scale_factor=2, mode="bilinear", align_corners=False)
    return nn.Sequential(up, nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
