"""SDF with controlled removals; defaults preserve the previous decoder math."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .upsampling import stage


@dataclass(frozen=True)
class SDFOptions:
    use_cls_modulation: bool = True
    use_semantic_gate: bool = True
    enhancement: str = "dual_branch"
    upsampling: str = "dysample"
    grid_alignment: str = "h16"
    modulation_init: str = "original"

    def validate(self):
        for name, choices in {"enhancement": {"identity", "dual_branch", "large_only", "dilated_only"},
                              "upsampling": {"bilinear", "dysample"},
                              "grid_alignment": {"h16", "native"},
                              "modulation_init": {"original", "identity"}}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"invalid {name}: {getattr(self, name)}")


class SDFDecoder(nn.Module):
    def __init__(self, in_channels=(384,)*4, channels=256, cls_dim=384, options=None):
        super().__init__()
        self.options = options or SDFOptions()
        self.options.validate()
        if len(in_channels) != 4 or channels < 4 or channels % 4:
            raise ValueError("SDF requires four levels and channels divisible by four")
        self.channels = channels
        self.cls_dim = cls_dim
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1, bias=False) for c in in_channels)
        if self.options.use_cls_modulation:
            self.modulation = nn.Sequential(nn.Linear(cls_dim, 128), nn.GELU(), nn.Linear(128, 6*channels))
            if self.options.modulation_init == "identity":
                nn.init.zeros_(self.modulation[-1].weight)
                nn.init.zeros_(self.modulation[-1].bias)
        if self.options.use_semantic_gate:
            self.gate = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                      nn.GELU(), nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        if self.options.enhancement in {"dual_branch", "large_only"}:
            self.large = nn.Sequential(nn.Conv2d(channels, channels//2, 7, padding=3, groups=channels//2, bias=False),
                                       nn.BatchNorm2d(channels//2), nn.GELU())
        if self.options.enhancement in {"dual_branch", "dilated_only"}:
            self.dilated = nn.Sequential(nn.Conv2d(channels, channels//2, 3, padding=2, dilation=2, bias=False),
                                         nn.BatchNorm2d(channels//2), nn.GELU())
        if self.options.enhancement != "identity":
            fusion_channels = channels if self.options.enhancement == "dual_branch" else channels//2
            self.fusion = nn.Conv2d(fusion_channels, channels, 1)
        self.up1 = stage(channels, self.options.upsampling)
        self.up2 = stage(channels, self.options.upsampling)
        self.refinement = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                        nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, features, cls_token, image_size):
        if len(features) != 4 or len({f.shape[-2:] for f in features}) != 1:
            raise ValueError("expected four same-grid feature maps")
        if min(image_size) < 16:
            raise ValueError("image dimensions must be at least 16")
        projected = [p(x) for p, x in zip(self.projections, features)]
        details = torch.stack(projected[:3], dim=1)
        if self.options.use_cls_modulation:
            gamma, beta = self.modulation(cls_token).chunk(2, dim=1)
            gamma = gamma.reshape(-1, 3, self.channels, 1, 1)
            beta = beta.reshape_as(gamma)
            if self.options.modulation_init == "identity":
                gamma = gamma + 1
            details = details * gamma + beta
        details = details.sum(dim=1)
        if self.options.use_semantic_gate:
            details = details * self.gate(projected[3])
        x = details + projected[3]
        if self.options.enhancement == "dual_branch":
            x = self.fusion(torch.cat([self.large(x), self.dilated(x)], dim=1))
        elif self.options.enhancement == "large_only":
            x = self.fusion(self.large(x))
        elif self.options.enhancement == "dilated_only":
            x = self.fusion(self.dilated(x))
        if self.options.grid_alignment == "h16":
            x = F.interpolate(x, size=tuple(s//16 for s in image_size), mode="bilinear", align_corners=False)
        # Native option also keeps the same two x2 stages, then aligns to H/4.
        x = self.refinement(self.up2(self.up1(x)))
        target = tuple(s//4 for s in image_size)
        if x.shape[-2:] != target:
            x = F.interpolate(x, size=target, mode="bicubic", align_corners=False)
        return x


class SimpleFusionDecoder(nn.Module):
    """Projected sum + the same bilinear refinement stages used in SDF."""
    def __init__(self, in_channels=(384,)*4, channels=256, cls_dim=384):
        super().__init__()
        self.decoder = SDFDecoder(in_channels, channels, cls_dim=cls_dim, options=SDFOptions(
            use_cls_modulation=False, use_semantic_gate=False,
            enhancement="identity", upsampling="bilinear"))

    def forward(self, features, cls_token, image_size):
        return self.decoder(features, cls_token, image_size)
