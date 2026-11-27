"""Local, explicitly weighted DINOv2 backbone with a shared AdaBins-style head."""
from pathlib import Path
import torch
from torch import nn
from .sdf import SDFDecoder, SimpleFusionDecoder, SDFOptions
from .adabins import AdaBinsHead


class NearDepthV2(nn.Module):
    def __init__(self, backbone, model_config):
        super().__init__()
        model_config.validate()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad = False
        for p in backbone.blocks[-2:].parameters():
            p.requires_grad = True
        for p in backbone.norm.parameters():
            p.requires_grad = True
        if model_config.decoder == "sdf":
            self.decoder = SDFDecoder(channels=model_config.feature_dim, options=model_config.sdf)
        elif model_config.decoder == "simple":
            self.decoder = SimpleFusionDecoder(channels=model_config.feature_dim)
        else:
            raise NotImplementedError("DPT comparison has not been migrated; do not substitute another decoder")
        self.decoder_initialization = None
        if model_config.decoder_init_seed is not None:
            self.decoder_initialization = initialize_decoder_from_canonical(
                self.decoder, model_config.feature_dim, model_config.decoder_init_seed)
        if model_config.common_init_seed is None:
            self.head = AdaBinsHead(model_config.feature_dim, model_config.n_bins, model_config.rho_min)
        else:
            # Build this CPU head under an isolated RNG, unaffected by a decoder's
            # number/order of random draws; restore the caller's RNG afterwards.
            with torch.random.fork_rng(devices=[]):
                # torch.manual_seed also reseeds CUDA; this is a CPU-only scope.
                torch.random.default_generator.manual_seed(model_config.common_init_seed)
                self.head = AdaBinsHead(model_config.feature_dim, model_config.n_bins, model_config.rho_min)
    def forward(self, rgb):
        if rgb.shape[-2] % 14 or rgb.shape[-1] % 14:
            raise ValueError("input dimensions must be divisible by 14")
        outputs = self.backbone.get_intermediate_layers(
            rgb, n=[2, 5, 8, 11], reshape=True, return_class_token=True)
        features = [value[0] for value in outputs]
        fused = self.decoder(features, outputs[-1][1], rgb.shape[-2:])
        return self.head(fused, rgb.shape[-2:])


def module_state_sha256(module):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def initialize_decoder_from_canonical(decoder, channels, seed):
    """Copy retained tensors from one seeded full SDF, without retaining a ghost model.

    Single-branch fusion takes the matching input-channel slice of the full
    fusion kernel. This equals removing the other branch contribution while
    actually removing its parameters/compute, rather than padding zero features.
    """
    target = decoder.decoder if isinstance(decoder, SimpleFusionDecoder) else decoder
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        canonical = SDFDecoder(in_channels=tuple(p.in_channels for p in target.projections),
                               channels=channels, cls_dim=target.cls_dim, options=SDFOptions())
    source = canonical.state_dict()
    updates, origins = {}, {}
    for name, tensor in target.state_dict().items():
        original = source[name]
        rule = "copy"
        if target.options.modulation_init == "identity" and name in {"modulation.2.weight", "modulation.2.bias"}:
            original = torch.zeros_like(tensor)
            rule = "intentional_identity_zero"
        elif name == "fusion.weight" and tensor.shape != original.shape:
            if target.options.enhancement == "large_only":
                original = original[:, :channels//2]
                rule = "input_first_half"
            elif target.options.enhancement == "dilated_only":
                original = original[:, channels//2:]
                rule = "input_second_half"
        if tensor.shape != original.shape:
            raise ValueError(f"unsupported canonical decoder mapping: {name}")
        updates[name] = original.detach().clone()
        origins[name] = rule
    target.load_state_dict(updates, strict=True)
    return {"policy": "canonical_full_sdf_v1", "seed": seed,
            "canonical_state_sha256": module_state_sha256(canonical),
            "initialized_state_sha256": module_state_sha256(target), "tensor_rules": origins,
            "decoder_parameters": sum(p.numel() for p in target.parameters())}


def build_model(model_config, backbone_repo, backbone_weights):
    if model_config.backbone != "dinov2_vits14_reg":
        raise ValueError("currently supports dinov2_vits14_reg only")
    repo, weights = Path(backbone_repo).resolve(), Path(backbone_weights).resolve()
    if not (repo / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("provide local DINOv2 repo and backbone-only weights")
    backbone = torch.hub.load(str(repo), model_config.backbone, source="local", pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    return NearDepthV2(backbone, model_config)
