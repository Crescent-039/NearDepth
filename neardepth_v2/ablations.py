"""One explicit experiment matrix; no exhaustive combination search or training."""
from dataclasses import replace
from .models.sdf import SDFOptions


# Each specification starts from the same FULL options, never another variant.
SPECS = {
    'full': ('reference', '完整SDF', 'sdf', {}),
    'simple': ('reference', '投影求和+双线性上采样基线', 'simple',
               {'use_cls_modulation': False, 'use_semantic_gate': False, 'enhancement': 'identity', 'upsampling': 'bilinear'}),
    'without_cls': ('module', '仅去除CLS调制', 'sdf', {'use_cls_modulation': False}),
    'without_gate': ('module', '仅去除语义门控', 'sdf', {'use_semantic_gate': False}),
    'without_mrfe': ('module', '仅移除整个双分支增强和融合', 'sdf', {'enhancement': 'identity'}),
    'bilinear': ('module', '仅将两级DySample替换为双线性插值', 'sdf', {'upsampling': 'bilinear'}),
    'large_only': ('branch', '保留大核分支及其对应融合核，移除空洞分支', 'sdf', {'enhancement': 'large_only'}),
    'dilated_only': ('branch', '保留空洞分支及其对应融合核，移除大核分支', 'sdf', {'enhancement': 'dilated_only'}),
    'without_cls_gate': ('interaction', '同时去除CLS调制与语义门控', 'sdf',
                         {'use_cls_modulation': False, 'use_semantic_gate': False}),
    'cls_only': ('progressive', '基线只增加CLS调制', 'sdf',
                 {'use_semantic_gate': False, 'enhancement': 'identity', 'upsampling': 'bilinear'}),
    'cls_gate': ('progressive', '基线增加CLS调制和语义门控', 'sdf',
                 {'enhancement': 'identity', 'upsampling': 'bilinear'}),
    'native_grid': ('implementation_control', '取消H/16中间对齐，仍保留两级x2与最终H/4对齐', 'sdf', {'grid_alignment': 'native'}),
    'identity_init': ('implementation_control', 'CLS仿射调制改为恒等初始化', 'sdf', {'modulation_init': 'identity'}),
}

COMPARISONS = {
    'single_removal': ['full', 'without_cls', 'without_gate', 'without_mrfe', 'bilinear'],
    'enhancement_branches': ['without_mrfe', 'large_only', 'dilated_only', 'full'],
    'cls_gate_interaction': ['full', 'without_cls', 'without_gate', 'without_cls_gate'],
    'progressive_addition': ['simple', 'cls_only', 'cls_gate', 'bilinear', 'full'],
    'implementation_controls': ['full', 'native_grid', 'identity_init'],
}


def ablation_models(base_model, seed):
    if base_model.decoder != 'sdf' or base_model.sdf != SDFOptions():
        raise ValueError('ablation base must be the canonical full SDF, not an already-ablated model')
    common_seed = base_model.common_init_seed if base_model.common_init_seed is not None else seed
    decoder_seed = base_model.decoder_init_seed if base_model.decoder_init_seed is not None else common_seed
    return {name: replace(base_model, decoder=decoder, sdf=replace(SDFOptions(), **changes),
                          common_init_seed=common_seed, decoder_init_seed=decoder_seed)
            for name, (_, _, decoder, changes) in SPECS.items()}
