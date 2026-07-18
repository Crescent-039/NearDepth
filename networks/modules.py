import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    from ablation_study.DPT_blocks import FeatureFusionBlock_custom, _make_scratch
except ModuleNotFoundError:
    FeatureFusionBlock_custom = None
    _make_scratch = None
try:
    from ablation_study.vit import ProjectReadout
except ModuleNotFoundError:
    ProjectReadout = None

# ================SDF模块组==start===============

class SDFDecoder(nn.Module):
    """
    Semantic-Driven Frequency Decoder
    1. 抛弃平行融合，采用最深层引导的语义-细节非对称路由 (Semantic-Detail Routing)
    2. 双频解耦几何增强 (Frequency-Decoupled Enhancer) 替代单路大核
    3. H/12 安全过渡带配合 DySample 进行边界保护上采样
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        self.out_channels = out_channels
        # 通道统一对齐
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels_list
        ])

        # GLAM 调制器 (保留你精彩的 CLS 动态参数设计)
        # 注意：这里只调制前3层(细节层)，第4层(语义层)作为Boss单独使用
        self.glam_mlp = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, 3 * out_channels * 2) # 只需生成前 3 层的 Gamma, Beta
        )

        # 语义门控掩码生成器 (Innovation 1: Semantic Routing)
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid() # 生成 0-1 的空间注意力图
        )

        # 双频解耦增强器 (Innovation 2: Frequency Decoupling)
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=7, padding=3, groups=out_channels//2, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU()
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=2, dilation=2, bias=False), # 膨胀卷积抓边缘
            nn.BatchNorm2d(out_channels//2),
            nn.GELU()
        )
        self.freq_fusion = nn.Conv2d(out_channels, out_channels, 1)

        # 二阶段上采样器，并且别忘了用卷积块和BN和激活函数调制边缘
        self.dysample_stage1 = nn.Sequential(
            DySample(out_channels, scale=2, style='lp', groups=4, dyscope=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.dysample_stage2 = nn.Sequential(
            DySample(out_channels, scale=2, style='lp', groups=4, dyscope=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, features, cls_token, img_h, img_w):
        # 特征投影分离
        proj_feats = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        detail_feats = proj_feats[:3]   # 浅层/中层负责物理细节
        semantic_feat = proj_feats[3]   # 最深层负责全局语义 (Boss)

        # GLAM 动态调制 (针对细节层)
        glam_params = self.glam_mlp(cls_token) #[B, 3 * 256 * 2]
        gamma = glam_params[:, :768].view(-1, 3, 256, 1, 1)  
        beta = glam_params[:, 768:].view(-1, 3, 256, 1, 1) 
        
        modulated_details = 0
        for i in range(3):
            modulated_details = modulated_details + (detail_feats[i] * gamma[:, i] + beta[:, i])

        # 语义门控路由
        # Boss 层生成掩码，滤除细节层中的背景噪声，只保留对深度估计有用的高频边缘
        spatial_gate = self.semantic_gate(semantic_feat) #[B, 1, H/14, W/14]
        filtered_details = modulated_details * spatial_gate
        
        # 细节与语义正式汇合
        fused_tokens = filtered_details + semantic_feat

        # 双频解耦增强
        low_f = self.low_freq_conv(fused_tokens)
        high_f = self.high_freq_conv(fused_tokens)
        enhanced_feat = self.freq_fusion(torch.cat([low_f, high_f], dim=1)) #[B, 256, H/14, W/14]

        # 安全逐级上采样，从H/14到H/16，从H/16到H/8，再从H/8到H/4
        target_h4, target_w4 = img_h // 4, img_w // 4
        target_h16, target_w16 =  img_h // 16, img_w // 16
        out_h14 = enhanced_feat
        out_h16 = F.interpolate(out_h14, size=(target_h16, target_w16), mode='bilinear', align_corners=False)
        out_h8 = self.dysample_stage1(out_h16)
        out_h4 = self.dysample_stage2(out_h8)

        out_h4 = self.refinement(out_h4)
        
        if out_h4.shape[-2:] != (target_h4, target_w4):
            out_h4 = F.interpolate(out_h4, size=(target_h4, target_w4), mode='bicubic', align_corners=False)
            
        return out_h4


class DPTNeckOfficialLike(nn.Module):
    """
    基于 ablation_study 官方文件改写的 DPT decoder。

    设计目标：
    1. 尽量复用官方 DPT 的 reassemble / scratch / refinenet 范式；
    2. 保留当前 AincradNet 的 DINOv2 backbone，不改主干，只替换 SDF neck；
    3. 输出仍保持为 [B, 256, H/4, W/4]，便于直接接入现有 AdaBins head。
    """
    def __init__(self, in_channels_list, out_channels=256, use_bn=False):
        super().__init__()
        if _make_scratch is None or ProjectReadout is None:
            raise ImportError("dpt_official requires the optional 'timm' dependency")
        if len(in_channels_list) != 4:
            raise ValueError(f"DPTNeckOfficialLike expects 4 feature levels, got {len(in_channels_list)}")

        self.pre_scratch_channels = [96, 192, 384, 768]

        self.readout_projects = nn.ModuleList([
            ProjectReadout(in_channels_list[0], start_index=1),
            ProjectReadout(in_channels_list[1], start_index=1),
            ProjectReadout(in_channels_list[2], start_index=1),
            ProjectReadout(in_channels_list[3], start_index=1),
        ])

        self.act_postprocess1 = nn.Sequential(
            nn.Conv2d(in_channels_list[0], self.pre_scratch_channels[0], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(
                in_channels=self.pre_scratch_channels[0],
                out_channels=self.pre_scratch_channels[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
            ),
        )
        self.act_postprocess2 = nn.Sequential(
            nn.Conv2d(in_channels_list[1], self.pre_scratch_channels[1], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(
                in_channels=self.pre_scratch_channels[1],
                out_channels=self.pre_scratch_channels[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
        )
        self.act_postprocess3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], self.pre_scratch_channels[2], kernel_size=1, stride=1, padding=0),
        )
        self.act_postprocess4 = nn.Sequential(
            nn.Conv2d(in_channels_list[3], self.pre_scratch_channels[3], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(
                in_channels=self.pre_scratch_channels[3],
                out_channels=self.pre_scratch_channels[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.scratch = _make_scratch(self.pre_scratch_channels, out_channels, groups=1, expand=False)

        activation = nn.ReLU(False)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(
            out_channels, activation, deconv=False, bn=use_bn, expand=False, align_corners=True
        )
        self.scratch.refinenet2 = FeatureFusionBlock_custom(
            out_channels, activation, deconv=False, bn=use_bn, expand=False, align_corners=True
        )
        self.scratch.refinenet3 = FeatureFusionBlock_custom(
            out_channels, activation, deconv=False, bn=use_bn, expand=False, align_corners=True
        )
        self.scratch.refinenet4 = FeatureFusionBlock_custom(
            out_channels, activation, deconv=False, bn=use_bn, expand=False, align_corners=True
        )

    def _apply_readout(self, feat, cls_token, readout_project):
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        seq = torch.cat([cls_token.unsqueeze(1), tokens], dim=1)
        seq = readout_project(seq)
        feat = seq.transpose(1, 2).contiguous().reshape(b, c, h, w)
        return feat

    def forward(self, features, cls_token=None, img_h=None, img_w=None):
        if cls_token is None:
            raise ValueError("DPTNeckOfficialLike requires cls_token for official ProjectReadout.")
        if img_h is None or img_w is None:
            raise ValueError("DPTNeckOfficialLike requires img_h and img_w.")

        layer_1 = self._apply_readout(features[0], cls_token, self.readout_projects[0])
        layer_2 = self._apply_readout(features[1], cls_token, self.readout_projects[1])
        layer_3 = self._apply_readout(features[2], cls_token, self.readout_projects[2])
        layer_4 = self._apply_readout(features[3], cls_token, self.readout_projects[3])

        layer_1 = self.act_postprocess1(layer_1)
        layer_2 = self.act_postprocess2(layer_2)
        layer_3 = self.act_postprocess3(layer_3)
        layer_4 = self.act_postprocess4(layer_4)

        target_sizes = [
            (img_h // 4, img_w // 4),
            (img_h // 8, img_w // 8),
            (img_h // 16, img_w // 16),
            (img_h // 32, img_w // 32),
        ]
        layer_1 = F.interpolate(layer_1, size=target_sizes[0], mode='bilinear', align_corners=True)
        layer_2 = F.interpolate(layer_2, size=target_sizes[1], mode='bilinear', align_corners=True)
        layer_3 = F.interpolate(layer_3, size=target_sizes[2], mode='bilinear', align_corners=True)
        layer_4 = F.interpolate(layer_4, size=target_sizes[3], mode='bilinear', align_corners=True)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)

        if path_4.shape[-2:] != layer_3_rn.shape[-2:]:
            path_4 = F.interpolate(
                path_4,
                size=layer_3_rn.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)

        if path_3.shape[-2:] != layer_2_rn.shape[-2:]:
            path_3 = F.interpolate(
                path_3,
                size=layer_2_rn.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)

        if path_2.shape[-2:] != layer_1_rn.shape[-2:]:
            path_2 = F.interpolate(
                path_2,
                size=layer_1_rn.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if path_1.shape[-2:] != (img_h // 4, img_w // 4):
            path_1 = F.interpolate(path_1, size=(img_h // 4, img_w // 4), mode='bilinear', align_corners=True)
        return path_1


# ================SDF模块组==end===============

# ================AdaBins模块组==start================
class LiteAdaBinsHead(nn.Module):
    """Adaptive bins whose public output is normalized relative depth.

    Bins are learned in the internal coordinate ``u = 1 / rho``. Their
    centers are converted back to ``rho`` before the pixel-wise expectation.
    """

    def __init__(self, in_channels=256, n_bins=100, min_val=1e-3, max_val=1.0, norm='linear', feat_h2_channels=32, h2_alpha_max=0.15):
        super(LiteAdaBinsHead, self).__init__()

        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        if not 0.0 < self.min_val < self.max_val <= 1.0:
            raise ValueError(
                "relative-depth bounds must satisfy 0 < min_val < max_val <= 1, "
                f"got [{self.min_val}, {self.max_val}]"
            )
        # 极简全局 Bins 生成器
        # 输入: DINOv2 CLS Token (384维), 输出: n_bins 个宽度
        self.bin_mlp = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, n_bins)
        )
        # 局部像素区间分类器
        # 替换了原版 mViT，极其轻量
        self.pixel_classifier = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, n_bins, kernel_size=1)
        )
        
        # 为了做加法残差
        self.align_h2 = nn.Sequential(
            nn.Conv2d(feat_h2_channels, n_bins, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_bins)
            # 不能加 GELU 或 ReLU！
            # 因为输出将直接加到 Logits (未归一化的概率) 上，Logits 需要包含负数。
        )
        
        # 多阶段动态上采样模块
        self.dysample_2x_stage1 = nn.Sequential(
            DySample(n_bins, scale=2, style='lp', groups=4, dyscope=False),
            nn.Conv2d(n_bins, n_bins, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_bins),
            nn.ReLU(inplace=True))
        
        self.dysample_2x_stage2 = nn.Sequential(
            DySample(n_bins, scale=2, style='lp', groups=4, dyscope=False),
            nn.Conv2d(n_bins, n_bins, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_bins),
            nn.ReLU(inplace=True))

        self.refinement = nn.Sequential(
            nn.Conv2d(n_bins, n_bins, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_bins),
            nn.ReLU(inplace=True))

        self.h2_alpha_max = h2_alpha_max
        self.h2_alpha_ratio = 1.0
        # 可学习注入强度，初始化几乎0（避免训练初期抄纹理）
        self.h2_alpha_logit = nn.Parameter(torch.tensor(-6.0))
        # 边界门控，只输出1通道mask（不直接输出n_bins纹理）
        self.h2_gate = nn.Sequential(
            nn.Conv2d(feat_h2_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, cls_token, feat_h2=None):
        # x: DPT 特征图[B, 256, H/4, W/4]
        # cls_token: DINOv2 全局特征 [B, 384]
        # 生成自适应区间宽度 (利用全局先验)

        # 在逆空间计算范围
        # 假设 min_val=0.1m, max_val=10m
        # 则 min_inv = 1/10 = 0.1, max_inv = 1/0.1 = 10
        min_inv = 1.0 / self.max_val
        max_inv = 1.0 / self.min_val

        # 生成自适应区间宽度 (在逆空间进行)
        bin_widths_logits = self.bin_mlp(cls_token) 
        bin_widths_normed = F.softmax(bin_widths_logits.float(), dim=1).to(x.dtype)
        
        # 转换为逆空间的物理宽度
        inv_bin_widths = (max_inv - min_inv) * bin_widths_normed
        inv_bin_widths = F.pad(inv_bin_widths, (1, 0), mode='constant', value=min_inv)
        
        # 计算逆空间的边界 (Inverse Edges)
        inv_bin_edges = torch.cumsum(inv_bin_widths, dim=1) #[B, n_bins + 1]

        # 2. 计算逆空间的中心点并还原回物理深度 (单位：米)
        inv_centers = 0.5 * (inv_bin_edges[:, :-1] + inv_bin_edges[:, 1:]) # [B, n_bins]
        
        # 将逆空间中心点转回物理深度中心点
        # 现中心点在近处非常密集，在远处非常稀疏
        relative_centers = (1.0 / inv_centers).clamp(
            min=self.min_val, max=self.max_val
        )
        centers = relative_centers.view(-1, self.n_bins, 1, 1) #[B, n_bins, 1, 1]
        # 生成像素概率映射 (Pixel-wise Probability)    H/4版本的暂且保留，可做输出
        pixel_logits_h4 = self.pixel_classifier(x) # [B, n_bins, H/4, W/4]
        pixel_probs_h4 = F.softmax(pixel_logits_h4.float(), dim=1).to(x.dtype)
        # 概率加权求和，得到最终深度
        pred_depth_h4 = torch.sum(pixel_probs_h4 * centers, dim=1, keepdim=True) #[B, 1, H/4, W/4]
        # 特征域多阶段上采样
        # Stage 1:[B, 100, H/4, W/4] -> [B, 100, H/2, W/2]
        pixel_logits_h2 = self.dysample_2x_stage1(pixel_logits_h4)

        
        # 残差连接
        if feat_h2 is not None:
            # 维度对齐：把 32 通道的物理特征，映射为 100 通道的概率偏置
            aligned_h2_edges = self.align_h2(feat_h2)
            if aligned_h2_edges.shape[-2:] != pixel_logits_h2.shape[-2:]:
                aligned_h2_edges = F.interpolate(
                    aligned_h2_edges, size=pixel_logits_h2.shape[-2:],
                    mode='bicubic', align_corners=False)
            gate_h2 = self.h2_gate(feat_h2)
            if gate_h2.shape[-2:] != pixel_logits_h2.shape[-2:]:
                gate_h2 = F.interpolate(gate_h2, size=pixel_logits_h2.shape[-2:],
                                        mode='bicubic', align_corners=False)
            alpha = torch.sigmoid(self.h2_alpha_logit) * self.h2_alpha_max * self.h2_alpha_ratio
            edge_bias = torch.tanh(aligned_h2_edges)                  # 限幅防爆
            pixel_logits_h2 = pixel_logits_h2 + alpha * gate_h2 * edge_bias

        
        # Stage 2:[B, 100, H/2, W/2] -> [B, 100, H, W]
        pixel_logits_h = self.dysample_2x_stage2(pixel_logits_h2)
        # 精炼一下输出的深度图
        pixel_logits_h = self.refinement(pixel_logits_h)
        
        # 在最高分辨率下计算 Softmax
        pixel_probs_h = F.softmax(pixel_logits_h.float(), dim=1).to(x.dtype)
        # 概率加权求和，得到最终高分辨率深度图[B, 1, H, W]
        # 此时得到的深度图，边缘将保持着特征图级别的极高锐利度
        pred_depth_h = torch.sum(pixel_probs_h * centers, dim=1, keepdim=True)
        pred_depth_h = pred_depth_h.clamp(min=self.min_val, max=self.max_val)
        # 为了 Loss 计算的一致性，bin_edges 建议也转回物理深度返回
        # 但由于 cumsum 是在逆空间做的，转回后顺序会反过来，这里通常返回真实深度的 edges
        relative_bin_edges = (1.0 / torch.flip(inv_bin_edges, dims=[1])).clamp(
            min=self.min_val, max=self.max_val
        )

        return relative_bin_edges, pred_depth_h


# ================AdaBins模块组==end================


# ================边缘精细化模块组==start================
class DetailRefiner(nn.Module):
    def __init__(self, in_channels=4, mid_channels=32, delta_max=0.05):
        super(DetailRefiner, self).__init__()
        self.delta_max = delta_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.delta_head = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)  # 预测残差
        self.mask_head  = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 边界区域mask
        )

    def forward(self, coarse_depth, rgb_image):
        concat_input = torch.cat([coarse_depth, rgb_image], dim=1)
        feat = self.conv1(concat_input)
        feat = self.conv2(feat)

        delta = self.delta_max * torch.tanh(self.delta_head(feat))   # 限幅
        mask  = self.mask_head(feat)                                   # [0,1]

        refined_depth = coarse_depth + mask * delta
        refined_depth = torch.clamp(refined_depth, min=1e-6)

        return refined_depth, mask, delta
        
# ================边缘精细化模块组==end================


# ================GLNeck模块组==start===============

class GLNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # GLAM 调制生成器 (替代 SDT 的笨重 Concat 和静态权重)
        # 输入 384维的 CLS，生成 4层*256 的 Gamma(缩放) 和 Beta(偏移)，以及 4个动态权重
        self.glam_mlp = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Linear(128, 4 * out_channels * 2 + 4) # Gamma, Beta 和 Weights
        )
        
        # 将 4 层特征统一下采样维度到 256
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels_list
        ])

        # LRGB 大核重参数化模块 (替代 SDT 的 3x3 SDE)
        self.lrgb_enhancer = nn.Sequential(
            # 训练时可以写成多分支，部署时转为单一 7x7 DwConv
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1) # 通道混合
        )

        self.dysample_3x = DySample(in_channels=out_channels, scale=3, style='lp', groups=4, dyscope=False)

    def forward(self, features, cls_token, img_h, img_w):
        # 解析 GLAM 的调制参数
        glam_params = self.glam_mlp(cls_token) #[B, 4 * 256 * 2 + 4]
        
        # 分离出 Gamma, Beta 和动态权重 Weights
        gamma = glam_params[:, :1024].view(-1, 4, 256, 1, 1)  # 4层的缩放因子
        beta = glam_params[:, 1024:2048].view(-1, 4, 256, 1, 1) # 4层的偏移因子
        layer_weights = F.softmax(glam_params[:, 2048:], dim=-1) # 4层的动态融合权重 [B, 4]

        fused_tokens = 0
        for i in range(4):
            # 投影维度
            proj_feat = self.projections[i](features[i]) # [B, 256, H/14, W/14]
            # 施加 GLAM 仿射调制：F' = F * gamma + beta
            modulated_feat = proj_feat * gamma[:, i] + beta[:, i]
            # 施加动态权重融合
            fused_tokens = fused_tokens + (modulated_feat * layer_weights[:, i].view(-1, 1, 1, 1))

        # 施加 LRGB 几何增强
        enhanced_feat = self.lrgb_enhancer(fused_tokens)

        target_h4, target_w4 = img_h // 4, img_w // 4
        # A. 微调插值到 H/12 (极其轻微的 1.16 倍拉伸，保护高频特征)
        target_h12, target_w12 = img_h // 12, img_w // 12
        feat_h12 = F.interpolate(enhanced_feat, size=(target_h12, target_w12), mode='bicubic', align_corners=False)
        # B. 整数倍动态采样到 H/4 (3 倍放大，DySample 的主场)
        out_h4 = self.dysample_3x(feat_h12)
        # C. 尺寸安全对齐保护 (防止长宽不能被 12 完美整除时差 1 个像素)
        if out_h4.shape[-2:] != (target_h4, target_w4):
            out_h4 = F.interpolate(out_h4, size=(target_h4, target_w4), mode='bicubic', align_corners=False)
            
        return out_h4 # 完美输出 [B, 256, H/4, W/4]
        
# ================GLNeck模块组==end===============

        
# ================DPT模块组==start================
class DPTNeck(nn.Module):
    """
    DPT
    Input:  List of features [H/4, H/8, H/16, H/32]
    Output: Fused feature map [H/4]
    DPT模块需要对DINOV2-S的四个层进行逐层采样，需要一个三阶段重组（Reassemble）模块从每一层输出token中恢复类图像表征
    还需要一个融合模块逐级向上合并特征，最后输出
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        # 因为 DINOv2 的 4 层特征全是 H/14 分辨率
        # 需要在 Reassemble 里分别缩放到 H/4, H/8, H/16, H/32 来构造特征金字塔
        self.reassemble_4  = Reassemble(in_channels_list[0], out_channels, target_stride=4)
        self.reassemble_8  = Reassemble(in_channels_list[1], out_channels, target_stride=8)
        self.reassemble_16 = Reassemble(in_channels_list[2], out_channels, target_stride=16)
        self.reassemble_32 = Reassemble(in_channels_list[3], out_channels, target_stride=32)

        # 实例化四个fusion模块
        self.fusion04 = FusionBlock(out_channels, use_fusion=False, upsample=True)
        self.fusion03 = FusionBlock(out_channels, use_fusion=True, upsample=True)
        self.fusion02 = FusionBlock(out_channels, use_fusion=True, upsample=True)
        self.fusion01 = FusionBlock(out_channels, use_fusion=True, upsample=False)

    def forward(self, features, img_h, img_w):
        # 获取每一层特征，构造特征金字塔
        feat_4  = self.reassemble_4(features[0], img_h, img_w)
        feat_8  = self.reassemble_8(features[1], img_h, img_w)
        feat_16 = self.reassemble_16(features[2], img_h, img_w)
        feat_32 = self.reassemble_32(features[3], img_h, img_w)

        out04 = self.fusion04(None, feat_32)
        out03 = self.fusion03(out04, feat_16)
        out02 = self.fusion02(out03, feat_8)
        out01 = self.fusion01(out02, feat_4)
        # DPT模块输出经过融合后的特征图，大小为H/4, W/4
        return out01


class DPTReassembleStatic(nn.Module):
    """
    静态重组模块，尽量贴近官方 DPT 的 deterministic reassemble。
    对于 DINOv2 的 H/14 特征，使用固定插值/池化构造多尺度金字塔。
    """
    def __init__(self, in_channels, out_channels, target_stride):
        super().__init__()
        self.target_stride = target_stride
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, img_h, img_w):
        x = self.project(x)
        target_h = img_h // self.target_stride
        target_w = img_w // self.target_stride
        if x.shape[-2:] == (target_h, target_w):
            return x

        mode = 'bilinear' if target_h >= x.shape[-2] else 'area'
        if mode == 'bilinear':
            return F.interpolate(x, size=(target_h, target_w), mode=mode, align_corners=True)
        return F.interpolate(x, size=(target_h, target_w), mode=mode)


class ResidualConvUnitOfficial(nn.Module):
    def __init__(self, features, activation, bn=False):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not bn)
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlockOfficial(nn.Module):
    def __init__(self, features, activation, bn=False, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.resConfUnit1 = ResidualConvUnitOfficial(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnitOfficial(features, activation, bn)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            res_out = self.resConfUnit1(xs[1])
            if output.shape[2:] != res_out.shape[2:]:
                output = F.interpolate(output, size=res_out.shape[2:], mode='bilinear', align_corners=True)
            output = output + res_out
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


class FusionBlock(nn.Module):
    def __init__(self, channels, use_fusion=True, upsample=True):
        super(FusionBlock, self).__init__()
        # 两个RCU 模块，Residual Conv Unit ：x = Conv(ReLU(Conv(x))) + x
        self.RCU01 = ResidualConvUnit(channels)
        self.RCU02 = ResidualConvUnit(channels)
        # 定义最后的投影层
        self.projs = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        # 是否融合上一级特征的开关
        self.use_fusion = use_fusion
        # 是否继续上采样的开关，最后一层无需上采样，最后输出H/4大小的特征图
        self.upsample = upsample
        # 如果有上采样就使用动态采样
        if upsample:
            self.dysample = DySample(channels, scale=2, style='lp', groups=4, dyscope=False)
            self.refiner = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, deep_features, current_features):
        """
        deep_features是从上一级融合完传过来的特征（B, 256, H/2n, W/2n)
        current_features是当前层经过聚合层投影过来的特征（B, 256, H/n, W/n)
        将 current_features 经过一个 RCU 提取特征，再和deep_features相加
        因为上一级特征经过Resample0.5，所以维度可以直接和current_features对齐
        相加后经过一个重采样和投影
        """
        current = self.RCU01(current_features)

        if self.use_fusion and deep_features is not None:
            # 动态尺寸对齐保护机制
            # 如果从上一层传过来的 deep_features 尺寸与当前的 current 不匹配 (比如 160 和 161)
            # 则使用双线性插值，将 deep_features 强行对齐到 current 的尺寸
            if deep_features.shape[-2:] != current.shape[-2:]:
                deep_features = F.interpolate(
                    deep_features, 
                    size=current.shape[-2:], 
                    mode='bicubic', 
                    align_corners=False
                )
            out = current + deep_features
        else:
            out = current

        out = self.RCU02(out)
        # 改用动态采样代替双线性插值
        if self.upsample:
            out = self.dysample(out)
            out = self.refiner(out)

        out = self.projs(out)
        return out


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()

        # 为了节省计算量和参数，卷积层接BN时，通常设置bias=False
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)
        # 暂且不清楚批量归一化会不会对真实深度估计造成影响，反正先加上，毕竟原论文加了
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        shortcut = x
        # 先过一遍relu，预激活
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out + shortcut

"""
class Reassemble(nn.Module):

    def __init__(self, in_channels, out_channels, target_stride):
        super(Reassemble, self).__init__()

        self.target_stride = target_stride
        # 投影层，对齐通道数
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, img_h, img_w):
        x = self.project(x)
        # 动态计算目标尺寸 (比如 img_h=644, target_stride=4, 则 target_h=161)
        target_h = img_h // self.target_stride
        target_w = img_w // self.target_stride
        # 使用双线性插值进行缩放 (完美兼容任意输入尺寸，杜绝除不尽的问题)
        x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return x
"""


class Reassemble(nn.Module):
    """
    融合 DySample 精髓的动态重组模块 (Fractional DySample Reassemble)
    不仅支持 DINOv2 的非整数倍动态上采样 (如 14 -> 4 = 3.5倍)，
    而且保留了 DySample 极低算力、极高锐度的偏移采样机制。
    """
    def __init__(self, in_channels, out_channels, target_stride):
        super(Reassemble, self).__init__()
        self.target_stride = target_stride
        
        # 投影层：对齐通道数到 DPT 需要的 256
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # 汲取 DySample 精髓：极简偏移量预测 (1x1卷积)
        # 输出 2 个通道，代表 (dx, dy)
        self.offset_conv = nn.Conv2d(out_channels, 2, kernel_size=1)
        
        # 使用 std=0.001 的极小方差初始化
        # 保证训练初期偏移量几乎为 0，退化为普通双线性插值，防止网络直接崩溃
        nn.init.normal_(self.offset_conv.weight, mean=0, std=0.001)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x, img_h, img_w):
        # x: DINOv2 的特征图 [B, 384, H/14, W/14]
        B, C, H_src, W_src = x.shape
        x_proj = self.project(x) # [B, 256, H/14, W/14]
        
        # 动态计算 DPT 当前层需要的目标尺寸 (例如 H/4, H/8)
        target_h = img_h // self.target_stride
        target_w = img_w // self.target_stride
        
        # 在低分辨率 (H/14) 预测偏移量，计算量极小！
        offset_low = self.offset_conv(x_proj) #[B, 2, H/14, W/14]
        
        # 将偏移量适配到目标分辨率 (替代 pixel_shuffle，完美支持 3.5 倍这种小数)
        offset_target = F.interpolate(offset_low, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        # 构造目标分辨率的标准网格坐标 [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, target_h, device=x.device),
            torch.linspace(-1, 1, target_w, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1) # [B, target_h, target_w, 2]
        
        # 汲取 DySample 的乘法因子：* 0.25 (详见原代码 forward_lp)
        # 加上偏移量，形成畸变网格
        # 注意维度转换: offset_target 是[B, 2, H, W]，需 permute 为 [B, H, W, 2]
        dynamic_grid = base_grid + offset_target.permute(0, 2, 3, 1) * 0.25
        
        # 用 grid_sample 去原始高清投影特征上“捞”像素！
        out = F.grid_sample(x_proj, dynamic_grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return out


# 适用于密集预测任务的动态上采样模块DySample
# 效果比双线性插值好，而且轻量化，比卷积上采样算力消耗少

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# ================DPT模块组==end================

#=================边缘引导模块组==start================

class EdgeGuidedAttention(nn.Module):
    """
    - edge_feat只用于生成空间门控，不直接拼接进语义值
    - 门控允许正负（可增强也可抑制），避免只增强纹理
    """
    def __init__(self, semantic_dim=256, edge_dim=64, gate_max=0.3):
        super().__init__()
        self.gate_max = gate_max
        self.edge_gate = nn.Sequential(
            nn.Conv2d(edge_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh()  # [-1, 1]
        )
        # 可学习强度，初值近0
        self.beta_logit = nn.Parameter(torch.tensor(-6.0))
        self.refine = nn.Sequential(
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, semantic_dim),
            nn.GELU()
        )
    def forward(self, semantic_feat, edge_feat):
        if semantic_feat.shape[-2:] != edge_feat.shape[-2:]:
            edge_feat = F.interpolate(edge_feat, size=semantic_feat.shape[-2:], mode='bicubic', align_corners=False)
        gate = self.edge_gate(edge_feat)  # [B,1,H,W], [-1,1]
        beta = torch.sigmoid(self.beta_logit) * self.gate_max
        out = semantic_feat * (1.0 + beta * gate)  # 广播到通道维
        out = self.refine(out)
        return out

#=================边缘引导模块组==end================






