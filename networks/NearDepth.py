import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings


from .modules import DPTNeck, DPTNeckOfficialLike, LiteAdaBinsHead, DySample, DetailRefiner, GLNeck, SDFDecoder, EdgeGuidedAttention


# 网络前向传播过程的通道数一定要搞明白
class NearDepth(nn.Module):
    def __init__(self, img_size, neck_type='sdf'):
        super().__init__()
        self.img_size = img_size
        self.neck_type = neck_type

        self.backbone = torch.hub.load(repo_or_dir='pretrained_model/DINOV2', model='dinov2_vits14_reg', source='local')
        # 允许骨干网络微调
        # 先全部冻结
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 解冻最后2个Transformer block
        for p in self.backbone.blocks[-2:].parameters():
            p.requires_grad = True
        # 建议把最后norm也解冻
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        self.patch_size = 14
        self.embed_dim = 384
        
        # 动态获取通道
        self.in_channels = [self.embed_dim, self.embed_dim, self.embed_dim, self.embed_dim]
        if self.neck_type == 'sdf':
            self.neck = SDFDecoder(self.in_channels, 256)
        elif self.neck_type == 'dpt_custom':
            self.neck = DPTNeck(self.in_channels, 256)
        elif self.neck_type == 'dpt_official':
            self.neck = DPTNeckOfficialLike(self.in_channels, 256)
        else:
            raise ValueError(f"Unsupported neck_type: {self.neck_type}")
        
         # 提取 H/2 极高清浅层特征 (32通道)
        self.stem_h2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # 继续提取 H/4 浅层特征 (64通道)
        self.stem_h4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # 边缘引导注意力模块
        #self.EGA_fusion = EdgeGuidedAttention(semantic_dim=256, edge_dim=64)
        
        # H/4 的融合层保持不变
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        # AdaBins输出头
        self.AdaBins = LiteAdaBinsHead(256, 100, 0.1, 3.0, 'linear')
        # 边缘精修模块
        #self.Refiner = DetailRefiner(in_channels=4, mid_channels=32, delta_max=0.05)

    def extract_dinov2_features(self, x):
        """提取 DINOv2 特征并处理 Register Tokens"""
        B, C, H, W = x.shape
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        
        # 获取中间层特征
        # 获取第 2, 5, 8, 11 层的特征，return_class_token=True 以获取全局语义
        # DINOv2 官方提供 get_intermediate_layers
        intermediate_outputs = self.backbone.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, return_class_token=True)
        
        out_features =[]
        for item in intermediate_outputs:
            feat, cls = item  # feat可能是[B,C,H,W]或[B,H,W,C]
            if feat.ndim == 4 and feat.shape[1] == self.embed_dim:
                feat_2d = feat
            elif feat.ndim == 4 and feat.shape[-1] == self.embed_dim:
                feat_2d = feat.permute(0, 3, 1, 2).contiguous()
            elif feat.ndim == 3:
                feat_2d = feat.reshape(B, h_feat, w_feat, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            else:
                raise RuntimeError(f"Unexpected feat shape: {feat.shape}")
            out_features.append(feat_2d)
            cls_token = cls
            
        return out_features, cls_token

    def forward(self, x):
        # DINOv2 的输入宽高必须是14倍数
        # 动态 Padding 机制
        B, C, orig_H, orig_W = x.shape
        pad_h = (self.patch_size - orig_H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - orig_W % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            
        padded_H, padded_W = x.shape[2], x.shape[3]

        # 骨干网络特征提取
        features, cls_token = self.extract_dinov2_features(x)
        # DPT 特征融合 
        # 传入原始宽高(填充后)，方便 DPT 内部算缩放比例
        if self.neck_type == 'dpt_custom':
            warnings.warn(
                "当前 neck_type='dpt_custom' 使用的是项目内手搓 DPT 版本，并非官方实现；"
                "若用于论文对比实验，建议优先使用 neck_type='dpt_official'。",
                UserWarning,
                stacklevel=2,
            )
            fused_feat = self.neck(features, padded_H, padded_W)
        else:
            fused_feat = self.neck(features, cls_token, padded_H, padded_W)
         # 提取 H/2 物理边缘 (准备传给 AdaBins 内部)
        feat_h2 = self.stem_h2(x) # [B, 32, H/2, W/2]
        # 提取 H/4 物理边缘
        shallow_edges = self.stem_h4(feat_h2) # [B, 64, H/4, W/4]

        if fused_feat.shape[-2:] != shallow_edges.shape[-2:]:
            shallow_edges = F.interpolate(shallow_edges, size=fused_feat.shape[-2:], mode='bicubic', align_corners=False)
        
        concat_feat = torch.cat([fused_feat, shallow_edges], dim=1)
        guided_fused_feat = self.feature_fusion(concat_feat) #[B, 256, H/4, W/4]
        # 在 H/4 处和DPT输出处，使用边缘引导模块完成第一次大尺度融合
        #guided_fused_feat = self.EGA_fusion(fused_feat, shallow_edges)
        # 传入AdaBins
        bin_edges, pre_depth = self.AdaBins(guided_fused_feat, cls_token, feat_h2)
        '''
        # 精修
        rgb_for_refine = x
        if rgb_for_refine.shape[-2:] != pre_depth.shape[-2:]:
            rgb_for_refine = F.interpolate(rgb_for_refine, size=pre_depth.shape[-2:], mode='bicubic', align_corners=False)
        pre_depth, refine_mask, refine_delta = self.Refiner(pre_depth, rgb_for_refine)
        '''
        # 如果前面做了 Padding，这里要把多余的像素裁掉，恢复原图尺寸
        if pad_h > 0 or pad_w > 0:
            pre_depth = pre_depth[:, :, :orig_H, :orig_W]

        return pre_depth, bin_edges


if __name__ == '__main__':


    print("正在初始化模型...")
    # 2. 实例化模型
    # 确保你的 Mask2Former 权重路径是对的，否则会报错
    model = NearDepth(img_size=640, neck_type='sdf')
    print("模型初始化成功！")


    # 3. 准备虚拟输入数据
    # Batch=1, Channel=3, H=224, W=224
    dummy_input = torch.randn(1, 3, 640, 640)

    # 如果有显卡，放到显卡上跑
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    print(f"正在进行前向传播测试... 输入形状: {dummy_input.shape}")
    output = model(dummy_input)
    #print(f"输入尺寸: {dummy_input.shape}")
    #print(f"输出尺寸: {output.shape}")




