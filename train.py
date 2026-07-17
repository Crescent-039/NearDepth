import csv
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision import transforms

# 引入你的模型和Loss
from networks.Aincrad_Net import AincradNet
from networks.Loss import AincradLoss
from utils import compute_metrics
from dataset import GraspNetDepthDataset, DREDS_CatKnown_Dataset, HypersimDepthDataset, PaperFigureDataset, depth_collate_pad

from networks.Loss import solve_scale_shift, grad_mag, erode_valid_mask

# 训练主流程
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description='Train AincradNet with configurable neck for ablation study')
    parser.add_argument('--neck-type', type=str, default='sdf', choices=['sdf', 'dpt_custom', 'dpt_official'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--exp-suffix', type=str, default='')
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    # --- 配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = (360, 640)
    batch_size = args.batch_size  # 显存小的改 1
    lr = args.lr
    epochs = args.epochs
    neck_type = args.neck_type  # 可选: 'sdf' | 'dpt_custom' | 'dpt_official'

    exp_name = f'AincradNet_{neck_type}'
    if args.exp_suffix:
        exp_name = f'{exp_name}_{args.exp_suffix}'
    exp_root = os.path.join('experiments', exp_name)
    weights_dir = os.path.join(exp_root, 'weights')
    history_dir = os.path.join(exp_root, 'history')
    vis_dir = os.path.join(exp_root, 'visuals')
    highres_vis_dir = os.path.join(exp_root, 'highres')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(highres_vis_dir, exist_ok=True)

    print(f"初始化模型 (Device: {device})...")

    # --- 初始化 ---
    model = AincradNet(img_size, neck_type=neck_type)
    criterion = AincradLoss(stage='coarse', min_depth=1e-3, max_depth=200).to(device)

    # 恢复权重
    weights_path = "./weights/latest.pth"

    start_epoch = 0 # 记录起始 epoch，默认为 0

    if os.path.exists(weights_path):
        print(f"正在从 {weights_path} 加载一阶段训练权重...")
        checkpoint = torch.load(weights_path, map_location='cpu')
        # 解析真正的 model state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        # 加载模型参数
        model.load_state_dict(state_dict, strict=True)
        print("模型权重加载成功！")
    else:
        print(f"权重文件 {weights_path} 不存在，跳过加载")
        checkpoint = None

    model = model.to(device)

    lr_backbone = 1e-5
    lr_head = 1e-5
    # 不训练骨干网络
    trainable_params =[
        {'params': model.backbone.blocks[-2:].parameters(), 'lr': lr_backbone},
        {'params': model.backbone.norm.parameters(), 'lr': lr_backbone},
        {'params': model.neck.parameters(), 'lr': lr_head},
        {'params': model.AdaBins.parameters(), 'lr': lr_head},
        {'params': model.stem_h2.parameters(), 'lr': lr_head},
        {'params': model.stem_h4.parameters(), 'lr': lr_head},
        {'params': model.feature_fusion.parameters(), 'lr': lr_head}
    ]
    # 优化器和学习率调度器
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
     # 如果 checkpoint 存在且含有优化器状态，则恢复
    if checkpoint is not None and isinstance(checkpoint, dict):
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("优化器状态恢复成功！")
            except Exception as e:
                print(f"优化器状态恢复失败（可能是因为改变了 trainable_params 结构），将使用初始优化器。提示: {e}")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"将从 Epoch {start_epoch + 1} 继续训练。")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=4, min_lr=1e-7)

    # ---------------- 实例化训练集与验证集 ----------------
    Hypersim_train_dataset = HypersimDepthDataset('../autodl-tmp/Hypersim_dataset', mode='train', target_size=img_size, sample_size=35000, overfit_one=False)
    Hypersim_val_dataset = HypersimDepthDataset('../autodl-tmp/Hypersim_dataset', mode='val', target_size=img_size, sample_size=500)
    
    Grasp_train_dataset = GraspNetDepthDataset('../autodl-tmp/graspnet_dataset', mode='train', target_size=img_size, sample_size=25000, overfit_one=False)
    Grasp_val_dataset = GraspNetDepthDataset('../autodl-tmp/graspnet_dataset', mode='val', target_size=img_size, sample_size=500)

    # 将 D435 真实内参传给 DREDS，使其在加载时归一化到 D435 视角（消除焦距差异）
    DREDS_train_dataset = DREDS_CatKnown_Dataset('../autodl-tmp/DREDS_CatKnown_Dataset', mode='train', target_size=img_size, sample_size=40000, overfit_one=False,
                                                  ref_fx=Grasp_train_dataset.ref_fx, ref_img_w=Grasp_train_dataset.ref_img_w)
    DREDS_val_dataset = DREDS_CatKnown_Dataset('../autodl-tmp/DREDS_CatKnown_Dataset', mode='val', target_size=img_size, sample_size=500,
                                                ref_fx=Grasp_val_dataset.ref_fx, ref_img_w=Grasp_val_dataset.ref_img_w)
    
    PaperFigure = PaperFigureDataset('../autodl-tmp/PaperFigureDataset', mode='train', target_size=img_size, sample_size=16, overfit_one=False)
    
    # 合并数据集
    combined_train_dataset = ConcatDataset([DREDS_train_dataset, Grasp_train_dataset, Hypersim_train_dataset])
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # 三套验证集
    Grasp_val_loader = DataLoader(Grasp_val_dataset, batch_size=batch_size, shuffle=False)
    DREDS_val_loader = DataLoader(DREDS_val_dataset, batch_size=batch_size, shuffle=False)
    Hypersim_val_loader = DataLoader(Hypersim_val_dataset, batch_size=batch_size, shuffle=False)

    best_rmse = float('inf')  # 记录验证集历史最佳 RMSE
    history = []

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # 用训练集训练
        model.train()
        train_loss_total = 0.0
        print(f"Epoch {epoch + 1} 开始训练...")
        for i, (rgb_global, depth_global, content_mask) in enumerate(train_loader):
            # 全局和局部的图
            rgb_global = rgb_global.to(device)
            depth_global = depth_global.to(device)
            content_mask = content_mask.to(device)
            # 清零梯度
            optimizer.zero_grad()
            # 第一轮全局图前向传播
            pred_depth_g, bin_edges_g = model(rgb_global)
            if epoch == 1:
                criterion.set_stage('edge')
            # 直接在训练循环里内置mask传给loss，更灵活
            valid_mask = ((depth_global > 1e-3) & (depth_global < 200) & (content_mask > 0.5))
            # 计算 Loss
            loss_g, loss_dict_g = criterion(pred_depth_g, bin_edges_g, depth_global, valid_mask)
            # 反向传播
            loss_g.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            # 记录当前 Batch 的总 Loss (Global + Local)
            current_total_loss = loss_g.item()
            train_loss_total += current_total_loss

            if i % 20 == 0:
                print(f"  Step [{i}/{len(train_loader)}] Loss_G: {loss_g.item():.4f} | Total: {current_total_loss:.4f}")
                
        print(f"allocated:{torch.cuda.memory_allocated()/1024**2} | reserved:{torch.cuda.memory_reserved()/1024**2}")
        
        avg_train_loss = train_loss_total / len(train_loader)

        # 验证集验证阶段
        model.eval()
        val_loss_total = 0.0
        valid_val_batches = 0
        # 像素级累计统计
        metric_tot = {
            'n': 0,
            'abs_rel_sum': 0.0,
            'sq_err_sum': 0.0,
            'd1_sum': 0.0,
            'd2_sum': 0.0,
            'd3_sum': 0.0
        }
        print(f"Epoch {epoch + 1} 开始验证...")

        # 验证 GraspNet
        grasp_results = validate(model, Grasp_val_loader, device, criterion, "GraspNet")
        # 验证 DREDS
        dreds_results = validate(model, DREDS_val_loader, device, criterion, "DREDS")
        # 验证 DREDS
        Hypersim_results = validate(model, Hypersim_val_loader, device, criterion, "Hypersim")
        # 根据验证集loss来调整学习率而并非训练集loss，降低过拟合风险
        avg_combined_val_loss = (grasp_results['loss'] + dreds_results['loss'] + Hypersim_results['loss']) / 3
        scheduler.step(avg_combined_val_loss) # 这里建议优先看真实场景的 loss

        # ---------------- 打印报告与最佳模型保存 ----------------
        print(f"📊 [Epoch {epoch+1} Report]")
        print(f"  Train Loss: {avg_train_loss:.4f}")

        avg_rmse = (grasp_results['rmse'] + dreds_results['rmse'] + Hypersim_results['rmse']) / 3
        avg_rmse_aligned = (grasp_results['rmse_aligned'] + dreds_results['rmse_aligned'] + Hypersim_results['rmse_aligned']) / 3
        
        # 基于验证集 RMSE 保存最好的模型
        if avg_rmse_aligned < best_rmse:
            best_rmse = avg_rmse_aligned
            print(f" 🏆 发现更优模型 (grasp RMSE: {grasp_results['rmse']:.4f})(dreds RMSE: {dreds_results['rmse']:.4f})(dreds RMSE: {Hypersim_results['rmse']:.4f})，已保存 Best Checkpoint.")
            print(f"(avg RMSE aligned: {best_rmse:.4f})(grasp RMSE aligned: {grasp_results['rmse_aligned']:.4f})(dreds RMSE aligned: {dreds_results['rmse_aligned']:.4f})(dreds RMSE aligned: {Hypersim_results['rmse_aligned']:.4f})，已保存 Best Checkpoint.")

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            
            torch.save(checkpoint, os.path.join(weights_dir, 'best.pth'))
        
        current_head_lr = optimizer.param_groups[2]['lr']
        current_backbone_lr = optimizer.param_groups[0]['lr']

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss_grasp': float(grasp_results['loss']),
            'val_loss_dreds': float(dreds_results['loss']),
            'val_loss_Hypersim': float(Hypersim_results['loss']),
            'val_loss_mean': float(avg_combined_val_loss),
            'rmse_grasp': float(grasp_results['rmse']),
            'rmse_dreds': float(dreds_results['rmse']),
            'Hypersim_dreds': float(Hypersim_results['rmse']),
            'rmse_aligned_grasp': float(grasp_results['rmse_aligned']),
            'rmse_aligned_dreds': float(dreds_results['rmse_aligned']),
            'rmse_aligned_Hypersim': float(Hypersim_results['rmse_aligned']),
            'rmse_aligned_mean': float(avg_rmse_aligned),
            'lr_head': float(current_head_lr),
            'lr_backbone': float(current_backbone_lr),
        })
        save_training_history(history, history_dir)
        save_loss_curves(history, history_dir, exp_name)

        print(f"Epoch [{epoch + 1}/{epochs}] val_Loss: {avg_combined_val_loss:.4f} | train_Loss: {avg_train_loss:.4f}")
        print(f"SILog_g: {loss_dict_g['silog']:.4f} | MAE_g: {loss_dict_g['MAE']:.4f} | Edge_g: {loss_dict_g['edge']:.4f} | low_g: {loss_dict_g['low']:.4f} \
 | nonedge_g: {loss_dict_g['nonedge']:.4f} | ssi_g: {loss_dict_g['ssi']:.4f} | gm_g: {loss_dict_g['gm']:.4f} | Current Head LR: {current_head_lr:.6f} | Current Backbone LR: {current_backbone_lr:.6f}")

        if epoch % 1 == 0:
            # 简单的可视化保存
            save_visualization(rgb_global, depth_global, pred_depth_g, content_mask, epoch, vis_dir)
            # 插值放大回原图尺寸
            # 使用 bicubic (双三次插值) 比 bilinear 边缘更平滑，伪影更少
            final_depth_resized = F.interpolate(
                pred_depth_g.float(),           
                size=(720, 1280),
                mode='bicubic',
                align_corners=False
            )
            # 转为 numpy (原图大小的物理深度值)
            depth_array_final = final_depth_resized[0, 0].detach().cpu().numpy() 
            # 直接保存一张最高清（与原图完全等分辨率）的彩色深度图
            plt.imsave(
                os.path.join(highres_vis_dir, f"{epoch}_depth_highres.png"), 
                depth_array_final, 
                cmap='Spectral'
            )
            torch.save(model.state_dict(), os.path.join(weights_dir, 'latest.pth'))

    end_time = time.time()
    total_seconds = end_time - start_time
    print("✅ 训练测试完成！")
    print(f"总耗时: {total_seconds}")
    
    torch.save(model.state_dict(), os.path.join(weights_dir, 'final.pth'))


# 验证函数
def validate(model, val_loader, device, criterion, dataset_name="Dataset"):
    model.eval()
    val_loss_total = 0.0
    metric_tot = {
        'n': 0, 'abs_rel_sum': 0.0, 'sq_err_sum': 0.0,
        'd1_sum': 0.0, 'd2_sum': 0.0, 'd3_sum': 0.0, 'abs_rel_sum_aligned': 0.0, 'sq_err_sum_aligned': 0.0
    }
    
    with torch.no_grad():
        for (rgb_global, depth_global, content_mask) in val_loader:
            rgb_g = rgb_global.to(device)
            depth_g = depth_global.to(device)
            content_mask = content_mask.to(device)
            
            pred_depth_g, bin_edges_g = model(rgb_g)
            
            valid_mask = ((depth_g > 1e-3) & (depth_g < 200) & (content_mask > 0.5))
            # 计算 Loss
            loss_v, _ = criterion(pred_depth_g, bin_edges_g, depth_g, valid_mask)
            val_loss_total += float(loss_v.item())
            
            # 计算指标
            stats = compute_metrics(pred_depth_g, depth_g, min_depth=1e-3, max_depth=200, valid_mask=valid_mask)
            if stats is None: continue
            
            metric_tot['n'] += int(stats['n'])
            metric_tot['abs_rel_sum'] += float(stats['abs_rel_sum'])
            metric_tot['sq_err_sum'] += float(stats['sq_err_sum'])
            metric_tot['d1_sum'] += float(stats['d1_sum'])
            metric_tot['d2_sum'] += float(stats['d2_sum'])
            metric_tot['d3_sum'] += float(stats['d3_sum'])

            # ========debug==========
            mask = valid_mask
            scale, shift = solve_scale_shift(pred_depth_g, depth_g, mask)
            pred_aligned = scale * pred_depth_g + shift
            pred_aligned = torch.clamp(pred_aligned, min=1e-4)
            # 用 pred_aligned 计算一套 AbsRel 和 RMSE，与原始指标对比
            stats_aligned = compute_metrics(pred_aligned, depth_g, min_depth=1e-3, max_depth=200, valid_mask=valid_mask)
            
            metric_tot['abs_rel_sum_aligned'] += float(stats_aligned['abs_rel_sum'])
            metric_tot['sq_err_sum_aligned'] += float(stats_aligned['sq_err_sum'])

    # 计算平均指标
    n = max(metric_tot['n'], 1)
    results = {
        'loss': val_loss_total / max(len(val_loader), 1),
        'abs_rel': metric_tot['abs_rel_sum'] / n,
        'rmse': (metric_tot['sq_err_sum'] / n) ** 0.5,
        'abs_rel_aligned': metric_tot['abs_rel_sum_aligned'] / n,
        'rmse_aligned': (metric_tot['sq_err_sum_aligned'] / n) ** 0.5,
        'd1': metric_tot['d1_sum'] / n,
        'd2': metric_tot['d2_sum'] / n,
        'd3': metric_tot['d3_sum'] / n,
    }
    
    print(f" {dataset_name} -> Loss: {results['loss']:.4f} | AbsRel: {results['abs_rel']:.4f} | RMSE: {results['rmse']:.4f}")
    print(f" {dataset_name} -> AbsRel_aligned: {results['abs_rel_aligned']:.4f} | RMSE_aligned: {results['rmse_aligned']:.4f}")
    print(f" δ1: {results['d1']*100:.2f}% | δ2: {results['d2']*100:.2f}% | δ3: {results['d3']*100:.2f}%")
    return results


def build_flat_region_mask(gt_depth, edge_threshold_q=0.75, min_depth=1e-3, max_depth=200.0):
    """
    只选择深度梯度小（平坦区域）且有效的像素用于绝对尺度监督
    """
    valid = ((gt_depth > min_depth) & (gt_depth < max_depth)).float()
    
    # 计算深度梯度幅值
    gmag = grad_mag(gt_depth)  # [B,1,H,W]
    
    # 对每张图，找出梯度较小的平坦区域（低于 q 分位数）
    B = gt_depth.shape[0]
    flat_mask = torch.zeros_like(valid)
    for b in range(B):
        v = valid[b] > 0.5
        vals = gmag[b][v]
        if vals.numel() < 32:
            continue
        thr = torch.quantile(vals, edge_threshold_q)
        flat_mask[b] = (gmag[b] < thr).float() * valid[b]
    
    # 再做腐蚀，确保远离边缘
    flat_mask = erode_valid_mask(flat_mask, k=5)
    return flat_mask


def save_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, 'history.json')
    csv_path = os.path.join(save_dir, 'history.csv')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if history:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


def _smooth_curve(values, alpha=0.35):
    if len(values) == 0:
        return []
    smoothed = [float(values[0])]
    for v in values[1:]:
        smoothed.append(alpha * float(v) + (1 - alpha) * smoothed[-1])
    return smoothed


def save_loss_curves(history, save_dir, exp_name):
    if not history:
        return

    os.makedirs(save_dir, exist_ok=True)

    epochs = [item['epoch'] for item in history]
    train_loss = [item['train_loss'] for item in history]
    val_loss_mean = [item['val_loss_mean'] for item in history]
    val_loss_grasp = [item['val_loss_grasp'] for item in history]
    val_loss_dreds = [item['val_loss_dreds'] for item in history]
    rmse_aligned_mean = [item['rmse_aligned_mean'] for item in history]

    best_loss_idx = int(np.argmin(val_loss_mean))
    best_rmse_idx = int(np.argmin(rmse_aligned_mean))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ax1 = axes[0]
    ax1.plot(epochs, train_loss, color='#1f77b4', linewidth=2.2, label='Train Loss')
    ax1.plot(epochs, _smooth_curve(train_loss), color='#1f77b4', linewidth=1.4, linestyle='--', alpha=0.75, label='Train Loss (EMA)')
    ax1.plot(epochs, val_loss_mean, color='#d62728', linewidth=2.2, label='Val Loss (Mean)')
    ax1.plot(epochs, _smooth_curve(val_loss_mean), color='#d62728', linewidth=1.4, linestyle='--', alpha=0.75, label='Val Loss (Mean, EMA)')
    ax1.plot(epochs, val_loss_grasp, color='#ff9896', linewidth=1.2, alpha=0.9, label='Val Loss (GraspNet)')
    ax1.plot(epochs, val_loss_dreds, color='#c49c94', linewidth=1.2, alpha=0.9, label='Val Loss (DREDS)')
    ax1.scatter([epochs[best_loss_idx]], [val_loss_mean[best_loss_idx]], color='#2ca02c', s=50, zorder=5, label=f'Best Val @ Epoch {epochs[best_loss_idx]}')
    ax1.set_title(f'{exp_name} Loss Convergence', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8, frameon=True)

    ax2 = axes[1]
    ax2.plot(epochs, rmse_aligned_mean, color='#9467bd', linewidth=2.2, label='Val RMSE Aligned (Mean)')
    ax2.plot(epochs, _smooth_curve(rmse_aligned_mean), color='#9467bd', linewidth=1.4, linestyle='--', alpha=0.75, label='Val RMSE Aligned (EMA)')
    ax2.scatter([epochs[best_rmse_idx]], [rmse_aligned_mean[best_rmse_idx]], color='#2ca02c', s=50, zorder=5, label=f'Best RMSE @ Epoch {epochs[best_rmse_idx]}')
    ax2.set_title(f'{exp_name} Validation Accuracy Trend', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.legend(fontsize=8, frameon=True)

    fig.suptitle('Training Dynamics for Ablation Study', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, 'training_curves.png'), bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'training_curves.pdf'), bbox_inches='tight')
    plt.close(fig)

    # 单独导出论文更常用的 loss 收敛图：仅训练/验证主曲线，版式更简洁
    fig_loss, ax = plt.subplots(figsize=(8.2, 5.6), dpi=300)
    ax.plot(epochs, train_loss, color='#1f77b4', linewidth=2.6, label='Training Loss')
    ax.plot(epochs, _smooth_curve(train_loss), color='#1f77b4', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.plot(epochs, val_loss_mean, color='#d62728', linewidth=2.6, label='Validation Loss')
    ax.plot(epochs, _smooth_curve(val_loss_mean), color='#d62728', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.scatter([epochs[best_loss_idx]], [val_loss_mean[best_loss_idx]], color='#2ca02c', s=60, zorder=5)
    ax.annotate(
        f'Best Val: {val_loss_mean[best_loss_idx]:.4f}\nEpoch {epochs[best_loss_idx]}',
        xy=(epochs[best_loss_idx], val_loss_mean[best_loss_idx]),
        xytext=(12, 12),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb', alpha=0.95),
    )
    ax.set_title(f'{exp_name} Convergence Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.35)
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(save_dir, 'loss_convergence_paper.png'), bbox_inches='tight')
    fig_loss.savefig(os.path.join(save_dir, 'loss_convergence_paper.pdf'), bbox_inches='tight')
    plt.close(fig_loss)


def save_visualization(rgb, gt, pred, content_mask, epoch, save_dir='demo_results'):
    """保存训练过程中的图片对比"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 取 Batch 中的第一张图
    img = rgb[0].permute(1, 2, 0).cpu().numpy()
    # 反归一化以便显示
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    gt_d = gt[0, 0].detach().cpu().numpy()
    pred_d = pred[0, 0].detach().cpu().numpy()
    vis_mask = content_mask[0, 0].detach().cpu().numpy() > 0.5

    gt_d_vis = gt_d.copy()
    pred_d_vis = pred_d.copy()
    gt_d_vis[~vis_mask] = np.nan
    pred_d_vis[~vis_mask] = np.nan

    valid_gt = gt_d_vis[np.isfinite(gt_d_vis) & (gt_d_vis > 0.001)]
    if valid_gt.size == 0:
        print(f"\n[Epoch {epoch} Diagnosis] 当前样本无有效可视化区域，跳过保存。")
        return
    # 设定统一的显示范围
    vmin = valid_gt.min()
    vmax = valid_gt.max()
    print(f"\n[Epoch {epoch} Diagnosis]")
    print(f"  GT Range:   Min={valid_gt.min():.4f}m, Max={valid_gt.max():.4f}m, Mean={valid_gt.mean():.4f}m")
    pred_valid = pred_d_vis[np.isfinite(pred_d_vis)]
    print(f"  Pred Range: Min={pred_valid.min():.4f}m, Max={pred_valid.max():.4f}m, Mean={pred_valid.mean():.4f}m")

    plt.figure(figsize=(25, 15))
    plt.subplot(2, 2, 1);
    plt.title("RGB");
    plt.imshow(img);
    plt.axis('off')

    # 使用统一的 vmin/vmax
    plt.subplot(2, 2, 2);
    plt.title("GT Depth");
    im_gt=plt.imshow(gt_d_vis, cmap='Spectral', vmin=vmin, vmax=vmax);
    plt.axis('off')
    plt.colorbar(im_gt, fraction=0.046, pad=0.04, label='Meters')

    plt.subplot(2, 2, 3);
    plt.title(f"Pred Depth (E{epoch})");
    im_pred=plt.imshow(pred_d_vis, cmap='Spectral', vmin=vmin, vmax=vmax);
    plt.axis('off')
    plt.colorbar(im_pred, fraction=0.046, pad=0.04, label='Meters')

    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.close()


if __name__ == '__main__':
    train()

    
