from PIL import Image
import torch


def letterbox_resize(img, target_size, fill_value=0, method=Image.BILINEAR):
    """
    img: PIL Image
    target_size: (width, height), e.g. (640, 640)
    fill_value: 填充的颜色/数值，RGB通常是0或114，Depth必须是0
    method: 插值方法
    """
    w, h = img.size
    target_w, target_h = target_size
    
    # 1. 计算缩放比例，取最小的那个，保证图片能完整塞进去
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 2. 缩放图片
    img_resized = img.resize((new_w, new_h), method)
    
    # 3. 创建一张全黑/全0的底图
    if img.mode == 'RGB':
        new_img = Image.new('RGB', (target_w, target_h), (fill_value, fill_value, fill_value))
    else: # 深度图 (I;16) 或 (L)
        new_img = Image.new(img.mode, (target_w, target_h), fill_value)
        
    # 4. 把缩放后的图片粘贴到左上角 (0, 0)
    new_img.paste(img_resized, (0, 0))
    
    return new_img



def compute_metrics(pred, gt, min_depth=1e-3, max_depth=2.5, valid_mask=None):
    """
    计算标准的深度评估指标：AbsRel, RMSE, delta < 1.25, 1.25^2, 1.25^3
    """
    # 默认有效区域由 GT 深度范围决定；若外部提供 valid_mask，则与范围掩码取交集
    range_mask = (gt > min_depth) & (gt < max_depth)
    if valid_mask is None:
        valid_mask = range_mask
    else:
        valid_mask = valid_mask.bool() & range_mask
    # 函数返回统计量（sum/count），最后统一算最终指标, 避免指标虚高
    n = valid_mask.sum().item()
    if n == 0:
        return None
    pred = pred[valid_mask].clamp(min=min_depth, max=max_depth)
    gt = gt[valid_mask]
    
    abs_rel_sum = torch.sum(torch.abs(gt - pred) / gt).item()
    sq_err_sum  = torch.sum((gt - pred) ** 2).item()
    ratio = torch.max(pred / gt, gt / pred)
    d1_sum = (ratio < 1.25).sum().item()
    d2_sum = (ratio < 1.25**2).sum().item()
    d3_sum = (ratio < 1.25**3).sum().item()
    return {
        "n": n,
        "abs_rel_sum": abs_rel_sum,
        "sq_err_sum": sq_err_sum,
        "d1_sum": d1_sum,
        "d2_sum": d2_sum,
        "d3_sum": d3_sum
    }





    