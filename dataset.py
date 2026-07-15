import os
# 必须开启此环境变量，否则 OpenCV 无法读取 EXR 深度图
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ==================相机内参归一化工具==================
# D415 官方典型内参（RGB 传感器在 640px 宽度下）
# 若 DREDS 数据集另有说明，请修改此常量
D415_FX_DEFAULT = 605.0
D415_IMG_W_DEFAULT = 640


def load_d435_intrinsics(graspnet_root):
    """
    从 GraspNet 数据集读取 D435 的真实相机内参。
    自动扫描第一个找到的 camK.npy，同时尝试读取对应 RGB 图像尺寸。
    返回: (fx, img_w) -- 内参焦距和对应的图像宽度
    """
    import glob as _glob
    cam_k_files = _glob.glob(
        os.path.join(graspnet_root, 'train_*', 'scene_*', 'realsense', 'camK.npy')
    )
    if not cam_k_files:
        print("⚠️ 未找到 camK.npy，使用 D435 默认内参 fx=383, img_w=1280")
        return 383.0, 1280

    cam_k_path = sorted(cam_k_files)[0]
    K = np.load(cam_k_path)   # [3,3]
    fx = float(K[0, 0])

    # 尝试读取同目录下第一张 RGB 图推断原始宽度
    rgb_dir = os.path.join(os.path.dirname(cam_k_path), 'rgb')
    rgb_files = _glob.glob(os.path.join(rgb_dir, '*.png'))
    if rgb_files:
        sample_img = cv2.imread(sorted(rgb_files)[0], cv2.IMREAD_COLOR)
        img_w = sample_img.shape[1] if sample_img is not None else 1280
    else:
        img_w = 1280  # GraspNet 默认 1280×720

    print(f"✅ D435 参考内参: fx={fx:.2f}，原始图像宽度={img_w}px")
    return fx, img_w


def _center_crop_or_pad(rgb, depth, target_h, target_w, content_mask=None):
    """对 rgb [H,W,3] 和 depth [H,W] 做中心裁剪或 pad 至 (target_h, target_w)。
    content_mask: [H,W]，1 表示真实内容区域，0 表示后续 pad 区域。
    """
    h, w = rgb.shape[:2]
    if content_mask is None:
        content_mask = np.ones((h, w), dtype=np.uint8)
    # 裁剪
    if h > target_h:
        y0 = (h - target_h) // 2
        rgb = rgb[y0:y0 + target_h]
        depth = depth[y0:y0 + target_h]
        content_mask = content_mask[y0:y0 + target_h]
    if w > target_w:
        x0 = (w - target_w) // 2
        rgb = rgb[:, x0:x0 + target_w]
        depth = depth[:, x0:x0 + target_w]
        content_mask = content_mask[:, x0:x0 + target_w]
    # pad
    h, w = rgb.shape[:2]
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    if pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0:
        rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_REPLICATE)
        depth = cv2.copyMakeBorder(depth, pad_top, pad_bot, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=0.0)
        content_mask = cv2.copyMakeBorder(content_mask, pad_top, pad_bot, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=0)
    return rgb, depth, content_mask


def canonical_resize_to_ref_camera(rgb_img, depth_img,
                                    src_fx, src_img_w,
                                    ref_fx, ref_img_w,
                                    target_h, target_w):
    # 1. 目标参考相机的归一化焦距 (像素占比)
    f_norm_ref = ref_fx / ref_img_w
    
    # 2. 为了在 target_w (640) 下保持相同的视场，期望的焦距应该是：
    target_f = f_norm_ref * target_w
    
    # 3. 计算对原图的严格物理缩放比例 (目标焦距 / 源相机真实焦距)
    scale = target_f / src_fx
    
    h, w = rgb_img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # 使用线性插值处理RGB，最近邻插值处理深度
    rgb_scaled  = cv2.resize(rgb_img,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    depth_scaled = cv2.resize(depth_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    content_mask = np.ones((new_h, new_w), dtype=np.uint8)

    # 裁剪或 Padding 填补到 360x640 
    rgb_out, depth_out, content_mask = _center_crop_or_pad(rgb_scaled, depth_scaled, target_h, target_w, content_mask)
    return rgb_out, depth_out, content_mask
# ======================================================


class GraspNetDepthDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=(360, 640), max_missing_ratio=0.2, sample_size=2560, overfit_one=False):
        """
        GraspNet-1B 单目深度估计专用数据集
        :param root_dir: graspnet_dataset 的根目录
        :param target_size: (H, W) 网络输入尺寸，默认 (364, 644)，必须是 14 的倍数
        :param max_missing_ratio: 深度图最大允许的无效空洞比例 (默认 20%)
        :param sample_size: 随机抽取的小样本数量 (如 2000)。若为 None，则使用全部有效数据。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_h, self.target_w = target_size
        self.max_missing_ratio = max_missing_ratio

        self.overfit_one = overfit_one
        
        # DINOv2 标准归一化参数
        self.mean =[0.485, 0.456, 0.406]
        self.std =[0.229, 0.224, 0.225]

        # 读取 D435 真实内参，作为正则相机基准（供 DREDS 等其他数据集归一化使用）
        self.ref_fx, self.ref_img_w = load_d435_intrinsics(root_dir)

        # 扫描所有可能的文件路径
        print("🔍 正在扫描数据集目录...")
        all_rgb_paths, all_depth_paths = self._scan_dataset()
        print(f"共发现 {len(all_rgb_paths)} 组 RGB-Depth 数据对。")
        '''
        # 多线程极速过滤坏样本 # 占用内存过大，暂时关掉
        print(f"🧹 正在依据空洞阈值 (>{max_missing_ratio*100}%) 过滤劣质深度图...")
        self.valid_pairs = self._filter_bad_samples_multithreaded(all_rgb_paths, all_depth_paths)
        '''
        self.valid_pairs = list(zip(all_rgb_paths, all_depth_paths))
        # 随机小样本抽样 (如果设置了 sample_size)
        if sample_size is not None:
            if sample_size > len(self.valid_pairs):
                print(f"⚠️ 警告：请求的样本数 ({sample_size}) 大于有效样本总数 ({len(self.valid_pairs)})，将使用全部有效样本。")
            else:
                if self.overfit_one and sample_size == 1:
                    self.valid_pairs = [self.valid_pairs[23000]]
                    print("单张过拟合实验模式")
                else:
                    # 保证验证集抽样的绝对确定性
                    if self.mode == 'val':
                        random.seed(42)
                    self.valid_pairs = random.sample(self.valid_pairs, sample_size)
                    random.seed()
                    print(f"🎲 已抽取 {sample_size} 张图像。")

    def _scan_dataset(self):
        rgb_paths = []
        depth_paths = []
        train_folders =['train_1', 'train_2', 'train_3', 'train_4']
        
        for train_dir in train_folders:
            train_path = os.path.join(self.root_dir, train_dir)
            if not os.path.exists(train_path):
                continue
                
            # 遍历 scene_0000 到 scene_0099
            for scene_folder in sorted(os.listdir(train_path)):
                if not scene_folder.startswith('scene_'):
                    continue

                # 提取 Scene ID (例如 'scene_0085' -> 85)
                scene_id = int(scene_folder.split('_')[1])
                
                # 按照 9:1 划分：0~89 为 train，90~99 为 val
                if self.mode == 'train' and scene_id >= 90:
                    continue
                if self.mode == 'val' and scene_id < 90:
                    continue
                
                # 只取 realsense 文件夹
                rs_rgb_dir = os.path.join(train_path, scene_folder, 'realsense', 'rgb')
                rs_depth_dir = os.path.join(train_path, scene_folder, 'realsense', 'depth')    # 大模型推理出的尖锐边缘深度图depth
                
                if not os.path.exists(rs_rgb_dir) or not os.path.exists(rs_depth_dir):
                    continue
                    
                # 寻找所有 png
                img_files = sorted(glob.glob(os.path.join(rs_rgb_dir, '*.png')))
                for rgb_p in img_files:
                    # 推导对应的 depth 路径
                    filename = os.path.basename(rgb_p)
                    depth_p = os.path.join(rs_depth_dir, filename)
                    
                    if os.path.exists(depth_p):
                        rgb_paths.append(rgb_p)
                        depth_paths.append(depth_p)
                        
        return rgb_paths, depth_paths

    def _check_single_image(self, rgb_path, depth_path):
        """检查单张深度图的有效性，返回合格的路径对，否则返回 None"""
        try:
            # cv2.IMREAD_UNCHANGED 保证读取真实的 16-bit 深度数据
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return None
            
            # GraspNet 的 Realsense 深度图中，0 通常代表无效/没扫到的像素
            zero_pixels = np.sum(depth == 0)
            total_pixels = depth.size
            missing_ratio = zero_pixels / total_pixels
            
            if missing_ratio > self.max_missing_ratio:
                return None  # 坏样本，丢弃
                
            return (rgb_path, depth_path)
        except Exception:
            return None

    # 该函数读取所有图片，过于占内存
    def _filter_bad_samples_multithreaded(self, rgb_paths, depth_paths):
        valid_pairs =[]
        discarded_count = 0
        
        # 使用多线程加速文件 I/O 读取验证 (大大缩短初始化时间)
        with ThreadPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            futures =[executor.submit(self._check_single_image, r, d) for r, d in zip(rgb_paths, depth_paths)]
            
            # 使用 tqdm 显示进度条
            for future in tqdm(futures, desc="过滤进度"):
                result = future.result()
                if result is not None:
                    valid_pairs.append(result)
                else:
                    discarded_count += 1
                    
        print("\n" + "="*50)
        print("📊 数据集过滤报告:")
        print(f"❌ 过滤掉的坏样本 (缺损>{self.max_missing_ratio*100}%): {discarded_count} 张")
        print(f"✅ 剩下的好样本 (用于训练): {len(valid_pairs)} 张")
        print("="*50 + "\n")
        
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.valid_pairs[idx]
        
        # 读取图像 (OpenCV 默认 BGR，需转为 RGB)
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 读取深度图 (uint16格式)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # 转换为物理距离 (GraspNet 深度单位是毫米，转换为米)
        depth_img = depth_img.astype(np.float32) / 1000.0
        
        if self.mode == 'train' and (not self.overfit_one):
            base_mask = np.ones_like(depth_img, dtype=np.uint8)
            rgb_global, depth_global, content_mask = augment_train(
                rgb_img, depth_img,
                target_size=(self.target_h, self.target_w),
                mean=self.mean, std=self.std,
                min_depth=1e-3, max_depth=3,
                multi_scales=None,  # 先设None，避免batch维度不一致
                extra_mask=base_mask,
                return_mask=True
            )
        else:
            base_mask = np.ones_like(depth_img, dtype=np.uint8)
            rgb_global, depth_global, content_mask = preprocess_eval(
                rgb_img, depth_img,
                target_size=(self.target_h, self.target_w),
                mean=self.mean, std=self.std,
                min_depth=1e-3, max_depth=3,
                extra_mask=base_mask,
                return_mask=True
            )
        return rgb_global, depth_global, content_mask

        
class DREDS_CatKnown_Dataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=(360, 640), max_missing_ratio=0.2, sample_size=None,
                 overfit_one=False, exr_backend='openexr', debug_exr=False,
                 ref_fx=None, ref_img_w=None):
        """
        DREDS-CatKnown 模拟数据集预处理
        :param root_dir: DREDS 数据集的根目录 (包含 train_part0 ~ train_part4)
        :param mode: 'train' 或 'val' (9:1 场景隔离划分)
        :param target_size: (H, W) 网络输入尺寸
        :param ref_fx: D435 参考焦距（从 GraspNetDepthDataset.ref_fx 传入）
        :param ref_img_w: D435 参考图像宽度（从 GraspNetDepthDataset.ref_img_w 传入）
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_h, self.target_w = target_size
        self.max_missing_ratio = max_missing_ratio

        self.overfit_one = overfit_one
        
        # DINOv2 归一化参数
        self.mean =[0.485, 0.456, 0.406]
        self.std =[0.229, 0.224, 0.225]

        # D435 参考内参（正则相机基准）
        self.ref_fx   = ref_fx   if ref_fx   is not None else 383.0
        self.ref_img_w = ref_img_w if ref_img_w is not None else 1280
        # D415 仿真内参（源相机）
        self.src_fx    = D415_FX_DEFAULT
        self.src_img_w = D415_IMG_W_DEFAULT
        print(f"📷 DREDS 内参归一化: D415(fx={self.src_fx}, w={self.src_img_w}) → D435(fx={self.ref_fx:.2f}, w={self.ref_img_w})"
              f"  缩放比例≈{((self.ref_fx/self.ref_img_w)/(self.src_fx/self.src_img_w)):.3f}")

        print(f"🔍 正在扫描 DREDS 数据集目录 (模式: {self.mode.upper()})...")
        all_rgb_paths, all_depth_paths = self._scan_and_split_dataset()
        print(f"共为 {self.mode.upper()} 阶段分配了 {len(all_rgb_paths)} 组候选数据对。")
        '''
        print(f"🧹 正在多线程过滤无效深度图 (空洞阈值 {self.max_missing_ratio*100}%)...")
        self.valid_pairs = self._filter_bad_samples_multithreaded(all_rgb_paths, all_depth_paths)
        '''
        self.valid_pairs = list(zip(all_rgb_paths, all_depth_paths))
        if sample_size is not None:
            if sample_size > len(self.valid_pairs):
                print(f"⚠️ 警告：请求样本数大于总数，使用全部 {len(self.valid_pairs)} 张。")
            else:
                if self.overfit_one and sample_size == 1:
                    self.valid_pairs = [self.valid_pairs[0]]
                    print("单张过拟合实验模式")
                else:
                    # 保证验证集抽样的绝对确定性
                    if self.mode == 'val':
                        random.seed(42)
                    self.valid_pairs = random.sample(self.valid_pairs, sample_size)
                    random.seed()
                    print(f"🎲 已抽取 {sample_size} 张图像。")

    
    
    def _scan_and_split_dataset(self):
        """扫描所有 part，汇总所有 scene，并进行 9:1 的硬性物理隔离"""
        all_scene_dirs =[]
        
        # 收集所有的 scene 文件夹
        for i in range(5):
            part_name = f"train_part{i}"
            sub_part_name = f"part{i}"
            part_dir = os.path.join(self.root_dir, part_name, sub_part_name)
            
            if not os.path.exists(part_dir):
                continue
                
            # 收集 00001 到 01513 等序号文件夹
            scene_folders = sorted(os.listdir(part_dir))
            for sf in scene_folders:
                scene_path = os.path.join(part_dir, sf)
                if os.path.isdir(scene_path):
                    all_scene_dirs.append(scene_path)
                    
        # 全局排序以保证每次运行的划分绝对一致
        all_scene_dirs = sorted(all_scene_dirs)
        
        # 场景隔离划分 (杜绝同场景相邻帧泄露到验证集)
        split_idx = int(len(all_scene_dirs) * 0.99)
        if self.mode == 'train':
            assigned_scenes = all_scene_dirs[:split_idx]
        else:
            assigned_scenes = all_scene_dirs[split_idx:]
            
        # 在分配好的场景中收集所有的 0000~0029 数据对
        rgb_paths = []
        depth_paths =[]
        
        for scene_dir in assigned_scenes:
            for frame_idx in range(30):
                # 构造前缀 0000, 0001, ..., 0029
                prefix = f"{frame_idx:04d}"
                rgb_p = os.path.join(scene_dir, f"{prefix}_color.jpg")
                depth_p = os.path.join(scene_dir, f"{prefix}_depth_120.exr")
                
                if os.path.exists(rgb_p) and os.path.exists(depth_p):
                    rgb_paths.append(rgb_p)
                    depth_paths.append(depth_p)
                    
        return rgb_paths, depth_paths

    def _check_single_image(self, rgb_path, depth_path):
        """单图校验逻辑，专门针对 EXR 浮点数据"""
        try:
            # 读取 EXR 格式，返回的是 float32 数据
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                return None
                
            # 有些 EXR 读取出来会是 3 通道 (H, W, 3)，取单通道即可
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
                
            # EXR 中的无效像素通常表现为 0, NaN(非数字), 或 Inf(无穷大)
            invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
            invalid_pixels = np.sum(invalid_mask)
            
            if (invalid_pixels / depth.size) > self.max_missing_ratio:
                return None
                
            return (rgb_path, depth_path)
        except Exception:
            return None

    def _filter_bad_samples_multithreaded(self, rgb_paths, depth_paths):
        valid_pairs =[]
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures =[executor.submit(self._check_single_image, r, d) for r, d in zip(rgb_paths, depth_paths)]
            for future in tqdm(futures, desc=f"{self.mode.upper()} 过滤"):
                res = future.result()
                if res:
                    valid_pairs.append(res)
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.valid_pairs[idx]
        
        # 加载 RGB
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_img.shape[:2]
        
        # 加载 EXR 深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(depth_img.shape) == 3:
            depth_img = depth_img[:, :, 0]
            
        # 清洗浮点脏数据：将 NaN 和 Inf 转为 0.0 (后续 Loss 计算时通过 valid_mask > 0 过滤)
        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
        # 把深度图对齐到 rgb 分辨率
        depth_h, depth_w = depth_img.shape[:2]
        if (depth_h, depth_w) != (orig_h, orig_w):
            depth_img = cv2.resize(depth_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # DREDS 深度单位为"米(Meters)"，无需转换。
        # 如果发现尺度差 1000 倍（毫米），取消注释：
        # depth_img = depth_img / 1000.0

        # ★ 正则相机归一化：将 D415 视角图像 resize 到 D435 视角
        #   - 深度值（米）不变，仅做空间缩放以对齐 FOV
        #   - 输出已裁剪/pad 到 (target_h, target_w)
        rgb_img, depth_img, content_mask = canonical_resize_to_ref_camera(
            rgb_img, depth_img,
            src_fx=self.src_fx, src_img_w=orig_w,   # 用实际图像宽度，更准确
            ref_fx=self.ref_fx, ref_img_w=self.ref_img_w,
            target_h=self.target_h, target_w=self.target_w
        )
        
        if self.mode == 'train' and (not self.overfit_one):
            rgb_global, depth_global, content_mask = augment_train(
                rgb_img, depth_img,
                target_size=(self.target_h, self.target_w),
                mean=self.mean, std=self.std,
                min_depth=1e-3, max_depth=3,
                multi_scales=None,  # 先设None，避免batch维度不一致
                extra_mask=content_mask,
                return_mask=True
            )
        else:
            rgb_global, depth_global, content_mask = preprocess_eval(
                rgb_img, depth_img,
                target_size=(self.target_h, self.target_w),
                mean=self.mean, std=self.std,
                min_depth=1e-3, max_depth=3,
                extra_mask=content_mask,
                return_mask=True
            )
        return rgb_global, depth_global, content_mask
        

# ==================工具函数=================

def _clamp_img_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

    
def _random_resized_crop_params(h, w, target_size, scale=(0.6, 1.0), trials=10):
    target_h, target_w = target_size
    target_ratio = target_w / target_h  # 例如 640/360 = 1.777
    
    # 1. 计算在当前原图中最能框出的、符合目标长宽比的最大外接矩形
    if w / h > target_ratio:
        # 原图比目标更宽，以高度为基准
        max_h = h
        max_w = int(max_h * target_ratio)
    else:
        # 原图比目标更高，以宽度为基准
        max_w = w
        max_h = int(max_w / target_ratio)
        
    max_area = max_h * max_w
    
    # 2. 在最大符合比例的矩形基础上，做随机缩放面积
    for _ in range(trials):
        target_area = max_area * random.uniform(scale[0], scale[1])
        crop_h = int(round(np.sqrt(target_area / target_ratio)))
        crop_w = int(round(crop_h * target_ratio))
        
        # 确保不越界
        if 0 < crop_w <= w and 0 < crop_h <= h:
            y1 = random.randint(0, h - crop_h)
            x1 = random.randint(0, w - crop_w)
            # 返回裁剪参数，以及当前物理视角的缩放比例 (zoom_factor)
            return x1, y1, crop_w, crop_h
            
    # fallback：如果随机失败，直接中心裁剪最大符合比例的区域
    crop_w, crop_h = max_w, max_h
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    return x1, y1, crop_w, crop_h
    
    
def _apply_color_jitter_rgb(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
    # rgb: uint8, [H,W,3], RGB
    img = rgb.astype(np.float32)
    # brightness
    if brightness > 0:
        b = random.uniform(1 - brightness, 1 + brightness)
        img = img * b
    # contrast
    if contrast > 0:
        c = random.uniform(1 - contrast, 1 + contrast)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c + mean
    img = _clamp_img_uint8(img)
    # saturation + hue（转 HSV）
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    if saturation > 0:
        s = random.uniform(1 - saturation, 1 + saturation)
        hsv[..., 1] *= s
    if hue > 0:
        # OpenCV hue范围[0,179]
        h_shift = random.uniform(-hue, hue) * 179.0
        hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img
    
def _apply_gamma_rgb(rgb, gamma_range=(0.9, 1.1)):
    gamma = random.uniform(*gamma_range)
    table = (np.arange(256) / 255.0) ** gamma * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(rgb, table)
    
def _apply_blur_noise_jpeg(rgb, p_blur=0.2, p_noise=0.2, p_jpeg=0.2):
    img = rgb.copy()
    # blur
    if random.random() < p_blur:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # noise
    if random.random() < p_noise:
        sigma = random.uniform(2.0, 8.0)
        noise = np.random.randn(*img.shape) * sigma
        img = _clamp_img_uint8(img.astype(np.float32) + noise)
    # jpeg artifact
    if random.random() < p_jpeg:
        q = random.randint(40, 90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return img
    
def _apply_cutout_rgb(rgb, p=0.2, max_frac=0.2):
    if random.random() > p:
        return rgb
    h, w = rgb.shape[:2]
    cut_w = int(w * random.uniform(0.05, max_frac))
    cut_h = int(h * random.uniform(0.05, max_frac))
    x1 = random.randint(0, max(0, w - cut_w))
    y1 = random.randint(0, max(0, h - cut_h))
    rgb = rgb.copy()
    rgb[y1:y1+cut_h, x1:x1+cut_w, :] = 0
    return rgb
# ----------------------------
# 主增强函数（训练）
# ----------------------------
def augment_train(
    rgb_img, depth_img,
    target_size=(360, 640),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    min_depth=1e-3, max_depth=3,
    # 几何增强
    p_flip=0.5,
    p_rotate=0.3,
    rotate_deg=2.0,
    #crop_scale=(0.6, 1.0),
    #crop_ratio=(0.9, 1.1),
    # 多分辨率（可选）
    multi_scales=None,   # e.g. [(352, 512), (384, 640), (448, 640)]
    # RGB外观增强
    p_color=0.8,
    p_gamma=0.3,
    gamma_range=(0.9, 1.1),
    p_blur_noise_jpeg=0.4,
    p_cutout=0,
    extra_mask=None,
    return_mask=False
):
    """
    rgb_img: uint8 RGB, [H,W,3]
    depth_img: float32 depth(m), [H,W]
    """
    assert rgb_img.ndim == 3 and depth_img.ndim == 2
    h, w = depth_img.shape
    if extra_mask is None:
        extra_mask = np.ones((h, w), dtype=np.uint8)
    else:
        extra_mask = extra_mask.astype(np.uint8)
    # 0) 构造有效mask（很重要）
    valid_mask = np.isfinite(depth_img) & (depth_img > min_depth) & (depth_img < max_depth) & (extra_mask > 0)
    depth = depth_img.copy()
    depth[~valid_mask] = 0.0
    # 1) Random Resized Crop（同步）
    x1, y1, cw, ch = _random_resized_crop_params(h, w, target_size, scale=(0.6, 1.0))
    rgb = rgb_img[y1:y1+ch, x1:x1+cw]
    depth = depth[y1:y1+ch, x1:x1+cw]
    valid_mask = valid_mask[y1:y1+ch, x1:x1+cw]
    # 2) 小角度旋转（同步）
    if random.random() < p_rotate:
        angle = random.uniform(-rotate_deg, rotate_deg)
        rh, rw = depth.shape
        center = (rw * 0.5, rh * 0.5)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rgb = cv2.warpAffine(
            rgb, M, (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        depth = cv2.warpAffine(
            depth, M, (rw, rh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        )
        valid_mask = cv2.warpAffine(
            valid_mask.astype(np.uint8), M, (rw, rh),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        ).astype(bool)
        
    # 3) Resize 到目标尺寸 / 多分辨率
    if multi_scales is not None and len(multi_scales) > 0:
        out_h, out_w = random.choice(multi_scales)  # 训练时随机尺度
    else:
        out_h, out_w = target_size
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    valid_mask = cv2.resize(valid_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    # 4) 水平翻转（同步）
    
    if random.random() < p_flip:
        rgb = cv2.flip(rgb, 1)
        depth = cv2.flip(depth, 1)
        valid_mask = cv2.flip(valid_mask.astype(np.uint8), 1).astype(bool)
    # 5) RGB外观增强（仅 RGB）
    
    if random.random() < p_color:
        rgb = _apply_color_jitter_rgb(rgb, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        
    if random.random() < p_gamma:
        rgb = _apply_gamma_rgb(rgb, gamma_range=gamma_range)
        
    if random.random() < p_blur_noise_jpeg:
        rgb = _apply_blur_noise_jpeg(rgb, p_blur=0.6, p_noise=0.6, p_jpeg=0.6)
    rgb = _apply_cutout_rgb(rgb, p=p_cutout, max_frac=0.2)
    # 6) 转 tensor
    rgb_t = TF.to_tensor(rgb)  # [3,H,W], 0~1
    rgb_t = TF.normalize(rgb_t, mean=mean, std=std)
    depth[~valid_mask] = 0.0
    depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)  # [1,H,W]
    mask_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)  # [1,H,W]
    if return_mask:
        return rgb_t, depth_t, mask_t
    return rgb_t, depth_t
# ----------------------------
# 验证/测试预处理（不要增强）
# ----------------------------
def preprocess_eval(
    rgb_img, depth_img,
    target_size=(360, 640),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    min_depth=1e-3, max_depth=3,
    extra_mask=None,
    return_mask=False
):
    rgb = cv2.resize(rgb_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    if extra_mask is None:
        extra_mask = np.ones(depth_img.shape[:2], dtype=np.uint8)
    extra_mask = cv2.resize(extra_mask.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    valid_mask = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth) & (extra_mask > 0)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    depth[~valid_mask] = 0.0
    rgb_t = TF.normalize(TF.to_tensor(rgb), mean=mean, std=std)
    depth_t = torch.from_numpy(depth).unsqueeze(0).float()
    mask_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0)
    if return_mask:
        return rgb_t, depth_t, mask_t
    return rgb_t, depth_t



def depth_collate_pad(batch, pad_value_rgb=0.0, pad_value_depth=0.0, pad_value_mask=0.0):
    """
    batch item:
      - (rgb, depth) or
      - (rgb, depth, mask)
    rgb:   [3,H,W]
    depth: [1,H,W]
    mask:  [1,H,W] (optional)
    """
    rgbs, depths = [], []
    max_h = max(item[0].shape[-2] for item in batch)
    max_w = max(item[0].shape[-1] for item in batch)
    for rgb, depth in batch:
        h, w = rgb.shape[-2], rgb.shape[-1]
        pad_h, pad_w = max_h - h, max_w - w
        # pad format: (left, right, top, bottom)
        pad = (0, pad_w, 0, pad_h)
        rgbs.append(F.pad(rgb, pad, value=pad_value_rgb))
        depths.append(F.pad(depth, pad, value=pad_value_depth))
        m = torch.ones((1, h, w), dtype=torch.bool)   # 原图区域=1
        m = F.pad(m, pad, value=0)                    # pad区域=0
        # 收集 mask
        if 'masks' not in locals():
            masks = []
        masks.append(m)
        
    rgbs = torch.stack(rgbs, dim=0)
    depths = torch.stack(depths, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return rgbs, depths, masks




    


        