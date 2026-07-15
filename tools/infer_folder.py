import depth_pro
from PIL import Image
import torch
import numpy as np
import time
import os
import glob    # 用glob查找文件目录和文件，并将搜索的到的结果返回到一个列表中
from tqdm import tqdm  # 进度条库
import argparse
import matplotlib.pyplot as plt
import cv2

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

# 设置CUDA优化配置
def setup_cuda_optimization():
    if torch.cuda.is_available():
        # 启用cudnn自动调优找到最佳卷积算法
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        # 关闭确定性以提高性能（牺牲可重复性）
        torch.backends.cudnn.deterministic = True
        # 禁用CUDA图缓存（对动态输入尺寸有用）
        torch.backends.cudnn.enabled = True
        print("CUDA优化配置已启用")

# 设置CUDA优化
# 这个优化可以极大增加推理速度
setup_cuda_optimization()

# 解析命令行参数
parser = argparse.ArgumentParser(description='Depth Pro批量推理优化版')
parser.add_argument('--input_folder', type=str, default='../../demo_data/rgb', help='输入文件夹路径')
parser.add_argument('--output_folder', type=str, default='../../demo_data/synthetic_depth', help='输出文件夹路径')
parser.add_argument('--precision', type=str, choices=['fp32', 'fp16'], default='fp16', help='推理精度 (fp32或fp16)')
parser.add_argument('--use_jit', action='store_true', default=True, help='使用TorchScript加速')
parser.add_argument('--compile_mode', type=str, choices=['trace', 'script'], default='trace', help='JIT编译模式')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"推理精度: {args.precision}")



def apply_jet_colormap(image_data):
    # 注意：jet_r是反转的jet，深色表示近（值小），浅色表示远（值大）
    # 将数据归一化到[0, 1]
    vmin = np.min(image_data)
    vmax = np.max(image_data)
    normalized = (image_data - vmin) / (vmax - vmin + 1e-8)

    colormap = plt.get_cmap('Spectral_r')
    depth_colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
    
    return depth_colored

# 批量处理函数（优化版）
def process_folder(input_folder, output_folder, use_fp16=False):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 获取文件夹中的所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        print(f"在 {input_folder} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    total_inference_time = 0.0
    processed_images = 0

    # 使用进度条显示处理进度
    with tqdm(total=len(image_files), desc="处理进度") as pbar:
        for input_rgb in image_files:
            try:
                # 加载和预处理图像
                image, _, f_px = depth_pro.load_rgb(input_rgb)
                image = transform(image)
                image = image.to(device)
                
                # FP16半精度优化
                if use_fp16 and torch.cuda.is_available():
                    image = image.half()

                # 计算推理时间
                torch.cuda.synchronize()  # 等待之前所有操作完成
                start = time.time()

                # 运行推理
                with torch.no_grad():  # 禁用梯度计算以加速
                    prediction = model.infer(image, f_px=f_px)

                torch.cuda.synchronize()  # 强制CPU等待GPU完成这次推理
                end = time.time()

                inference_time = end - start
                total_inference_time += inference_time
                processed_images += 1

                # 获取深度图和焦距
                depth = prediction["depth"].detach().cpu().numpy().squeeze()  # 深度值（米）
                inverse_depth = 1 / depth
                # 获取单通道16位png深度图
                depth_mm = (depth * 1000).astype(np.uint16)

                filename_stem, _ = os.path.splitext(os.path.basename(input_rgb))
                
                png_16bit_path = os.path.join(output_folder, f"{filename_stem}.png")
                cv2.imwrite(png_16bit_path, depth_mm)

                postfix_info = {
                    '当前图像': os.path.basename(input_rgb),
                    '时间': f'{inference_time:.4f}s'
                }
                # 判断 f_px 是否有值
                if f_px is not None:
                    # 如果有值，就添加到字典里
                    postfix_info['焦距(EXIF)'] = f'{f_px:.2f}px'
                else:
                    # 如果没有值，就去 prediction 里找模型估计的值
                    if "focallength_px" in prediction and prediction["focallength_px"] is not None:
                        estimated_f_px = prediction["focallength_px"].detach().cpu().item()
                        postfix_info['焦距(估计)'] = f'{estimated_f_px:.2f}px'
                # 更新进度条
                pbar.set_postfix(postfix_info)
                pbar.update(1)
                print(f"深度范围: {depth.min():.2f} 米 到 {depth.max():.2f} 米")

            except Exception as e:
                print(f"处理文件 {os.path.basename(input_rgb)} 时出错: {str(e)}")
                pbar.update(1)

    # 计算并显示平均推理时间
    if processed_images > 0:
        average_time = total_inference_time / processed_images
        print(f"成功处理 {processed_images} 张图像")
        print(f"平均推理时间: {average_time:.4f} 秒/张")
        print(f"总推理时间: {total_inference_time:.4f} 秒")
        print(f"输出保存在: {output_folder}")
    else:
        print("没有成功处理任何图像")


if __name__ == "__main__":
    # 加载模型和预处理变换
    print("加载模型中...")
    model, transform = depth_pro.create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,)
    model = torch.compile(model, mode="reduce-overhead")
    
    # 应用精度设置
    use_fp16 = args.precision == 'fp16' and torch.cuda.is_available()
    if use_fp16:
        model = model.half()  # 转换模型到FP16精度
    
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 开始批量处理
    start_total = time.time()
    process_folder(args.input_folder, args.output_folder, use_fp16=use_fp16)
    end_total = time.time()
    print(f"总耗时（包括加载模型和图像处理）: {end_total - start_total:.4f} 秒")
    print("\n使用说明:")
    print("- 使用 --precision fp16 启用半精度推理（默认启用）")
    print("- 使用 --input_folder 和 --output_folder 指定路径")
    print("- 例如: python infer_folder.py --input_folder ../data/新测试集 --output_folder ../data/新测试集_results --precision fp16")
    print("- 如需禁用特定优化: python infer_folder.py --use_jit False --precision fp32")