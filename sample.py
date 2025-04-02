import os
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.unet import UNet
from models.ddpm import DDPM
from utils import save_images, save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser(description="从DDPM模型生成样本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./samples", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=16, help="样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--img_size", type=int, default=64, help="图像尺寸")
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散时间步")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化模型
    unet = UNet(in_channels=3, out_channels=3, time_dim=256, device=device).to(device)
    ddpm = DDPM(unet, timesteps=args.timesteps, device=device).to(device)
    
    # 加载模型权重
    ddpm = load_model(ddpm, args.model_path, device)
    
    # 生成样本
    print("Generating samples...")
    ddpm.eval()
    
    # 计算批次数
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    all_samples = []
    
    # 按批次生成样本
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        with torch.no_grad():
            batch_samples = ddpm.sample(batch_size, args.img_size)
            all_samples.append(batch_samples)
    
    # 合并所有样本
    samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    
    # 保存网格图像
    grid_path = os.path.join(args.output_dir, "sample_grid.png")
    save_images(samples, grid_path, nrow=int(args.num_samples**0.5))
    
    # 保存单独的图像
    for i, sample in enumerate(samples):
        sample_path = os.path.join(args.output_dir, f"sample_{i+1}.png")
        save_images(sample.unsqueeze(0), sample_path)
    
    print(f"生成了 {args.num_samples} 个样本并保存到 {args.output_dir}")

if __name__ == "__main__":
    main()
