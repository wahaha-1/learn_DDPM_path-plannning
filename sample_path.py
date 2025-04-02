import os
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.unet import PathUNet
from models.ddpm import PathDDPM
from dataset import PathPlanningDataset
from utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description="从路径规划扩散模型生成路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="测试数据目录")
    parser.add_argument("--output_dir", type=str, default="./samples", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--path_length", type=int, default=64, help="路径长度")
    parser.add_argument("--num_samples", type=int, default=10, help="生成的样本数")
    parser.add_argument("--guide_scale", type=float, default=2.0, help="引导比例")
    parser.add_argument("--collision_avoid", action="store_true", help="启用碰撞避免")
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
    
    # 加载数据集
    test_dataset = PathPlanningDataset(args.data_dir, max_path_length=args.path_length)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # 初始化模型
    model = PathUNet(
        path_channels=2, 
        time_dim=256, 
        condition_channels=3, 
        path_length=args.path_length, 
        device=device
    ).to(device)
    
    ddpm = PathDDPM(
        model=model, 
        timesteps=args.timesteps, 
        device=device
    ).to(device)
    
    # 加载模型权重
    print(f"Loading model from {args.model_path}...")
    ddpm = load_model(ddpm, args.model_path, device)
    
    # 生成路径
    print("Generating paths...")
    ddpm.eval()
    
    # 限制生成的样本数
    sample_count = min(args.num_samples, len(test_dataset))
    batch_count = 0
    
    for i, batch in enumerate(test_loader):
        if i * args.batch_size >= sample_count:
            break
            
        # 获取数据
        condition = batch['condition'].to(device)
        
        # 生成路径
        with torch.no_grad():
            generated_paths = ddpm.sample(
                batch_size=condition.shape[0],
                path_length=args.path_length,
                condition=condition,
                guide_scale=args.guide_scale,
                collision_avoid=args.collision_avoid
            )
        
        # 可视化并保存
        test_dataset.visualize_batch(
            {key: val.cpu() for key, val in batch.items()},
            predictions=generated_paths.cpu(),
            save_path=os.path.join(args.output_dir, f"sample_batch_{i+1}.png")
        )
        
        # 保存每个样本
        for j in range(condition.shape[0]):
            if batch_count * args.batch_size + j >= sample_count:
                break
                
            sample_idx = batch_count * args.batch_size + j
            
            # 获取单个样本
            single_batch = {key: val[j:j+1].cpu() for key, val in batch.items()}
            single_prediction = generated_paths[j:j+1].cpu()
            
            # 保存单个样本
            test_dataset.visualize_batch(
                single_batch,
                predictions=single_prediction,
                save_path=os.path.join(args.output_dir, f"sample_{sample_idx+1}.png")
            )
        
        batch_count += 1
    
    print(f"生成了 {sample_count} 个路径样本并保存到 {args.output_dir}")

if __name__ == "__main__":
    main()
