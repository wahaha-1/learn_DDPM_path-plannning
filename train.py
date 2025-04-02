import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNet
from models.ddpm import DDPM
from utils import save_images, get_data_transforms, get_dataloader, save_model

def parse_args():
    parser = argparse.ArgumentParser(description="训练DDPM模型")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--img_size", type=int, default=64, help="图像尺寸")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--sample_interval", type=int, default=10, help="采样间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="保存间隔")
    parser.add_argument("--timesteps", type=int, default=1000, help="扩散时间步")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    transform = get_data_transforms(args.img_size)
    dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 初始化模型
    unet = UNet(in_channels=3, out_channels=3, time_dim=256, device=device).to(device)
    ddpm = DDPM(unet, timesteps=args.timesteps, device=device).to(device)
    
    # 配置优化器
    optimizer = ddpm.configure_optimizers(lr=args.lr)
    
    # 设置TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    for epoch in range(args.epochs):
        ddpm.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # 随机选择时间步
            t = torch.randint(0, args.timesteps, (images.shape[0],), device=device).long()
            
            # 计算损失
            optimizer.zero_grad()
            loss = ddpm.loss_function(images, t)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        # 记录损失
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # 采样和保存图像
        if (epoch + 1) % args.sample_interval == 0:
            ddpm.eval()
            with torch.no_grad():
                samples = ddpm.sample(16, args.img_size)
                save_path = os.path.join(args.output_dir, "samples", f"epoch_{epoch+1}.png")
                save_images(samples, save_path, nrow=4)
                writer.add_images("Generated", (samples + 1) / 2, epoch)
                
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_model(ddpm, os.path.join(args.output_dir, "models", f"ddpm_epoch_{epoch+1}.pt"))
    
    # 保存最终模型
    save_model(ddpm, os.path.join(args.output_dir, "models", "ddpm_final.pt"))
    writer.close()
    
    print("训练完成!")

if __name__ == "__main__":
    main()
