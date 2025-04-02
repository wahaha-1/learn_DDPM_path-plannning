import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.unet import PathUNet
from models.ddpm import PathDDPM
from dataset import PathPlanningDataset
from utils import save_model, load_model

def parse_args():
    parser = argparse.ArgumentParser(description="训练路径规划扩散模型")
    parser.add_argument("--data_dir", type=str, default="./data/train", help="训练数据目录")
    parser.add_argument("--val_dir", type=str, default="./data/validation", help="验证数据目录")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--path_length", type=int, default=64, help="路径长度")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--sample_interval", type=int, default=5, help="采样间隔")
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
    train_dataset = PathPlanningDataset(args.data_dir, max_path_length=args.path_length)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    # 加载验证数据集（如果存在）
    if os.path.exists(args.val_dir):
        val_dataset = PathPlanningDataset(args.val_dir, max_path_length=args.path_length)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=4
        )
    else:
        val_loader = None
    
    print(f"Training dataset size: {len(train_dataset)}")
    if val_loader:
        print(f"Validation dataset size: {len(val_dataset)}")
    
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
    
    # 配置优化器
    optimizer = ddpm.configure_optimizers(lr=args.lr)
    
    # 设置TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    for epoch in range(args.epochs):
        ddpm.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # 获取数据
            paths = batch['path'].to(device)  # [B, L, 2]
            condition = batch['condition'].to(device)  # [B, 3, H, W]
            valid_mask = batch['valid_mask'].to(device)  # [B, L]
            
            # 随机选择时间步
            t = torch.randint(0, args.timesteps, (paths.shape[0],), device=device).long()
            
            # 计算损失
            optimizer.zero_grad()
            loss = ddpm.loss_function(paths, t, condition, valid_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        # 记录损失
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")
        
        # 验证
        if val_loader and (epoch + 1) % args.sample_interval == 0:
            ddpm.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    paths = batch['path'].to(device)
                    condition = batch['condition'].to(device)
                    valid_mask = batch['valid_mask'].to(device)
                    
                    t = torch.randint(0, args.timesteps, (paths.shape[0],), device=device).long()
                    loss = ddpm.loss_function(paths, t, condition, valid_mask)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
            print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # 采样和可视化
        if (epoch + 1) % args.sample_interval == 0:
            ddpm.eval()
            with torch.no_grad():
                # 从验证集中获取一批样本
                if val_loader:
                    val_batch = next(iter(val_loader))
                    condition = val_batch['condition'].to(device)
                    
                    # 生成路径
                    generated_paths = ddpm.sample(
                        batch_size=min(4, condition.shape[0]),
                        path_length=args.path_length,
                        condition=condition[:4]
                    )
                    
                    # 可视化
                    val_dataset.visualize_batch(
                        {key: val[:4].cpu() for key, val in val_batch.items()},
                        predictions=generated_paths.cpu(),
                        save_path=os.path.join(args.output_dir, "samples", f"epoch_{epoch+1}.png")
                    )
                
        # 保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_model(ddpm, os.path.join(args.output_dir, "models", f"path_ddpm_epoch_{epoch+1}.pt"))
    
    # 保存最终模型
    save_model(ddpm, os.path.join(args.output_dir, "models", "path_ddpm_final.pt"))
    writer.close()
    
    print("训练完成!")

if __name__ == "__main__":
    main()
