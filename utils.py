import torch
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def save_images(images, path, nrow=8, normalize=True, value_range=(-1, 1)):
    """
    保存图像网格
    
    Args:
        images: [B, C, H, W] 图像张量
        path: 保存路径
        nrow: 每行图像数量
        normalize: 是否归一化
        value_range: 图像值范围
    """
    # 转换为网格
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
    
    # 转换为numpy并调整通道顺序
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    # 保存图像
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def get_data_transforms(img_size=64):
    """
    获取数据转换
    
    Args:
        img_size: 目标图像尺寸
    Returns:
        transform: 图像转换函数
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
def get_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True):
    """
    创建数据加载器
    
    Args:
        dataset: PyTorch数据集
        batch_size: 批量大小
        num_workers: 数据加载线程数
        shuffle: 是否随机打乱数据
    Returns:
        dataloader: 数据加载器
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def save_model(model, path):
    """
    保存模型
    
    Args:
        model: 模型
        path: 保存路径
    """
    torch.save(model.state_dict(), path)
    
def load_model(model, path, device):
    """
    加载模型
    
    Args:
        model: 模型
        path: 保存路径
        device: 运行设备
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def show_tensor_images(images, title=None, nrow=8, save_path=None):
    """
    显示张量图像
    
    Args:
        images: [B, C, H, W] 图像张量
        title: 图像标题
        nrow: 每行图像数量
        save_path: 保存路径
    """
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    if title:
        plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
