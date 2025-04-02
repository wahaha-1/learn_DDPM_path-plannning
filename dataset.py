import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MapSample:
    """
    地图样本类，用于兼容直接加载的.pt文件
    """
    @staticmethod
    def load(filepath):
        """从文件加载MapSample对象"""
        return torch.load(filepath)

class PathPlanningDataset(Dataset):
    """
    路径规划数据集
    """
    def __init__(self, data_dir, max_path_length=64, transform=None, map_flip=False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            max_path_length: 最大路径长度（用于填充/截断）
            transform: 数据增强转换
            map_flip: 是否反转地图值（某些数据集可能用0表示障碍物，1表示自由空间）
        """
        self.data_dir = data_dir
        self.max_path_length = max_path_length
        self.transform = transform
        self.map_flip = map_flip
        
        # 获取所有.pt文件路径
        self.file_paths = []
        for file in os.listdir(data_dir):
            if file.endswith('.pt'):
                self.file_paths.append(os.path.join(data_dir, file))
                
        print(f"加载了{len(self.file_paths)}个路径规划样本，来自目录: {data_dir}")
        
        # 加载一个样本以检查形状
        if len(self.file_paths) > 0:
            sample = self._load_sample(self.file_paths[0])
            print(f"地图尺寸: {sample['map'].shape}")
            print(f"路径点数: {sample['path'].shape[0]}")
        
    def __len__(self):
        return len(self.file_paths)
    
    def _load_sample(self, filepath):
        """加载单个样本文件"""
        try:
            # 尝试直接加载
            data = torch.load(filepath)
            
            # 检查是否为MapSample对象或类似字典
            if hasattr(data, 'map') and hasattr(data, 'start') and hasattr(data, 'goal') and hasattr(data, 'path'):
                # MapSample对象
                return {
                    'map': data.map.float(),
                    'start': data.start,
                    'goal': data.goal,
                    'path': data.path.float()
                }
            elif isinstance(data, dict) and all(k in data for k in ['map', 'start', 'goal', 'path']):
                # 字典格式
                return {
                    'map': data['map'].float(),
                    'start': data['start'],
                    'goal': data['goal'],
                    'path': data['path'].float()
                }
            else:
                raise ValueError(f"未知的数据格式: {type(data)}")
        except Exception as e:
            print(f"加载样本 {filepath} 失败: {str(e)}")
            raise
    
    def __getitem__(self, idx):
        # 加载数据
        data = self._load_sample(self.file_paths[idx])
        
        map_data = data['map'].float()  # [H, W]
        start = data['start'].float() if data['start'].dtype != torch.float else data['start']  # [2]
        goal = data['goal'].float() if data['goal'].dtype != torch.float else data['goal']  # [2]
        path = data['path'].float() if data['path'].dtype != torch.float else data['path']  # [N, 2]
        
        # 如果需要，反转地图值
        if self.map_flip:
            map_data = 1.0 - map_data
        
        # 标准化地图：确保0表示障碍物，1表示自由空间
        # 如果地图中有大于1的值，应归一化到[0,1]
        if map_data.max() > 1.0:
            map_data = map_data / map_data.max()
        
        # 创建起点和终点的热图表示
        H, W = map_data.shape
        start_map = torch.zeros((H, W))
        goal_map = torch.zeros((H, W))
        
        # 将坐标四舍五入为整数
        sx, sy = int(round(start[0].item())), int(round(start[1].item()))
        gx, gy = int(round(goal[0].item())), int(round(goal[1].item()))
        
        # 确保坐标在地图范围内
        sx = max(0, min(sx, W-1))
        sy = max(0, min(sy, H-1))
        gx = max(0, min(gx, W-1))
        gy = max(0, min(gy, H-1))
        
        start_map[sy, sx] = 1.0
        goal_map[gy, gx] = 1.0
        
        # 处理路径以固定长度
        path_length = path.shape[0]
        
        if path_length <= self.max_path_length:
            # 填充路径
            padded_path = torch.full((self.max_path_length, 2), -1.)
            padded_path[:path_length] = path
            processed_path = padded_path
        else:
            # 采样路径点
            indices = torch.linspace(0, path_length-1, self.max_path_length).long()
            processed_path = path[indices]
        
        # 归一化路径坐标到[-1, 1]
        # 假设地图尺寸是路径坐标的参考范围
        normalized_path = processed_path.clone()
        normalized_path[:, 0] = 2 * (normalized_path[:, 0] / (W - 1)) - 1
        normalized_path[:, 1] = 2 * (normalized_path[:, 1] / (H - 1)) - 1
        
        # 对于填充的无效点，保持为-1
        valid_mask = (processed_path != -1).all(dim=1)
        normalized_path[~valid_mask] = -1
        
        # 组合成三通道输入: 地图, 起点图, 终点图
        condition = torch.stack([map_data, start_map, goal_map], dim=0)
        
        # 应用数据增强
        if self.transform:
            condition = self.transform(condition)
        
        return {
            'condition': condition,  # [3, H, W]
            'path': normalized_path,  # [max_path_length, 2]
            'valid_mask': valid_mask,  # [max_path_length]
            'raw_map': map_data,  # 用于可视化
            'raw_start': start,
            'raw_goal': goal,
            'raw_path': path,
            'filename': os.path.basename(self.file_paths[idx])  # 用于跟踪
        }
    
    def visualize_sample(self, idx):
        """可视化数据样本"""
        sample = self[idx]
        map_data = sample['raw_map']
        start = sample['raw_start']
        goal = sample['raw_goal']
        path = sample['raw_path']
        filename = sample['filename']
        
        plt.figure(figsize=(10, 10))
        plt.imshow(map_data, cmap='gray')
        plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)
        plt.plot(start[0], start[1], 'ro', markersize=10)
        plt.plot(goal[0], goal[1], 'bo', markersize=10)
        plt.title(f"Sample {idx}: {filename}")
        plt.axis('equal')
        plt.show()
        
    def visualize_batch(self, batch, predictions=None, save_path=None):
        """可视化一批样本，可选择性地添加预测路径"""
        batch_size = len(batch['raw_map'])
        
        fig, axes = plt.subplots(1, batch_size, figsize=(5*batch_size, 5))
        if batch_size == 1:
            axes = [axes]
            
        for i in range(batch_size):
            map_data = batch['raw_map'][i]
            start = batch['raw_start'][i]
            goal = batch['raw_goal'][i]
            path = batch['raw_path'][i]
            
            axes[i].imshow(map_data, cmap='gray')
            axes[i].plot(path[:, 0], path[:, 1], 'g-', linewidth=2, label='Ground Truth')
            
            if predictions is not None:
                pred_path = predictions[i]
                # 反归一化预测路径
                H, W = map_data.shape
                denorm_path = pred_path.clone()
                denorm_path[:, 0] = (denorm_path[:, 0] + 1) / 2 * (W - 1)
                denorm_path[:, 1] = (denorm_path[:, 1] + 1) / 2 * (H - 1)
                
                # 只绘制有效点
                valid_mask = (pred_path != -1).all(dim=1)
                valid_path = denorm_path[valid_mask]
                
                axes[i].plot(valid_path[:, 0], valid_path[:, 1], 'r--', linewidth=2, label='Predicted')
                
            axes[i].plot(start[0], start[1], 'ro', markersize=10)
            axes[i].plot(goal[0], goal[1], 'bo', markersize=10)
            
            # 添加文件名（如果存在）
            if 'filename' in batch:
                axes[i].set_title(batch['filename'][i])
            else:
                axes[i].set_title(f"Sample {i}")
                
            axes[i].axis('equal')
            
            if i == 0:
                axes[i].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def generate_stats(self):
        """生成数据集的统计信息"""
        map_sizes = []
        path_lengths = []
        obstacle_ratios = []
        
        print("正在分析数据集统计信息...")
        
        for i in range(min(100, len(self))):  # 分析前100个样本或所有样本
            sample = self[i]
            map_data = sample['raw_map']
            path = sample['raw_path']
            
            map_sizes.append(map_data.shape)
            path_lengths.append(path.shape[0])
            
            # 计算障碍物比例
            obstacle_ratio = (map_data <= 0).float().mean().item()
            obstacle_ratios.append(obstacle_ratio)
        
        print(f"地图尺寸: {set(map_sizes)}")
        print(f"平均路径长度: {np.mean(path_lengths):.2f} ± {np.std(path_lengths):.2f}")
        print(f"最短路径: {min(path_lengths)}, 最长路径: {max(path_lengths)}")
        print(f"障碍物平均占比: {np.mean(obstacle_ratios)*100:.2f}%")
        
        return {
            'map_sizes': map_sizes,
            'path_lengths': path_lengths,
            'obstacle_ratios': obstacle_ratios
        }

# 用于测试数据集加载的函数
def test_dataset(data_dir, max_path_length=64, num_samples=3):
    """测试数据集加载和可视化"""
    dataset = PathPlanningDataset(data_dir, max_path_length)
    
    # 生成统计信息
    stats = dataset.generate_stats()
    
    # 可视化几个样本
    for i in range(min(num_samples, len(dataset))):
        print(f"可视化样本 {i}:")
        dataset.visualize_sample(i)
    
    return dataset

if __name__ == "__main__":
    # 测试功能
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data/train"  # 默认数据目录
    
    test_dataset(data_dir)
