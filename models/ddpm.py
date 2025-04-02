import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class PathDDPM(nn.Module):
    """
    用于路径规划的条件扩散模型
    """
    def __init__(
        self, 
        model, 
        beta_start=1e-4, 
        beta_end=0.02, 
        timesteps=1000, 
        loss_type="l2",
        device="cuda"
    ):
        """
        初始化DDPM
        
        Args:
            model: 噪声预测网络 (PathUNet)
            beta_start: beta调度起始值
            beta_end: beta调度终止值
            timesteps: 扩散步数
            loss_type: 损失函数类型，'l1' 或 'l2'
            device: 使用的设备
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.device = device
        
        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # 预计算不同时间步的值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0, t, noise=None):
        """
        前向扩散过程: q(x_t | x_0)
        
        Args:
            x_0: 原始路径点 [B, L, 2]
            t: 时间步 [B]
            noise: 可选的噪声
        Returns:
            x_t: t时刻的噪声路径
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # 在时间步t添加噪声
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        # x_t = √(α_t) * x_0 + √(1 - α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def loss_function(self, x_0, t, condition, valid_mask=None):
        """
        计算带掩码的训练损失
        
        Args:
            x_0: 原始路径 [B, L, 2]
            t: 随机时间步 [B]
            condition: 条件输入(地图+起点+终点) [B, 3, H, W]
            valid_mask: 有效点掩码 [B, L]
        Returns:
            loss: 损失值
        """
        # 添加噪声
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t, condition)
        
        # 应用掩码（如果提供）
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(-1)  # [B, L, 1]
            noise = noise * valid_mask
            predicted_noise = predicted_noise * valid_mask
            # 计算损失时只考虑有效点
            if self.loss_type == 'l1':
                loss = F.l1_loss(predicted_noise, noise, reduction='sum') / (valid_mask.sum() + 1e-8)
            else:
                loss = F.mse_loss(predicted_noise, noise, reduction='sum') / (valid_mask.sum() + 1e-8)
        else:
            # 计算损失
            if self.loss_type == 'l1':
                loss = F.l1_loss(predicted_noise, noise)
            else:
                loss = F.mse_loss(predicted_noise, noise)
            
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size, path_length, condition, guide_scale=2.0, collision_avoid=True, map_res=1.0):
        """
        从模型中条件采样路径
        
        Args:
            batch_size: 批量大小
            path_length: 路径长度
            condition: 条件输入(地图+起点+终点) [B, 3, H, W]
            guide_scale: 分类器自由引导比例
            collision_avoid: 是否启用碰撞避免
            map_res: 地图分辨率（用于碰撞检测）
        Returns:
            paths: 生成的路径 [B, L, 2]
        """
        self.model.eval()
        
        # 提取起点和终点信息
        B, _, H, W = condition.shape
        map_data = condition[:, 0]  # [B, H, W]
        start_map = condition[:, 1]  # [B, H, W]
        goal_map = condition[:, 2]  # [B, H, W]
        
        # 提取起点和终点坐标
        start_coords = []
        goal_coords = []
        
        for b in range(B):
            # 找到起点坐标
            start_y, start_x = torch.where(start_map[b] == 1.0)
            if len(start_y) > 0:
                start_coords.append(torch.tensor([start_x[0], start_y[0]], device=self.device))
            else:
                # 默认使用地图中心作为起点
                start_coords.append(torch.tensor([W/2, H/2], device=self.device))
                
            # 找到终点坐标
            goal_y, goal_x = torch.where(goal_map[b] == 1.0)
            if len(goal_y) > 0:
                goal_coords.append(torch.tensor([goal_x[0], goal_y[0]], device=self.device))
            else:
                # 默认使用地图中心作为终点
                goal_coords.append(torch.tensor([W/2, H/2], device=self.device))
        
        # 堆叠为批量坐标
        start_coords = torch.stack(start_coords).float()  # [B, 2]
        goal_coords = torch.stack(goal_coords).float()  # [B, 2]
        
        # 归一化坐标到[-1, 1]
        start_norm = torch.zeros_like(start_coords)
        start_norm[:, 0] = 2 * (start_coords[:, 0] / (W - 1)) - 1
        start_norm[:, 1] = 2 * (start_coords[:, 1] / (H - 1)) - 1
        
        goal_norm = torch.zeros_like(goal_coords)
        goal_norm[:, 0] = 2 * (goal_coords[:, 0] / (W - 1)) - 1
        goal_norm[:, 1] = 2 * (goal_coords[:, 1] / (H - 1)) - 1
        
        # 从纯噪声开始
        path = torch.randn(batch_size, path_length, 2, device=self.device)
        
        # 起点和终点引导：将第一个和最后一个点设置为起点和终点
        path[:, 0] = start_norm
        path[:, -1] = goal_norm
        
        # 反向扩散过程
        for t in tqdm(reversed(range(self.timesteps)), desc='生成路径'):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(path, t_batch, condition)
            
            # 无噪声系数
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            
            # 计算x_{t-1}
            beta = self.betas[t]
            if t > 0:
                noise = torch.randn_like(path)
            else:
                noise = torch.zeros_like(path)
                
            # 计算方差
            variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            std_dev = torch.sqrt(variance)
            
            # 计算均值
            pred_original = (path - predicted_noise * std_dev) / torch.sqrt(alpha)
            prev_path = pred_original * torch.sqrt(alpha_cumprod_prev) + torch.sqrt(1 - alpha_cumprod_prev) * noise
            
            # 起点和终点引导
            if t > 0:  # 最后一步不需要引导，直接确保起点和终点
                # 渐进式引导：随着t变小，引导越来越强
                guide_weight = 1 - t / self.timesteps
                prev_path[:, 0] = guide_weight * start_norm + (1 - guide_weight) * prev_path[:, 0]
                prev_path[:, -1] = guide_weight * goal_norm + (1 - guide_weight) * prev_path[:, -1]
            
            # 碰撞避免引导
            if collision_avoid and t < self.timesteps // 2:  # 只在后半程应用碰撞避免
                # 将路径反归一化到地图坐标
                path_denorm = torch.zeros_like(prev_path)
                path_denorm[:, :, 0] = (prev_path[:, :, 0] + 1) / 2 * (W - 1)
                path_denorm[:, :, 1] = (prev_path[:, :, 1] + 1) / 2 * (H - 1)
                
                # 对每个样本进行碰撞检测和修正
                for b in range(batch_size):
                    # 获取地图
                    curr_map = map_data[b]  # [H, W]
                    
                    # 对每个路径点进行碰撞检测
                    for i in range(1, path_length-1):  # 不修改起点和终点
                        x, y = path_denorm[b, i].long()
                        
                        # 确保坐标在地图范围内
                        x = torch.clamp(x, 0, W-1)
                        y = torch.clamp(y, 0, H-1)
                        
                        # 如果点在障碍物上
                        if curr_map[y, x] < 0.5:  # 假设<0.5表示障碍物
                            # 计算梯度方向（向远离障碍物的方向移动）
                            # 使用简单的距离变换或梯度下降方法
                            
                            # 搜索附近的自由空间点
                            search_radius = 5
                            valid_points = []
                            
                            for dy in range(-search_radius, search_radius+1):
                                for dx in range(-search_radius, search_radius+1):
                                    nx, ny = x + dx, y + dy
                                    if (0 <= nx < W and 0 <= ny < H and 
                                        curr_map[ny, nx] >= 0.5):
                                        dist = np.sqrt(dx**2 + dy**2)
                                        valid_points.append((nx, ny, dist))
                            
                            if valid_points:
                                # 选择最近的自由点
                                valid_points.sort(key=lambda p: p[2])
                                nx, ny, _ = valid_points[0]
                                
                                # 将点移动到该位置
                                path_denorm[b, i, 0] = nx
                                path_denorm[b, i, 1] = ny
                
                # 重新归一化到[-1, 1]
                prev_path[:, :, 0] = 2 * (path_denorm[:, :, 0] / (W - 1)) - 1
                prev_path[:, :, 1] = 2 * (path_denorm[:, :, 1] / (H - 1)) - 1
            
            # 更新路径
            path = prev_path
        
        self.model.train()
        
        # 确保起点和终点正确
        path[:, 0] = start_norm
        path[:, -1] = goal_norm
        
        return path
    
    def configure_optimizers(self, lr=2e-4, betas=(0.9, 0.999)):
        """
        配置优化器
        """
        return torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)


# 保留原始DDPM类以便兼容
class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(
        self, 
        model, 
        beta_start=1e-4, 
        beta_end=0.02, 
        timesteps=1000, 
        loss_type="l2",
        device="cuda"
    ):
        """
        初始化DDPM
        
        Args:
            model: 噪声预测网络 (U-Net)
            beta_start: beta调度起始值
            beta_end: beta调度终止值
            timesteps: 扩散步数
            loss_type: 损失函数类型，'l1' 或 'l2'
            device: 使用的设备
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.device = device
        
        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # 预计算不同时间步的值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def forward_diffusion(self, x_0, t, noise=None):
        """
        前向扩散过程: q(x_t | x_0)
        
        Args:
            x_0: 原始图像
            t: 时间步
            noise: 可选的噪声
        Returns:
            x_t: t时刻的噪声图像
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # 在时间步t添加噪声
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # x_t = √(α_t) * x_0 + √(1 - α_t) * ε
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def loss_function(self, x_0, t):
        """
        计算训练损失
        
        Args:
            x_0: 原始图像 [B, C, H, W]
            t: 随机时间步 [B]
        Returns:
            loss: 损失值
        """
        # 添加噪声
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # 预测噪声
        predicted_noise = self.model(x_t, t)
        
        # 计算损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        else:
            loss = F.mse_loss(predicted_noise, noise)
            
        return loss
    
    @torch.no_grad()
    def sample(self, n_samples, img_size, channels=3):
        """
        从模型中采样图像
        
        Args:
            n_samples: 采样数量
            img_size: 图像尺寸
            channels: 图像通道数
        Returns:
            imgs: 生成的图像 [B, C, H, W]
        """
        self.model.eval()
        
        # 从纯噪声开始
        img = torch.randn(n_samples, channels, img_size, img_size, device=self.device)
        
        # 反向扩散过程
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(img, t_batch)
            
            # 无噪声系数
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod_prev[t]
            
            # 计算x_{t-1}
            beta = self.betas[t]
            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            variance = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            
            # 均值系数
            mean_coeff1 = beta * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            mean_coeff2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
            
            # 计算均值
            mean = 1 / torch.sqrt(alpha) * (img - mean_coeff1 * predicted_noise)
            
            # 采样
            img = mean + torch.sqrt(variance) * noise
        
        self.model.train()
        
        # 归一化到[-1, 1]区间
        img = torch.clamp(img, -1.0, 1.0)
        
        return img
    
    def configure_optimizers(self, lr=2e-4, betas=(0.9, 0.999)):
        """
        配置优化器
        """
        return torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
