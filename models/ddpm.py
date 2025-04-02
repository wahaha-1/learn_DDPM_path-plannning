import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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
