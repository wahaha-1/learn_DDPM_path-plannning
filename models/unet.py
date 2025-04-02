import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    时间步嵌入模块，使用正弦余弦函数将时间步编码为向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: 时间步 [batch_size]
        Returns:
            embeds: 时间步嵌入 [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    U-Net中的基本卷积块
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None, up=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        ) if time_emb_dim else None
        
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t=None):
        # 第一个卷积层
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # 添加时间嵌入信息
        if self.time_mlp and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
            
        # 第二个卷积层
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # 上采样或下采样
        return self.transform(h)


class PathUNet(nn.Module):
    """
    用于路径规划的条件UNet模型
    """
    def __init__(self, path_channels=2, time_dim=256, condition_channels=3, 
                 path_length=64, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.path_length = path_length
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 下采样
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 自适应池化到8x8
        )
        
        # 路径嵌入
        self.path_embedding = nn.Linear(path_channels * path_length, 512)
        
        # 融合路径和条件
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, path_channels * path_length)
        )
        
        # 时间条件MLP
        self.time_mlp_fusion = nn.Sequential(
            nn.Linear(time_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
    def forward(self, x, t, condition):
        """
        Args:
            x: 输入噪声路径 [B, L, 2]
            t: 时间步 [B]
            condition: 条件输入(地图+起点+终点) [B, 3, H, W]
        Returns:
            predicted_noise: 预测的噪声 [B, L, 2]
        """
        # 时间嵌入
        t = t.to(self.device)
        t_emb = self.time_mlp(t)
        time_features = self.time_mlp_fusion(t_emb)
        
        # 扁平化路径
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # [B, L*2]
        
        # 路径特征
        path_features = self.path_embedding(x_flat)  # [B, 512]
        
        # 条件特征
        cond_features = self.condition_encoder(condition)  # [B, 256, 8, 8]
        cond_flat = cond_features.reshape(batch_size, -1)  # [B, 256*8*8]
        
        # 融合特征
        combined = torch.cat([path_features, cond_flat], dim=1)  # [B, 512+256*8*8]
        combined = combined + time_features  # 添加时间特征
        
        # 预测输出
        output = self.fusion(combined)  # [B, L*2]
        output = output.reshape(batch_size, self.path_length, -1)  # [B, L, 2]
        
        return output


class UNet(nn.Module):
    """
    用于DDPM的U-Net模型
    """
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 初始层
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # 下采样路径
        self.down1 = Block(64, 128, time_dim)
        self.down2 = Block(128, 256, time_dim)
        self.down3 = Block(256, 512, time_dim)
        
        # 瓶颈层
        self.bottleneck1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bottleneck2 = nn.Conv2d(512, 512, 3, padding=1)
        
        # 上采样路径
        self.up1 = Block(512, 256, time_dim, up=True)
        self.up2 = Block(256, 128, time_dim, up=True)
        self.up3 = Block(128, 64, time_dim, up=True)
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x, t):
        """
        Args:
            x: 输入噪声图像 [B, C, H, W]
            t: 时间步 [B]
        Returns:
            predicted_noise: 预测的噪声 [B, C, H, W]
        """
        # 时间嵌入
        t = t.to(self.device)
        t = self.time_mlp(t)
        
        # 初始特征提取
        x0 = self.conv0(x)
        
        # 下采样路径
        d1 = self.down1(x0, t)
        d2 = self.down2(d1, t)
        d3 = self.down3(d2, t)
        
        # 瓶颈部分
        b1 = F.relu(self.bottleneck1(d3))
        b2 = F.relu(self.bottleneck2(b1))
        
        # 上采样路径，带跳跃连接
        u1 = torch.cat([b2, d3], dim=1)
        u1 = self.up1(u1, t)
        
        u2 = torch.cat([u1, d2], dim=1)
        u2 = self.up2(u2, t)
        
        u3 = torch.cat([u2, d1], dim=1)
        u3 = self.up3(u3, t)
        
        return self.output(u3)
