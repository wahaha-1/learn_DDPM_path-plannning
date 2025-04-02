# DDPM 路径规划

这个项目实现了基于扩散模型的路径规划方法。使用DDPM (Denoising Diffusion Probabilistic Models) 来生成从起点到终点的无碰撞路径。

## 项目结构

```
DDPMpath/
├── models/                    # 模型定义
│   ├── unet.py                # U-Net和路径U-Net模型架构
│   └── ddpm.py                # DDPM和路径DDPM模型实现
├── utils.py                   # 工具函数
├── dataset.py                 # 路径规划数据集
├── train.py                   # 图像训练脚本
├── train_path.py              # 路径规划训练脚本
├── sample.py                  # 图像采样脚本
├── sample_path.py             # 路径采样脚本
├── output/                    # 输出目录
│   ├── samples/               # 生成的样本
│   └── models/                # 保存的模型
├── logs/                      # TensorBoard日志
└── README.md                  # 项目说明
```

## 文件说明

### models/unet.py
实现了用于噪声预测的模型架构：
- `SinusoidalPositionEmbeddings`: 时间步嵌入模块
- `Block`: 基本卷积块
- `UNet`: 传统的U-Net模型，用于图像生成
- `PathUNet`: 专门为路径规划设计的U-Net变体

### models/ddpm.py
实现了扩散模型算法：
- `DDPM`: 传统的扩散模型，用于图像生成
- `PathDDPM`: 专门为路径规划设计的扩散模型，包含条件生成和碰撞避免

### utils.py
包含各种工具函数，如：
- 图像保存和可视化
- 数据转换和加载
- 模型保存和加载

### dataset.py
用于路径规划的数据集类：
- `PathPlanningDataset`: 加载和预处理路径规划数据
- 包含数据可视化功能

### train.py
实现了图像生成模型的训练流程：
- 参数解析
- 数据加载
- 训练循环
- 定期采样和模型保存
- TensorBoard日志记录

### train_path.py
路径规划模型的训练流程：
- 加载路径规划数据集
- 初始化PathUNet和PathDDPM
- 训练循环和验证
- 定期采样和可视化

### sample.py
从训练好的模型生成样本：
- 加载预训练模型
- 批量生成样本
- 保存生成的图像

### sample_path.py
从训练好的模型生成路径：
- 加载预训练模型
- 根据输入的地图、起点和终点生成路径
- 可视化和保存结果

## 使用方法

### 安装依赖
```bash
pip install torch torchvision tqdm matplotlib tensorboard
```

### 准备数据
路径规划数据集应包含以下格式的.pt文件：
```
{
    'map': 2D tensor (H, W),       # 地图，0表示障碍物，1表示自由空间
    'start': tensor [x, y],         # 起点坐标
    'goal': tensor [x, y],          # 终点坐标
    'path': tensor (N, 2)           # 路径点序列
}
```

### 训练模型
```bash
python train_path.py --data_dir ./data/train --val_dir ./data/validation --output_dir ./output
```

### 生成路径
```bash
python sample_path.py --model_path ./output/models/path_ddpm_final.pt --data_dir ./data/test --output_dir ./samples --collision_avoid
```

## 参数说明

### 训练参数
- `--data_dir`: 训练数据目录
- `--val_dir`: 验证数据目录
- `--output_dir`: 输出目录
- `--log_dir`: 日志目录
- `--batch_size`: 批量大小
- `--path_length`: 路径长度
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--sample_interval`: 采样间隔轮数
- `--save_interval`: 模型保存间隔轮数
- `--timesteps`: 扩散时间步
- `--device`: 运行设备，cuda或cpu

### 采样参数
- `--model_path`: 模型权重路径
- `--data_dir`: 测试数据目录
- `--output_dir`: 输出目录
- `--batch_size`: 批量生成大小
- `--path_length`: 路径长度
- `--num_samples`: 生成的样本数
- `--guide_scale`: 引导强度
- `--collision_avoid`: 启用碰撞避免
- `--timesteps`: 扩散时间步
- `--device`: 运行设备

## 模型原理

DDPM由两个关键过程组成：
1. **前向扩散过程**：逐步向原始图像添加噪声，直到完全破坏图像信息
2. **反向扩散过程**：通过神经网络学习逐步去除噪声，最终从纯噪声恢复有意义的图像

在训练阶段，模型学习预测在每个时间步添加的噪声。采样时，我们从随机噪声开始，然后通过学习到的去噪过程逐步生成图像。

## 路径规划特点

本项目实现的路径规划方法具有以下特点：

1. **条件生成**：模型接受地图、起点和终点作为条件输入，生成符合这些条件的路径
2. **固定长度表示**：将所有路径填充或裁剪到固定长度，简化处理
3. **碰撞避免**：在采样过程中实现碰撞检测和修正，确保生成的路径不会穿过障碍物
4. **起点/终点约束**：在采样过程中施加强引导，确保路径从指定起点出发并到达指定终点
