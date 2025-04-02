# DDPM (Denoising Diffusion Probabilistic Models)

这个项目实现了基于U-Net架构的DDPM (Denoising Diffusion Probabilistic Models) 模型，用于图像生成。DDPM是一种强大的生成模型，通过逐步去噪的过程从随机噪声生成高质量图像。

## 项目结构

```
DDPMpath/
├── models/                    # 模型定义
│   ├── unet.py                # U-Net模型架构
│   └── ddpm.py                # DDPM模型实现
├── utils.py                   # 工具函数
├── train.py                   # 训练脚本
├── sample.py                  # 采样生成脚本
├── output/                    # 输出目录
│   ├── samples/               # 生成的样本
│   └── models/                # 保存的模型
├── logs/                      # TensorBoard日志
└── README.md                  # 项目说明
```

## 文件说明

### models/unet.py
实现了用于噪声预测的U-Net网络架构，包含以下核心组件：
- `SinusoidalPositionEmbeddings`: 时间步嵌入模块，使用正弦余弦函数将时间步编码为向量
- `Block`: U-Net的基本卷积块，包含上采样和下采样版本
- `UNet`: 完整的U-Net模型，包含下采样路径、瓶颈层和带跳跃连接的上采样路径

### models/ddpm.py
实现了完整的DDPM算法，包括：
- 前向扩散过程（添加噪声）
- 反向扩散过程（去噪采样）
- 训练损失计算
- beta调度和相关参数计算

### utils.py
包含各种工具函数，如：
- 图像保存和可视化
- 数据转换和加载
- 模型保存和加载

### train.py
实现了模型训练流程：
- 参数解析
- 数据加载
- 训练循环
- 定期采样和模型保存
- TensorBoard日志记录

### sample.py
从训练好的模型生成样本：
- 加载预训练模型
- 批量生成样本
- 保存生成的图像

## 使用方法

### 安装依赖
```bash
pip install torch torchvision tqdm matplotlib tensorboard
```

### 训练模型
```bash
python train.py --data_dir /path/to/your/dataset --output_dir ./output --batch_size 32 --epochs 100
```

### 生成样本
```bash
python sample.py --model_path ./output/models/ddpm_final.pt --output_dir ./samples --num_samples 16
```

## 参数说明

### 训练参数
- `--data_dir`: 数据集目录
- `--output_dir`: 输出目录
- `--log_dir`: 日志目录
- `--batch_size`: 批量大小
- `--img_size`: 图像尺寸
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--sample_interval`: 采样间隔轮数
- `--save_interval`: 模型保存间隔轮数
- `--timesteps`: 扩散时间步
- `--device`: 运行设备，cuda或cpu

### 采样参数
- `--model_path`: 模型权重路径
- `--output_dir`: 输出目录
- `--num_samples`: 生成的样本数
- `--batch_size`: 批量生成大小
- `--img_size`: 图像尺寸
- `--timesteps`: 扩散时间步
- `--device`: 运行设备

## 模型原理

DDPM由两个关键过程组成：
1. **前向扩散过程**：逐步向原始图像添加噪声，直到完全破坏图像信息
2. **反向扩散过程**：通过神经网络学习逐步去除噪声，最终从纯噪声恢复有意义的图像

在训练阶段，模型学习预测在每个时间步添加的噪声。采样时，我们从随机噪声开始，然后通过学习到的去噪过程逐步生成图像。

## 引用

本实现基于以下论文：
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239.
