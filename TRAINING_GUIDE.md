# 新模型训练指南

本指南说明如何使用 `new_train.py` 训练新的弱监督PET-CT肿瘤分割模型，该模型集成了SAM和MedSAM。

## 环境准备

### 1. 安装依赖
```bash
# 基础依赖
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy
pip install easydict

# SAM依赖
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. 下载SAM预训练模型

**推荐使用MedSAM（专门为医学图像优化）：**
```bash
# 创建checkpoints目录
mkdir -p ./checkpoints

# 下载MedSAM模型
# 访问: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=sharing
# 下载 medsam_vit_b.pth 到 ./checkpoints/ 目录
```

**或者使用标准SAM：**
```bash
# SAM ViT-B (推荐开始使用)
wget -O ./checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# SAM ViT-L (更高精度)
wget -O ./checkpoints/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

## 数据准备

### 数据集结构
确保您的数据集按以下结构组织：
```
data/PCLT20k/
├── train.txt                # 训练集图像ID列表
├── test.txt                 # 测试集图像ID列表
└── [patient_id]/
    ├── [image_id]_PET.png   # PET图像
    ├── [image_id]_CT.png    # CT图像
    └── [image_id]_mask.png  # 分割掩码
```

### 数据格式说明
- **PET/CT图像**: 灰度图像，PNG格式
- **分割掩码**: 二值图像，PNG格式（0为背景，255为肿瘤）
- **图像ID格式**: `patient_id_slice_id`

### train.txt 和 test.txt 格式
每行一个图像ID，例如：
```
patient001_slice001
patient001_slice002
patient002_slice001
...
```

## 训练命令

### 1. 基础训练（使用MedSAM）
```bash
python new_train.py \
    --data_root ./data/PCLT20k/ \
    --sam_type medsam \
    --sam_checkpoint ./checkpoints/medsam_vit_b.pth \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints_new/
```

### 2. 使用标准SAM训练
```bash
python new_train.py \
    --data_root ./data/PCLT20k/ \
    --sam_type standard_sam_vit_b \
    --sam_checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints_new/
```

### 3. 使用简化SAM（无需预训练模型）
```bash
python new_train.py \
    --data_root ./data/PCLT20k/ \
    --sam_type simplified_sam \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints_new/
```

### 4. 恢复训练
```bash
python new_train.py \
    --data_root ./data/PCLT20k/ \
    --sam_type medsam \
    --sam_checkpoint ./checkpoints/medsam_vit_b.pth \
    --resume ./checkpoints_new/experiment_20240918_120000/checkpoint_epoch_50.pth \
    --batch_size 4 \
    --epochs 100
```

## 主要参数说明

### 数据参数
- `--data_root`: 数据集根目录
- `--batch_size`: 批量大小（根据GPU内存调整，推荐2-8）
- `--num_workers`: 数据加载线程数

### 模型参数
- `--sam_type`: SAM模型类型
  - `medsam`: MedSAM模型（推荐用于医学图像）
  - `standard_sam_vit_b/l/h`: 标准SAM模型
  - `simplified_sam`: 简化SAM实现
- `--sam_checkpoint`: SAM预训练模型路径
- `--use_weak_supervision`: 是否使用弱监督（默认开启）

### 训练参数
- `--epochs`: 训练轮数
- `--lr`: 学习率（推荐1e-4到1e-5）
- `--weight_decay`: 权重衰减

### 保存和评估参数
- `--save_dir`: 检查点保存目录
- `--save_freq`: 保存频率（每N个epoch保存一次）
- `--eval_freq`: 评估频率（每N个epoch评估一次）
- `--print_freq`: 打印频率（每N个batch打印一次）

## 训练过程监控

### 输出文件
训练过程中会在保存目录生成：
- `training.log`: 训练日志
- `config.json`: 训练配置
- `checkpoint_epoch_N.pth`: 定期检查点
- `checkpoint_epoch_N_best.pth`: 最佳模型

### 监控指标
- **训练损失**: 包含分类损失、弱监督损失、分割损失
- **验证指标**: IoU、Dice系数、准确率

### 示例训练输出
```
Epoch [1], Batch [0/100], Loss: 2.1234
  pet_classification: 0.6543
  ct_classification: 0.6789
  weakly_supervised: 0.4321
  segmentation: 0.3581
Epoch [1/100] - Train Loss: 1.8765, LR: 0.000100
Validation - IoU: 0.6234, Dice: 0.7456, Acc: 0.8901
New best model saved with Dice: 0.7456
```

## 训练建议

### 1. 硬件要求
- **GPU**: 8GB+ VRAM（推荐RTX 3080或更高）
- **内存**: 16GB+ RAM
- **存储**: 10GB+可用空间

### 2. 超参数调优
- **学习率**: 从1e-4开始，如果收敛慢可以提高到5e-4
- **批量大小**: 根据GPU内存调整，通常2-8之间
- **训练轮数**: 50-200轮，观察验证指标收敛情况

### 3. 模型选择建议
- **医学图像**: 优先使用MedSAM
- **计算资源有限**: 使用simplified_sam
- **追求最高精度**: 使用standard_sam_vit_h

### 4. 故障排除
- **GPU内存不足**: 减小batch_size
- **SAM模型加载失败**: 检查checkpoint路径和文件完整性
- **数据加载错误**: 检查数据集结构和文件路径

## 性能对比

| SAM类型 | 模型大小 | 内存占用 | 训练速度 | 预期精度 |
|---------|----------|----------|----------|----------|
| simplified_sam | 小 | 低 | 快 | 中等 |
| standard_sam_vit_b | 中 | 中 | 中 | 高 |
| medsam | 中 | 中 | 中 | 最高（医学图像）|
| standard_sam_vit_l | 大 | 高 | 慢 | 很高 |
| standard_sam_vit_h | 很大 | 很高 | 很慢 | 最高 |

## 训练完成后

### 模型评估
最佳模型会自动保存为 `checkpoint_epoch_N_best.pth`，包含：
- 模型权重
- 优化器状态
- 训练配置
- 验证指标

### 模型使用
训练完成的模型可以用于：
- 新图像的肿瘤分割
- 模型微调
- 进一步研究和改进

现在您可以开始训练了！建议先用小数据集测试训练流程，确认无误后再进行完整训练。