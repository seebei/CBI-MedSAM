# CBI-MedSAM

CBI-MedSAM 是一个基于 [Segment Anything Model（SAM）](https://github.com/facebookresearch/segment-anything) 和 I-MedSAM 隐式解码思想构建的提示条件医学图像分割框架。模型在参数高效适配 SAM 图像编码器的基础上，引入结构增强的上下文处理、不确定性引导采样和边界感知细化，以改善医学目标的区域一致性与边界质量。

## 架构

- **图像编码器**：采用预训练的 SAM ViT-B 编码器，并通过低秩适应（LoRA）完成参数高效微调。
- **隐式解码器（INR）**：使用坐标条件预测执行由粗到细的分割。粗分支生成初始分割与隐式特征，细分支进一步修正选中的坐标。
- **结构增强上下文模块**：结合挤压与激发（SE）和交叉注意力（CCA），建模通道及空间上下文。
- **不确定性引导采样（UGS）**：根据粗预测的不确定性选择 Top-K 坐标，将细化计算集中到更难判断的位置。
- **边界感知细化模块**：结合局部上下文处理与由分割 logit 计算的 Sobel 边缘信息，对边界区域执行残差校正。

研究评估范围包括息肉分割、皮肤病变分割、跨数据集内镜迁移，以及四模态 BraTS2023 脑肿瘤分割。BraTS 适配以二维切片训练和推理，并将切片级 WT、TC、ET 预测重新组装为三维体积。

## 精简发布内容

本目录是从完整实验工程整理出的轻量发布版，包含：

- CBI-MedSAM 核心模型和 SAM 依赖代码；
- 数据加载、分布式支持和分割评价代码；
- sessile-Kvasir 单卡训练脚本；
- sessile-Kvasir 同域测试脚本；
- 从 sessile-Kvasir 到 CVC 的跨数据集测试脚本；
- Conda 环境文件。

为减小仓库体积，本版本不包含数据集、`work_dir`、SAM 权重、训练检查点、生成的预测图、IDE 配置、缓存文件及重复的实验脚本。皮肤病变和 BraTS2023 的数据适配与完整实验配置应按对应实验协议另行接入。

## 目录结构

```text
CBI-MedSAM-release/
├── main.py
├── dataset.py
├── dist.py
├── environment.yml
├── model/
│   ├── __init__.py
│   └── CBI_MedSAM.py
├── segment_anything/
├── utils/
└── scripts/
    ├── train_sessile.sh
    ├── test_sessile.sh
    └── test_sessile_to_CVC.sh
```

运行后会自动创建以下本地目录；它们已加入 `.gitignore`：

```text
dataset/
sam_ckp/
work_dir/
```

## 环境配置

代码基于 Ubuntu、Python 3.8、PyTorch 1.12 和 CUDA 11.3 环境整理。

```bash
conda env create -f environment.yml
conda activate cbi-medsam
```

## 准备数据集

为了验证二元息肉分割，可从下面的共享目录下载 sessile-Kvasir 和 CVC：

- [sessile-Kvasir 与 CVC 数据集](https://drive.google.com/drive/folders/101LDnr7Gget7ehZQkHCNH1csD2WCCBX6?usp=sharing)

在项目根目录执行：

```bash
mkdir -p dataset

# 将下载的压缩包放到项目根目录后解压
unzip sessile-Kvasir.zip -d dataset/
unzip CVC.zip -d dataset/
```

预期结构：

```text
dataset/
├── sessile-Kvasir/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
└── CVC/
    └── PNG/
        ├── Ground Truth/
        └── Original/
```

## 准备 SAM ViT-B 检查点

从 Meta 官方地址下载 SAM ViT-B 权重：

```bash
mkdir -p sam_ckp
wget -O sam_ckp/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

最终路径应为：

```text
sam_ckp/sam_vit_b_01ec64.pth
```

## 训练

在 sessile-Kvasir 上进行单卡训练：

```bash
GPU=0 \
IMAGE_SIZE=384 \
BATCH_SIZE=1 \
NUM_WORKERS=8 \
NUM_EPOCHS=200 \
bash scripts/train_sessile.sh
```

训练日志和检查点保存在：

```text
work_dir/CBI_MedSAM_sessile_Kvasir/
```

可以通过 `TASK_NAME` 修改实验名称。

## 评估

### sessile-Kvasir 同域评估

```bash
GPU=0 \
MODEL_CKPT=./work_dir/CBI_MedSAM_sessile_Kvasir/model_best.pth \
bash scripts/test_sessile.sh
```

### CVC 跨数据集评估

该实验直接使用 sessile-Kvasir 上训练的检查点：

```bash
GPU=0 \
MODEL_CKPT=./work_dir/CBI_MedSAM_sessile_Kvasir/model_best.pth \
bash scripts/test_sessile_to_CVC.sh
```

实际检查点文件名可能包含 DSC、HD、HD95 和 epoch 等信息，请将 `MODEL_CKPT` 替换为真实路径。

> 当前数据加载器在验证和测试时由真实掩码生成框提示。报告结果时应将其明确标注为 GT-box（oracle prompt）协议。

## 项目可用性

本精简版本已经提供核心源代码、环境文件以及 sessile-Kvasir/CVC 的训练和评估入口。数据集、SAM 预训练权重与任务训练检查点需按上面的说明单独下载或生成。

## 致谢

本项目基于 Meta AI 的 Segment Anything 和 I-MedSAM 相关工作构建。使用本代码开展研究时，请同时引用相应的基础工作，并在 CBI-MedSAM 论文正式发布后补充其正式引文。
