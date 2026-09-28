# CBI-MedSAM

CBI-MedSAM is a prompt-conditioned medical image segmentation framework built upon the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) and the implicit decoding paradigm of I-MedSAM. Based on parameter-efficient adaptation of the SAM image encoder, the framework incorporates structure-enhanced contextual modeling, uncertainty-guided sampling, and boundary-aware refinement to improve region consistency and boundary delineation in medical image segmentation.

## Architecture

- **Image Encoder**: Uses the pretrained SAM ViT-B image encoder with Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
- **Implicit Neural Representation (INR) Decoder**: Performs coarse-to-fine segmentation through coordinate-conditioned prediction. The coarse branch generates an initial segmentation prediction and implicit features, while the fine branch further refines selected coordinates.
- **Structure-Enhanced Context Module**: Combines Squeeze-and-Excitation (SE) and Criss-Cross Attention (CCA) to model channel-wise and spatial contextual information.
- **Uncertainty-Guided Sampling (UGS)**: Selects Top-K coordinates according to uncertainty estimated from coarse predictions, concentrating refinement computation on locations that are more difficult to classify.
- **Boundary-Aware Refinement Module**: Combines local contextual processing with Sobel edge information computed from segmentation logits to perform residual correction around object boundaries.

The evaluation scope includes polyp segmentation, skin-lesion segmentation, cross-dataset endoscopy transfer, and four-modality BraTS2023 brain-tumor segmentation. For the BraTS adaptation, the model is trained and evaluated on two-dimensional slices, and the slice-wise WT, TC, and ET predictions are reassembled into three-dimensional volumes.

## Repository Structure

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

The following local directories are created during setup or execution and are excluded through .gitignore:
dataset/
sam_ckp/
work_dir/

Environment Setup
The released code has been organized for an environment based on Ubuntu, Python 3.8, PyTorch 1.12, and CUDA 11.3.
conda env create -f environment.yml
conda activate cbi-medsam

Dataset Preparation
For binary polyp segmentation experiments, the sessile-Kvasir and CVC datasets can be obtained from the shared directory below:
- sessile-Kvasir and CVC datasets
From the project root, create the dataset directory and extract the downloaded archives:
mkdir -p dataset

# Place the downloaded archives in the project root and extract them
unzip sessile-Kvasir.zip -d dataset/
unzip CVC.zip -d dataset/

The expected directory structure is:
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

SAM ViT-B Checkpoint
Download the official SAM ViT-B checkpoint from Meta:
mkdir -p sam_ckp

wget -O sam_ckp/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

The checkpoint should be located at:
sam_ckp/sam_vit_b_01ec64.pth

Training
To train CBI-MedSAM on sessile-Kvasir using a single GPU:
GPU=0 \
IMAGE_SIZE=384 \
BATCH_SIZE=1 \
NUM_WORKERS=8 \
NUM_EPOCHS=200 \
bash scripts/train_sessile.sh

Training logs and checkpoints are saved to:
work_dir/CBI_MedSAM_sessile_Kvasir/

The experiment name can be changed through the TASK_NAME variable.
Evaluation
In-Domain Evaluation on sessile-Kvasir
GPU=0 \
MODEL_CKPT=./work_dir/CBI_MedSAM_sessile_Kvasir/model_best.pth \
bash scripts/test_sessile.sh

Cross-Dataset Evaluation on CVC
This experiment directly uses the checkpoint trained on sessile-Kvasir without target-domain fine-tuning:
GPU=0 \
MODEL_CKPT=./work_dir/CBI_MedSAM_sessile_Kvasir/model_best.pth \
bash scripts/test_sessile_to_CVC.sh

The actual checkpoint filename may contain the DSC, HD, HD95, and epoch values. Please replace MODEL_CKPT with the path to the checkpoint used in your experiment.
Evaluation protocol: The current data loader generates bounding-box prompts from the ground-truth masks during validation and testing. Reported results should therefore be explicitly described as using a GT-box (oracle-prompt) protocol.

Release Scope
This lightweight release provides the core source code, environment configuration, and training/evaluation entry points for the sessile-Kvasir and CVC experiments.
The datasets, pretrained SAM checkpoint, and task-specific trained checkpoints are not included in the repository and should be downloaded or generated separately according to the instructions above.
Acknowledgements
This project builds upon Meta AI's Segment Anything and the I-MedSAM framework.
When using this repository for research, please cite the corresponding foundational works. The formal citation for CBI-MedSAM will be added after the paper is published.
