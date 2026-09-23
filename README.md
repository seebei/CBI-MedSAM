# CBI-MedSAM

CBI-MedSAM is a medical image segmentation framework built on the Segment Anything Model (SAM) and the implicit decoding architecture of I-MedSAM. It combines structure-enhanced contextual processing with boundary-aware refinement for prompt-conditioned segmentation.

## Architecture

- **Image encoder:** A pretrained SAM ViT-B encoder adapted using low-rank adaptation (LoRA).
- **Implicit decoder:** Coordinate-based prediction with coarse segmentation and selective fine refinement.
- **Structure-enhanced contextual module:** Squeeze-and-excitation and criss-cross attention for channel and spatial context processing.
- **Uncertainty-guided sampling (UGS):** Coordinate selection within the implicit decoder to allocate fine-refinement queries.
- **Boundary-aware fine context module:** Local context processing and Sobel-guided residual correction of segmentation logits.

The framework is evaluated on polyp and skin-lesion segmentation, cross-dataset endoscopy transfer, and four-modality BraTS2023 brain-tumor segmentation. The BraTS adaptation assembles two-dimensional slice predictions into volumetric outputs.

## Project Availability

This repository currently provides a project overview. Source code, configuration files, and training and evaluation instructions will be added in a future update.
