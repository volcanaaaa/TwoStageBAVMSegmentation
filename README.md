This repository contains a deep learning-based medical image segmentation system for medical imaging data, specifically designed for brain AVM (Arteriovenous Malformation) segmentation tasks.
This project implements a two-stage segmentation approach:
Stage 1: Coarse detection of potential AVM regions using a 2D U-Net
Stage 2: Fine-grained 3D segmentation using a U-Net with CBAM attention mechanisms
Features：
3D medical image processing and segmentation
Two-stage segmentation pipeline for improved accuracy
CBAM (Convolutional Block Attention Module) integration
Support for various backbone architectures
Comprehensive training and evaluation scripts
Requirements：The project requires the following dependencies (see requirements.txt for specific versions).
