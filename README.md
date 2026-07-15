# GPU-Accelerated Satellite Image Change Detection for Disaster Assessment Using OpenMP and CUDA

**Author:** Bristi Biswas (Enrollment: 23104028)
**Institution:** Jaypee Institute of Information Technology, Sector 62, Noida
**Program:** High Performance Computing (HPC) Internship Program 2026

## Project Overview

Natural disasters cause significant damage to infrastructure and human life. Rapid assessment of affected areas is essential for effective disaster response. This project provides a High Performance Computing (HPC) framework designed to rapidly analyze pre-disaster and post-disaster satellite imagery to identify regions of significant change. 

By transitioning from a traditional sequential processing pipeline to parallelized architecture, this system dramatically reduces computational overhead. The project benchmarks a Sequential CPU baseline against an OpenMP-accelerated multi-core implementation and custom CUDA-based GPU kernels. Additionally, it integrates a PyTorch-based Siamese Convolutional Neural Network (CNN) for deep learning-driven change detection.

## Key Features

*   **Sequential Baseline:** A traditional CPU-bound computer vision pipeline utilizing Image Differencing, Thresholding, and Morphological Operations.
*   **OpenMP Acceleration:** C++ based multi-threading for CPU-intensive image preprocessing and batch patch generation.
*   **CUDA GPU Acceleration:** Custom C++ CUDA kernels (`__global__`) for highly parallelized pixel-level thresholding and absolute difference computations.
*   **Deep Learning Integration:** A Siamese CNN built in PyTorch featuring mixed-precision (AMP) training and multi-worker data loading.
*   **HPC Portability:** A modular architecture designed for local development with seamless deployment capabilities to the Ramanujan Universe (RU) HPC facility.

## Tech Stack

| Category | Technologies Used |
| :--- | :--- |
| **Languages** | Python 3.13, C++17 |
| **HPC & Parallelization** | OpenMP, NVIDIA CUDA (nvcc) |
| **Deep Learning** | PyTorch, Torchvision |
| **Computer Vision** | OpenCV, NumPy |

## Project Structure

```text
gpu-satellite-change-detection/
├── benchmarks/           # Scripts for measuring Execution Time, Speedup, Scalability
├── config/               # YAML configuration files
├── cuda/                 # .cu files, CUDA kernels, and C++ headers
├── data/                 # Raw satellite imagery (pre and post disaster)
├── datasets/             # Processed, normalized 256x256 image patches
├── docs/                 # Documentation and project synopsis
├── evaluation/           # Scripts for Accuracy, Precision, Recall metrics
├── models/               # PyTorch Siamese CNN architecture and Dataset loaders
├── openmp/               # C/C++ files for multi-core CPU parallelization
├── outputs/              # Final binary Change Maps and visual overlays
├── preprocessing/        # Python modules for resizing and patch extraction
├── results/              # CSV files containing benchmark tables
├── sequential/           # Baseline CPU change detection implementation
├── training/             # Deep learning training and validation scripts
└── utils/                # Reusable helper functions (logging, I/O)
