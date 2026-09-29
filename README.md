# CUDA Image Processing Project

## Overview

This project demonstrates GPU-accelerated image processing using NVIDIA CUDA.

The current implementation converts an RGB image to grayscale using a custom CUDA kernel. Each CUDA thread processes one image pixel, allowing the grayscale conversion to be performed in parallel on the GPU.

## Project Structure

```text
.
├── bin/
│   ├── image_processing.exe
│   └── lena_grayscale.png
├── data/
│   └── Lena.png
├── lib/
├── src/
│   └── image_processing.cu
├── INSTALL
├── Makefile
├── README.md
└── run.sh
