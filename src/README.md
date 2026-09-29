# CUDA Batch Image Processing

## Overview

This project demonstrates GPU-accelerated batch image processing using NVIDIA CUDA. The program processes 100 RGB images at 256 x 256 resolution and converts each image to grayscale using a CUDA kernel.

## GPU Processing

Each image is transferred from host memory to GPU memory. CUDA threads process pixels in parallel. For each pixel, the RGB channel values are averaged to produce a grayscale value.

The CUDA kernel uses a 2D configuration of 16 x 16 threads per block.

## Command-Line Usage

The program accepts three arguments:

    ./bin/image_processing [image_count] [width] [height]

Example:

    ./bin/image_processing 100 256 256

## Build

CUDA and nvcc are required.

    make clean
    make

## Run

The included script processes 100 images:

    ./run.sh

## Project Structure

- src/image_processing.cu - CUDA source code
- Makefile - build configuration
- run.sh - batch execution script
- data/ - generated image outputs
- execution.log - execution log

## Results

The project was successfully executed in a CUDA-enabled environment and processed 100 images. The execution produced 100 output image files.

## Lessons Learned

The project demonstrates how pixel-level image processing can be parallelized using CUDA. Each CUDA thread processes an individual pixel, allowing many pixels to be processed concurrently on the GPU.
