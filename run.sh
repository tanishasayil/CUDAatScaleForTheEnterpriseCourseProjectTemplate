#!/bin/bash
set -e

echo "Building CUDA image processing project..."
make

echo "Running CUDA image processing project..."
./bin/image_processing.exe

echo "Done."
