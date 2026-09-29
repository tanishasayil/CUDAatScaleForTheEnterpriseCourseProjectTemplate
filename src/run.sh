#!/bin/bash

mkdir -p data
rm -f data/output_*.ppm

./bin/image_processing 100 256 256 | tee execution.log
