#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

__global__ void processImage(const unsigned char* input,
                             unsigned char* output,
                             int width,
                             int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = (y * width + x) * 3;

    unsigned char r = input[index];
    unsigned char g = input[index + 1];
    unsigned char b = input[index + 2];

    unsigned char gray =
        static_cast<unsigned char>(
            (static_cast<int>(r) +
             static_cast<int>(g) +
             static_cast<int>(b)) / 3);

    output[index] = gray;
    output[index + 1] = gray;
    output[index + 2] = gray;
}

void savePPM(const std::string& filename,
             const std::vector<unsigned char>& image,
             int width,
             int height) {
    std::ofstream file(filename, std::ios::binary);

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data()),
               image.size());
}

int main(int argc, char* argv[]) {
    int imageCount = 100;
    int width = 256;
    int height = 256;

    if (argc > 1) {
        imageCount = std::atoi(argv[1]);
    }

    if (argc > 2) {
        width = std::atoi(argv[2]);
    }

    if (argc > 3) {
        height = std::atoi(argv[3]);
    }

    if (imageCount <= 0 || width <= 0 || height <= 0) {
        std::cerr << "Usage: " << argv[0]
                  << " [image_count] [width] [height]\n";
        return 1;
    }

    size_t imageSize =
        static_cast<size_t>(width) * height * 3;

    std::vector<unsigned char> hostInput(imageSize);
    std::vector<unsigned char> hostOutput(imageSize);

    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    cudaMalloc(&deviceInput, imageSize);
    cudaMalloc(&deviceOutput, imageSize);

    dim3 threads(16, 16);
    dim3 blocks(
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y);

    std::cout << "CUDA Batch Image Processing\n";
    std::cout << "Images: " << imageCount << "\n";
    std::cout << "Resolution: "
              << width << "x" << height << "\n";
    std::cout << "GPU kernel: RGB to grayscale\n";

    for (int image = 0; image < imageCount; ++image) {
        for (size_t i = 0; i < imageSize; i += 3) {
            hostInput[i] =
                static_cast<unsigned char>((i + image * 17) % 256);
            hostInput[i + 1] =
                static_cast<unsigned char>((i / 2 + image * 31) % 256);
            hostInput[i + 2] =
                static_cast<unsigned char>((i / 3 + image * 47) % 256);
        }

        cudaMemcpy(deviceInput,
                   hostInput.data(),
                   imageSize,
                   cudaMemcpyHostToDevice);

        processImage<<<blocks, threads>>>(
            deviceInput,
            deviceOutput,
            width,
            height);

        cudaError_t error = cudaGetLastError();

        if (error != cudaSuccess) {
            std::cerr << "Kernel error: "
                      << cudaGetErrorString(error) << "\n";
            cudaFree(deviceInput);
            cudaFree(deviceOutput);
            return 1;
        }

        cudaDeviceSynchronize();

        cudaMemcpy(hostOutput.data(),
                   deviceOutput,
                   imageSize,
                   cudaMemcpyDeviceToHost);

        std::string filename =
            "data/output_" + std::to_string(image) + ".ppm";

        savePPM(filename, hostOutput, width, height);

        if ((image + 1) % 10 == 0) {
            std::cout << "Processed "
                      << image + 1 << "/"
                      << imageCount << " images\n";
        }
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    std::cout << "Processing completed successfully.\n";

    return 0;
}
