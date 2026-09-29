#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void rgbToGrayscale(const uchar3 *input, unsigned char *output,
                               int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;

        uchar3 pixel = input[idx];

        output[idx] = static_cast<unsigned char>(
            0.299f * pixel.x +
            0.587f * pixel.y +
            0.114f * pixel.z
        );
    }
}

int main()
{
    Mat input = imread("data/Lena.png", IMREAD_COLOR);

    if (input.empty())
    {
        cerr << "Error: Could not load data/Lena.png" << endl;
        return 1;
    }

    int width = input.cols;
    int height = input.rows;
    size_t numPixels = static_cast<size_t>(width) * height;

    cout << "CUDA Image Processing Project" << endl;
    cout << "Input image: Lena.png" << endl;
    cout << "Image size: " << width << " x " << height << endl;

    uchar3 *d_input = nullptr;
    unsigned char *d_output = nullptr;

    cudaMalloc(&d_input, numPixels * sizeof(uchar3));
    cudaMalloc(&d_output, numPixels * sizeof(unsigned char));

    cudaMemcpy(
        d_input,
        input.ptr<uchar3>(),
        numPixels * sizeof(uchar3),
        cudaMemcpyHostToDevice
    );

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    rgbToGrayscale<<<gridSize, blockSize>>>(
        d_input,
        d_output,
        width,
        height
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cerr << "CUDA kernel launch failed: "
             << cudaGetErrorString(error) << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    cudaDeviceSynchronize();

    Mat output(height, width, CV_8UC1);

    cudaMemcpy(
        output.data,
        d_output,
        numPixels * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    );

    if (!imwrite("bin/lena_grayscale.png", output))
    {
        cerr << "Error: Could not save output image." << endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    cout << "CUDA grayscale processing completed successfully." << endl;
    cout << "Output: bin/lena_grayscale.png" << endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
