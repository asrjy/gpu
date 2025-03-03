#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

// error checking
// gaussian kernel is seperable; means we can perform 2d blur by doing 1d blur twice
// one 1d horizontal blur and another 1d vertical blur on the horizontal blur above
inline cudaError_t checkCudaErrors(cudaError_t error){
    if(error != cudaSuccess){
        fprintf(stderr, "CUDA ERROR = %d: %s (%s:%d)\n", error, cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return error;
}

vector<float> generateGaussianKernel1D(int kernelSize, float sigma){
    vector<float> kernel(kernelSize);
    int kernelRadius = kernelSize/2;
    float sum = 0.0f;

    for(int x = -kernelRadius; x <= kernelRadius; ++x){
        float gaussianValue = exp(-(x * x) / (2.0f * sigma * sigma));
        kernel[x + kernelRadius] = gaussianValue;
        sum += gaussianValue;
    }

    // normalizing kernel
    for(int i=0; i < kernelSize; ++i){
        kernel[i] /= sum;
    }
    return kernel;
}


// horizontal blur 
__global__
void gaussianBlurHorizontalKernel(const unsigned char* inputImage, unsigned char* intermediateImage, int width, int height, const float* kernel1D, int kernelSize){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width){
        float blurredPixelValue = 0.0f;
        int kernelRadius = kernelSize/2;

        for(int kernelColOffset = -kernelRadius; kernelColOffset <= kernelRadius; ++kernelColOffset){
            int neighbourCol = col + kernelColOffset;
            int clampedCol = max(0, min(neighbourCol, width - 1));

            blurredPixelValue += inputImage[row * width + clampedCol] * kernel1D[kernelColOffset + kernelRadius];
        }
        intermediateImage[row * width + col] = (unsigned char)blurredPixelValue;
    }
}


// vertical blur on result of horizontal blur
__global__ 
void gaussianBlurVerticalKernel(const unsigned char* intermediateImage, unsigned char* outputImage,
                                          int width, int height, const float* kernel1D, int kernelSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float blurredPixelValue = 0.0f;
        int kernelRadius = kernelSize / 2;

        for (int kernelRowOffset = -kernelRadius; kernelRowOffset <= kernelRadius; ++kernelRowOffset) {
            int neighborRow = row + kernelRowOffset;
            int clampedRow = max(0, min(neighborRow, height - 1)); // Clamp row index only (vertical blur)

            blurredPixelValue += intermediateImage[clampedRow * width + col] * kernel1D[kernelRowOffset + kernelRadius]; // Read from intermediateImage
        }
        outputImage[row * width + col] = (unsigned char)blurredPixelValue; // Write to final output image
    }
}


int main(){
    int width = 256;
    int height = 256;

    unsigned char* h_inputImage = new unsigned char[width * height];
    unsigned char* h_outputImage = new unsigned char[width * height];

    int kernelSize = 5;
    float sigma = 1.5f;

    vector<float> h_kernel1D = generateGaussianKernel1D(kernelSize, sigma);
    int kernelLength = h_kernel1D.size();

    float* d_kernel1D;
    checkCudaErrors(cudaMalloc((void**)&d_kernel1D, kernelLength * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_kernel1D, h_kernel1D.data(), kernelLength * sizeof(float), cudaMemcpyHostToDevice));

    unsigned char* d_intermediateImage;
    checkCudaErrors(cudaMalloc((void**)&d_intermediateImage, width * height * sizeof(unsigned char)));

    unsigned char* d_inputImage;
    unsigned char* d_outputImage;

    checkCudaErrors(cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1)/blockDim.x, (height + blockDim.y - 1)/blockDim.y);

    // creating cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // recording 'start' event at 0
    cudaEventRecord(start, 0);

    gaussianBlurHorizontalKernel<<<gridDim, blockDim>>>(d_inputImage, d_intermediateImage, width, height, d_kernel1D, kernelSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    gaussianBlurVerticalKernel<<<gridDim, blockDim>>>(d_intermediateImage, d_outputImage, width, height, d_kernel1D, kernelSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // recording 'stop' event
    cudaEventRecord(stop, 0);
    // wait for stop event to finish to be recorded and gpu to finish
    cudaEventSynchronize(stop);

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    printf("first few blurred pixel values are \n");
    for(int i =0; i < 10; ++i){
        printf("%d ", h_outputImage[i]);
    }
    printf("\n");

        // calculating execution time in milliseconds
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // milliseconds = milliseconds/100;

    printf("\n Kernel Execution Time: %.3f milliseconds\n", milliseconds);

        // freeing memory
    checkCudaErrors(cudaFree(d_intermediateImage)); 
    checkCudaErrors(cudaFree(d_kernel1D));
    checkCudaErrors(cudaFree(d_inputImage));
    checkCudaErrors(cudaFree(d_outputImage));
    delete[] h_inputImage;
    delete[] h_outputImage;

    return 0;
}