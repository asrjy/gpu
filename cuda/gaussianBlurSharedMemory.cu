// divide image into tiles 
// load a tile into shared memory per block
// blur pixels using shared memory 

// we perform padding of the input image so that we can load the tile into shared memory

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

//helper
inline cudaError_t checkCudaErrors(cudaError_t error){
    if(error != cudaSuccess){
        fprintf(stderr, "CUDA ERROR = %d: %s (%s:%d)\n", error, cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return error;
}

// global constants
#define BLOCK_SIZE 16  
// kernel size: 5
#define KERNEL_RADIUS 2 


// will be reused for horizontal and vertical
vector<float> generateGaussianKernel1D(int kernelSize, float sigma) {
    vector<float> kernel(kernelSize);
    int kernelRadius = kernelSize / 2;
    float sum = 0.0f;

    // performing the gaussian blur
    for (int x = -kernelRadius; x <= kernelRadius; ++x) {
        float gaussianValue = exp(-(x * x) / (2.0f * sigma * sigma));
        kernel[x + kernelRadius] = gaussianValue;
        sum += gaussianValue;
    }

    // normalize the kernel according to the gaussian blur formula
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum;
    }
    return kernel;
}


// horizontal blur
__global__ void gaussianBlurHorizontalKernel_SharedMem(const unsigned char* inputImage, unsigned char* intermediateImage,
                                                     int width, int height, const float* kernel1D, int kernelSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int kernelRadius = KERNEL_RADIUS; 
    int sharedTileHeight = BLOCK_SIZE;
    int sharedTileWidth = BLOCK_SIZE + 2 * kernelRadius;

    // declare the shared memory of size: 16 x 20
    __shared__ unsigned char sharedTile[BLOCK_SIZE][BLOCK_SIZE + 2 * KERNEL_RADIUS];

    // global memory coordinates
    int sharedTileStartRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int sharedTileStartCol = blockIdx.x * BLOCK_SIZE + threadIdx.x - kernelRadius;

    // cooperative loading into shared memory
    int sharedRow = threadIdx.y;
    int sharedCol = threadIdx.x;
    int globalRow = sharedTileStartRow;
    int globalCol = sharedTileStartCol + sharedCol;

    // loading the data into shared memory while it's in limits
    if (globalRow < height && globalCol >= 0 && globalCol < width) {
        sharedTile[sharedRow][sharedCol] = inputImage[globalRow * width + globalCol];
    } else {
        sharedTile[sharedRow][sharedCol] = 0; 
    }
    // wait for threads to load; synchronization
    __syncthreads(); 

    // blurrr
    if (row < height && col < width) {
        float blurredPixelValue = 0.0f;

        for (int kernelColOffset = -kernelRadius; kernelColOffset <= kernelRadius; ++kernelColOffset) {
            int sharedMemoryCol = sharedCol + kernelColOffset; // Access within sharedTile bounds
            if (sharedMemoryCol >= 0 && sharedMemoryCol < (BLOCK_SIZE + 2 * KERNEL_RADIUS)) {
                blurredPixelValue += sharedTile[sharedRow][sharedMemoryCol] * kernel1D[kernelColOffset + kernelRadius];
            }
            // blurredPixelValue += sharedTile[sharedRow][sharedMemoryCol] * kernel1D[kernelColOffset + kernelRadius];
        }
        intermediateImage[row * width + col] = (unsigned char)blurredPixelValue;
    }
}



// vertical blur
__global__ void gaussianBlurVerticalKernel_SharedMem(const unsigned char* intermediateImage, unsigned char* outputImage,
                                                    int width, int height, const float* kernel1D, int kernelSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int kernelRadius = KERNEL_RADIUS; 
    int sharedTileHeight = BLOCK_SIZE + 2 * kernelRadius; 
    int sharedTileWidth = BLOCK_SIZE;

    // declare the shared memory of size: 20 x 16
    __shared__ unsigned char sharedTile[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE];

    // calculating global memory indices
    int sharedTileStartRow = blockIdx.y * BLOCK_SIZE + threadIdx.y - kernelRadius; 
    int sharedTileStartCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // cooperative loading
    int sharedRow = threadIdx.y;
    int sharedCol = threadIdx.x;
    int globalRow = sharedTileStartRow + sharedRow; 
    int globalCol = sharedTileStartCol;

    // loading data into shared memory
    if (globalRow >= 0 && globalRow < height && globalCol < width) { // Check row and padded columns
        sharedTile[sharedRow][sharedCol] = intermediateImage[globalRow * width + globalCol];
    } else {
        sharedTile[sharedRow][sharedCol] = 0; 
    }
    // wait for all threads
    __syncthreads(); 

    // blurrr 2.0
    if (row < height && col < width) {
        float blurredPixelValue = 0.0f;

        for (int kernelRowOffset = -kernelRadius; kernelRowOffset <= kernelRadius; ++kernelRowOffset) {
            int sharedMemoryRow = sharedRow + kernelRowOffset; // Access within sharedTile bounds
            // blurredPixelValue += sharedTile[sharedMemoryRow][sharedCol] * kernel1D[kernelRowOffset + kernelRadius];
            if (sharedMemoryRow >= 0 && sharedMemoryRow < (BLOCK_SIZE + 2 * KERNEL_RADIUS)) {
                blurredPixelValue += sharedTile[sharedMemoryRow][sharedCol] * kernel1D[kernelRowOffset + kernelRadius];
            }
        }
        outputImage[row * width + col] = (unsigned char)blurredPixelValue;
    }
}


int main() {
    // stub
    int width = 256;
    int height = 256;

    unsigned char* h_inputImage = new unsigned char[width * height];
    unsigned char* h_outputImage = new unsigned char[width * height];
    if (!h_inputImage || !h_outputImage) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < width * height; ++i) {
        h_inputImage[i] = (i % 256);
    }

    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    unsigned char* d_intermediateImage;
    float* d_kernel1D;

    int kernelSize = 5;
    float sigma = 1.5f;
    vector<float> h_kernel1D = generateGaussianKernel1D(kernelSize, sigma);
    int kernelLength = h_kernel1D.size();

    checkCudaErrors(cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_intermediateImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_kernel1D, kernelLength * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_kernel1D, h_kernel1D.data(), kernelLength * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // creating cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // recording 'start' event at 0
    cudaEventRecord(start, 0);

    gaussianBlurHorizontalKernel_SharedMem<<<gridDim, blockDim>>>(d_inputImage, d_intermediateImage, width, height, d_kernel1D, kernelSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    gaussianBlurVerticalKernel_SharedMem<<<gridDim, blockDim>>>(d_intermediateImage, d_outputImage, width, height, d_kernel1D, kernelSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // recording 'stop' event
    cudaEventRecord(stop, 0);
    // wait for stop event to finish to be recorded and gpu to finish
    cudaEventSynchronize(stop);
    
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    printf("First few Gaussian Blurred (Shared Mem) pixel values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_outputImage[i]);
    }
    printf("\n");

    // calculating execution time in milliseconds
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // milliseconds = milliseconds/100;

    printf("\n Kernel Execution Time: %.3f milliseconds\n", milliseconds);

    checkCudaErrors(cudaFree(d_inputImage));
    checkCudaErrors(cudaFree(d_outputImage));
    checkCudaErrors(cudaFree(d_intermediateImage));
    checkCudaErrors(cudaFree(d_kernel1D));

    delete[] h_inputImage;
    delete[] h_outputImage;

    return 0;
}