#include "helpers.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__
void matrixTransposeKernel(const float* d_inputMatrix, float* d_outputMatrix, int height, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < height) && (col < width)){
        d_outputMatrix[col*height + row] = d_inputMatrix[row*width + col];
    }
}

int main(){
    int width = 256;
    int height = 256;

    float* h_inputMatrix= new float[height * width];
    float* h_outputMatrix= new float[height * width];

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            h_inputMatrix[i*width+j] = i*j*i;
        }
    }

    float* d_inputMatrix;
    float* d_outputMatrix;

    CUDA_CHECK(cudaMalloc((void**)&d_inputMatrix, sizeof(float)*height * width));
    CUDA_CHECK(cudaMalloc((void**)&d_outputMatrix, sizeof(float)*height * width));

    CUDA_CHECK(cudaMemcpy(d_inputMatrix, h_inputMatrix, sizeof(float)*height*width, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32);
    dim3 gridDim(((blockDim.x + height - 1)/blockDim.x), ((blockDim.y + width - 1)/blockDim.y));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matrixTransposeKernel<<<gridDim, blockDim>>>(d_inputMatrix, d_outputMatrix, height, width);
    CHECK_KERNEL_ERROR();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_outputMatrix, d_outputMatrix, sizeof(float)*width*height, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (h_outputMatrix[j * height + i] != h_inputMatrix[i * width + j]) {
                printf("Error: h_outputMatrix[%d][%d] = %f, h_inputMatrix[%d][%d] = %f\n",
                       j, i, h_outputMatrix[j * height + i], i, j, h_inputMatrix[i * width + j]);
                correct = false;
                break; 
            }
        }
        if (!correct) break; 
    }

    if (correct) {
        printf("Transpose verification successful!\n");
    } else {
        printf("Transpose verification failed!\n");
    }

    printf("\nFirst few elements of Input Matrix:\n");
    for (int i = 0; i < min(height, 4); ++i) {
        for (int j = 0; j < min(width, 4); ++j) {
            printf("%f ", h_inputMatrix[i * width + j]);
        }
        printf("\n");
    }

    printf("\nFirst few elements of Output Matrix (Transposed):\n");
    for (int i = 0; i < min(width, 4); ++i) {
        for (int j = 0; j < min(height, 4); ++j) {
            printf("%f ", h_outputMatrix[i * height + j]);
        }
        printf("\n");
    }

    printf("Elapsed Time(GPU): %f\n", milliseconds);

    CUDA_CHECK(cudaFree(d_inputMatrix));
    CUDA_CHECK(cudaFree(d_outputMatrix));

    return 0;
}