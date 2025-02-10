#include "helpers.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__
void sgemm_coalescing(const float* M, const floa* N, float* P, int Width, int Height){
    int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int y = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);

    if((x < Width) && (y < Height)){
        float tmp = 0.0;
        for(int i = 0; i < Width; i++){
            temp += M[x * Width + i] * B[i * Height + y];
        }
    }
    P[x * Height + y] = alpha * tmp + beta * P[x * Height + y];
}

__global__
void matMulKernel(const float* M, const float* N, float* P, int Width, int Height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ((row < Height) && (col < Width)) { 
        float PValue = 0.0f;
        for (int i = 0; i < Width; i++) {
            PValue += M[row * Width + i] * N[i * Width + col];
        }
        P[row * Width + col] = PValue;
    }
}

int main() {
    int width = 1024;
    int height = 512;

    float* h_A = new float[height * width];  
    float* h_B = new float[height * width];  
    float* h_C = new float[height * width];  

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_A[i * width + j] = 1.0f;
            h_B[i * width + j] = 2.0f;
        }
    }

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(float) * height * width));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeof(float) * height * width));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeof(float) * height * width));  

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(float) * height * width, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(float) * height * width, cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y); 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height);
    CHECK_KERNEL_ERROR();

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * height * width, cudaMemcpyDeviceToHost));

    printf("First 5 elements of result: \n");
    for (int i = 0; i < 5; i++) {
        printf("%f\t", h_C[i]);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    clock_t start_cpu, stop_cpu;
    double cpu_time_used;
    start_cpu = clock();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_C[i * width + j] = 0;
            for (int k = 0; k < height; k++) { 
                h_C[i * width + j] += h_A[i * height + k] * h_B[k * width + j];
            }
        }
    }

    printf("\nFirst 5 elements of result (CPU): \n");
    for (int i = 0; i < 5; i++) {
        printf("%f\t", h_C[i]);
    }

    stop_cpu = clock();
    cpu_time_used = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("\n\nElapsed Time(CPU): %f milliseconds", cpu_time_used);
    printf("\nElapsed Time(GPU): %.3f milliseconds", milliseconds);
    float x_faster = cpu_time_used / milliseconds;
    printf("\nGPU Implementation is %.2fx faster than CPU Implementation\n", x_faster);

    return 0;
}