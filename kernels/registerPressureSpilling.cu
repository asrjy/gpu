#include <cuda_runtime.h>
#include <stdio.h>
#include "helpers.h"

__global__
void low_register_pressure(float* input, float* output, int n){
    // low resigter pressure
    // calculating the square and saving it in the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float temp = input[idx];
        output[idx] = temp * temp;
    }
}

__global__ 
void high_register_pressure(float* input, float* output, int n){
    // high register pressure 
    // too many intermediate calculation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        float a = input[idx];
        float b = a * a;
        float c = b * b;
        float d = c * c;
        float e = d * d;
        float f = e * e;
        float g = f * f;
        float h = g * g;
        output[idx] = h;
    }
}

__global__
void register_spilling_kernel(float* input, float* output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // large local array forcing register spilling to local memory
    float local_array[100];

    if (idx < n){
        for(int i = 0; i < 100; ++i){
            local_array[i] = input[idx] * i;
        }
        float sum = 0.0f;
        for(int i = 0; i < 100; i++){
            sum += local_array[i];
        }
        output[idx] = sum;
    }
}

int main(){
    const int N = 1<<20; 
    const int BLOCK_SIZE = 256;

    float* h_input = new float[N];
    float* h_output = new float[N];

    for(int i = 0; i < N; i++){
        h_input[i] = float(i);
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // low register pressure kernel;
    CUDA_CHECK(cudaEventRecord(start));
    low_register_pressure<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Low Register Kernel Time: %f ms\n", milliseconds);

    // high register pressure kernel;
    CUDA_CHECK(cudaEventRecord(start));
    high_register_pressure<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("High Register Kernel Time: %f ms\n", milliseconds);

    // register spilling kernel;
    CUDA_CHECK(cudaEventRecord(start));
    register_spilling_kernel<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Register Spilling Kernel Time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
    return 0;
}