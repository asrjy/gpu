#include "helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__
void vectorMultiply(float *d_Output, const float *d_A, const float *d_B, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
        d_Output[i] = d_A[i] * d_B[i];
    }
}

__host__
int main(){
    int size = 1000000;
    size_t sizeBytes = size * sizeof(float);

    float *h_A, *h_B, *h_Output;
    float *d_A, *d_B, *d_Output;

    h_A = (float *)malloc(sizeBytes);
    h_B = (float *)malloc(sizeBytes);
    h_Output = (float *)malloc(sizeBytes);

    for(int i = 0; i < size; ++i){
        h_A[i] = i;
        h_B[i] = size - i;
    }

    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_Output, sizeBytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock -  1)/threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_A, d_B, size);

    cudaError_t kernelLaunchCheck = cudaGetLastError();
    if (kernelLaunchCheck != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s: %d\n", __FILE__, __LINE__);                  
        fprintf(stderr, "Error code: %d, %s\n", kernelLaunchCheck, cudaGetErrorString(kernelLaunchCheck));       
        exit(EXIT_FAILURE);                              
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_Output, d_Output, sizeBytes, cudaMemcpyDeviceToHost);

    printf("Verifying Results: \n");
    for (int i = 0; i < 10; ++i){
        printf("a[%d] = %f, b[%d] = %f, a[%d] * b[%d] = %f\n",i, h_A[i], i, h_B[i], i, i, h_Output[i]);
    }

    printf("Time Elapsed (GPU): %.3f seconds \n", milliseconds);

    clock_t start_cpu, stop_cpu;
    double cpu_time_used;
    start_cpu = clock();

    for (int i =0; i < size; i++){
        h_Output[i] = h_A[i] * h_B[i];
    }

    stop_cpu = clock();
    cpu_time_used = ((double)(stop_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    printf("Time Elapsed (CPU): %.3f\n", cpu_time_used);


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Output));

    free(h_A);
    free(h_B);
    free(h_Output);

    return 0;
}