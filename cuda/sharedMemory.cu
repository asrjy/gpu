#include <stdio.h>
#include <cuda_runtime.h>
#include "helpers.h"

__global__ 
void sharedMemoryLoader(int* input, int* output, int N){
    __shared__ int shared_data[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < N){
        shared_data[threadIdx.x] = input[tid];
    }

    __syncthreads();

    if(tid < N){
        output[tid] = shared_data[threadIdx.x];
    }
}

int main(){
    int N = 256;
    int h_input[256], h_output[256];
    for(int i=0; i<N; i++){
        h_input[i] = i;
    }

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    sharedMemoryLoader<<<1,256>>>(d_input, d_output, N);
    CHECK_KERNEL_ERROR();
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N*sizeof(int), cudaMemcpyDeviceToHost));

    for(int i=0; i<10; i++){
        printf("%d ", h_output[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}