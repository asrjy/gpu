#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "helpers.h"

__global__
void hillis_steele_scan(float *x, float *y, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // initially copy the input into the output array
    if(idx < n){
        y[idx] = x[idx];    
    }
    __syncthreads();

    for(int d = 0; d < log2(n); ++d){
        if (idx < n){
            if (idx >= pow(2, d)) {
                y[idx] = y[idx] + y[idx - (int)pow(2, d)];
            }
        }
        __syncthreads();
    }
}

__global__
void hillis_steele_scan_shared(float *g_idata, float *g_odata, int n){
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load data from global to shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __synchthreads();

    // each thread loads one element from global memory into shared memory
    for(int d = 0; d < log2(blockDim.x); ++d){
        if(tid < blockDim.x){
            if (tid >= pow(2, d)){
                sdata[tid] = sdata[tid] + sdata[tid - (int)pow(2, d)];
            }
        }
        __synchthreads();
    }
    if ( i < n){
        g_odata[i] = sdata[tid];
    }
}

int main(){
    int n = 1024;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;

    std::vector <float> h_idata(n);
    for(int i = 0; i < n; ++i){
        h_idata[i] = (float)i + 1;
    }
    std::vector <float> h_odata(n);

    float *d_idata, *d_odata;
    CUDA_CHECK(cudaMalloc((void**)&d_idata, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_odata, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedMemSize = blockSize * sizeof(float);

    hillis_steele_scan_shared<<<numBlocks, blockSize, sharedMemSize>>>(d_idata, d_odata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_odata.data(), d_odata, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_idata[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_odata[i] << " ";
    }
    std::cout << std::endl;

    
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));

    return 0;
}