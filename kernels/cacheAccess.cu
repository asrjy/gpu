#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "helpers.h"

__global__
void cache_access_kernel(float *data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        // sequential access. cache friendly
        float val1 = data[idx];
        // load from cache. likely in l1 or l2 cache
        float val2 = data[idx];

        data[idx] = val1 + val2;
    }
}

__global__ 
void strided_access_kernel(float *data, int size, int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        int strided_idx = idx * stride;
        // strided access. less cache friendly
        if(strided_idx < size){
            // load from global memory. less chances of being in cache
            float val1 = data[strided_idx];
            // load (may be in cache but less likely)
            float val2 = data[strided_idx];

            data[strided_idx] = val1 * 2.0f;
        }
    }
}

int main(){
    int size = 1024 * 1024;
    int stride =64;

    std::vector<float> h_data(size);
    for(int i = 0; i < size; ++i){
        h_data[i] = (float)i;
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    int blocksize = 256;
    int numBlocks = (size + blockSize - 1)/blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(blockSize);

    cache_access_kernel<<<grdiDim, blockDim>>>(d_data, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    strided_access_kernel<<<gridDim, blockDim>>>(d_data, size, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));

    std::cout << "Kernels Executed. " << std::endl;

    return 0;
}