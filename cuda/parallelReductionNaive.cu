#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "helpers.h"

__global__
void reduce_block(float* g_input_data, float* g_output_data, int n){
    // shared memory partial sums
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // this thread will load one element from global to shared memory
    // pad with 0 if out of limit
    sdata[tid] = (i < n) ? g_data[i]: 0;
    __syncthreads();

    // perform parallel reduction after all threads load data
    for(int s = blockDim.x/2; s > 0; s >>= 1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result back to global memory
    if (tid == 0){
        g_odata[blockDim.x] = sdata[0];
    }
}

int main(){
    int n = 1024 * 1024;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;

    std::vector<float> h_idata(n);
    for(int i =0; i < n; i++){
        h_idata[i] = (float)i;
    }

    std::vector<float> h_odata(numBlocks);

    float *d_idata, *d_odata;
    CUDA_CHECK(cudaMalloc((void**)&d_idata, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_odata, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedMemSize = blockSize * sizeof(float);

    reduce_block<<<numBlocks, blockSize, sharedMemSize>>>(d_idata, d_odata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_odata.data(), d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float totalSum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        totalSum += h_odata[i];
    }

    float expectedSum = (float)(n - 1) * n / 2.0f; // Sum of 0 to n-1
    std::cout << "Sum: " << totalSum << std::endl;
    std::cout << "Expected Sum: " << expectedSum << std::endl;

    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));

    return 0;
}