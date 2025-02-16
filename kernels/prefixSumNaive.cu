#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include "helpers.h"

__global__
void parallel_scan_naive(float *x, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int d = 0; d < log2(n); d++){
        if(idx >= pow(2, d)) {
            x[idx] += x[idx - (int)pow(2, d)];
        }
    }
    __syncthreads();
}

int main(){
    int n = 16;
    std::vector<float> h_x(n);
    for(int i = 0; i < 16; ++i){
        h_x[i] = (float)i+1;
    }
    std::vector<float> h_y(n);

    float *d_x;
    CUDA_CHECK(cudaMalloc((void**)&d_x, sizeof(float) * n));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = n;
    dim3 blockDim(blockSize);
    dim3 gridDim(1);

    parallel_scan_naive<<<gridDim, blockDim>>>(d_x, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_x[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < n; ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_x));
    return 0;
}