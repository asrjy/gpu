#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "helpers.h"

__global__
void divergent_kernel(int* data, int n){
    // kernel with warp divergence
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        if (idx % 2 == 0){
            data[idx] *= 2;
        } else {
            data[idx] += 1;
        }
    }
}

__global__
void optimized_kernel(int* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        // process indices as well
        if (idx%2 == 0){
            data[idx] *= 2;
        }
        // synchronize threads within warp
        __syncwarp();
        if (idx%2 != 0){
            data[idx] += 1;
        }
    }
}

void initialize_data(std::vector<int>& data){
    for(int i = 0; i < data.size(); ++i){
        data[i] = i;
    }
}

bool verify_results(const std::vector<int>& data){
    for (int i =0; i < data.size(); i++){
        int expected = (i % 2 == 0) ? i * 2: i+1;
        if(data[i] != expected){
            std::cout << "mismatch at " << i << ": " << data[i] << " != " << expected << std::endl;
            return false;
        }
    }
    return true; 
}

int main(){
    const int N = 1 << 24;
    const int BLOCK_SIZE = 256;

    std::vector<int> h_data(N);
    initialize_data(h_data);

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    divergent_kernel<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    float divergent_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&divergent_ms, start, stop));

    bool divergent_correct = verify_results(h_data);

    initialize_data(h_data);
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    optimized_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    float optimized_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&optimized_ms, start, stop));
    
    bool optimized_correct = verify_results(h_data);

    std::cout << "divergent kernel time: " << divergent_ms << " ms" << std::endl;
    std::cout << "optimized kernel time: " << optimized_ms << " ms" << std::endl;
    std::cout << "speedup: " << divergent_ms / optimized_ms << "x" << std::endl;

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;

}