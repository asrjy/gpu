#include <cuda_runtime.h>
#include <stdio.h>
#include "helpers.h"

__global__
void bitonic_sort_kernel(int* data, int j, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;

    if (ixj > i){
        // sorting direction
        int dir = ((i & k) == 0);
        int ai = data[i];
        int bi = data[ixj];

        // compare and swap if needed
        if ((ai > bi) == dir){
            data[i] = bi;
            data[ixj] = ai;
        }
    }   
}

void bitonic_sort(int *d_data, int n){
    // outer loop for bitonic sequence length
    for (int k = 2; k <= n; k *= 2){
        // inner loop for sub-sequence length
        for (int j = k/2; j > 0; j /= 2){
            int numThreads = n/2;
            int blockSize = 256;
            int numBlocks = (numThreads + blockSize - 1)/blockSize;

            bitonic_sort_kernel<<<numBlocks, blockSize>>>(d_data, j, k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }    
    }
}

int main(){
    // best if power of 2
    const int N = 1024;

    int *h_data = new int[N];
    for(int i = 0; i < N; i++){
        h_data[i] = rand() % 1000;
    }

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    bitonic_sort(d_data, N);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool sorted = true;
    for(int i = 1; i < N; i++){
        if(h_data[i] < h_data[i-1]){
            // printf("\n%d, %d\n", h_data[i], h_data[i-1]);
            sorted = false;
            break;
        }
    }
    printf("Array is %s\n", sorted ? "sorted" : "not sorted");

    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}