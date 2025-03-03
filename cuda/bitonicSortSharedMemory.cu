#include <stdio.h>
#include <cuda_runtime.h>
#include "helpers.h"

__global__
void bitonic_sort_shared_memory_kernel(int* g_data, const int N){
    extern __shared__ int shared_data[];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // load data from global memory to shared memory 
    if (gid < N){
        shared_data[tid] = g_data[gid];
    }
    __syncthreads();

    // k is the current power of 2 sub-array size
    for(int k = 2; k <= blockDim.x; k *= 2){
        // j is the size of pairs to compare
        for(int j = k/2; j > 0; j /= 2){
            // calculate partner index
            int ixj = tid ^ j;

            // ensuring we are only comparing within the current block
            if (ixj > tid){
                // determine sort direction
                bool ascending = ((tid & k) == 0);

                // compare and swap if needed 
                if (ascending == (shared_data[tid] > shared_data[ixj])){
                    // swap 
                    int temp = shared_data[tid];
                    shared_data[tid] = shared_data[ixj];
                    shared_data[ixj] = temp;
                }
            }
        }
        __syncthreads();
    }

    // write results back to global memory
    if (gid < N){
        g_data[gid] = shared_data[tid];
    }
}

__global__
void bitonic_merge_kernel(int* data, const int N, int j, int k){
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N){
        unsigned int ixj = tid ^ j;
        if(ixj > tid){
            if (ixj < N){
                bool ascending = ((tid & k) == 0);
                if (ascending == (data[tid] > data[ixj])) {
                   int temp = data[tid];
                   data[tid] = data[ixj];
                   data[ixj] = temp; 
                }
            }
        }
    }
}

void bitonic_sort_shared(int* d_data, const int N){
    const int BLOCK_SIZE = 256;

    int grid_size = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;

    bitonic_sort_shared_memory_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, N);

    for(int k = BLOCK_SIZE * 2; k <= N; k *= 2){
        for(int j = k/2; j > 0; j /= 2){
            bitonic_merge_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, N, j , k);
            cudaDeviceSynchronize();
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

    bitonic_sort_shared(d_data, N);

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