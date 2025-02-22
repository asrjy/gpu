#include <cuda_runtime.h>
#include <stdio.h>
#include "helpers.h"

__global__
void histogram_shared_memory(unsigned int* input, unsigned int* histogram, int n, int num_bins){
    // assuming 256 bins
    __shared__ unsigned int shared_hist[256];
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int lid = threadIdx.x;

    // initialize shared memory
    if (lid < num_bins){
        shared_hist[lid] = 0;
    }
    __syncthreads();

    // process int
    if (tid < n){
        unsigned int item = input[tid];
        atomicAdd(&shared_hist[item], 1);
    }
    __syncthreads();

    // merge shared memory histogram into shared memory
    if (lid < num_bins){
        atomicAdd(&histogram[lid], shared_hist[lid]);
    }
}

int main(){
    const int N = 1000000;
    const int NUM_BINS = 256;
    const int BLOCK_SIZE = 256;

    unsigned int* h_input = new unsigned int[N];
    unsigned int* h_histogram = new unsigned int[NUM_BINS]();
    unsigned int* h_hist_correct = new unsigned int[NUM_BINS]();

    // initializing with random values. ensuring no value exceeds number of bins
    for(int i = 0; i < N; i++) {
        h_input[i] = rand() % NUM_BINS;
    }

    // calucating on cpu for verification
    for(int i = 0; i < N; i++) {
        h_hist_correct[h_input[i]]++;
    }

    unsigned int *d_input, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(unsigned int) * N));
    CUDA_CHECK(cudaMalloc(&d_histogram, sizeof(unsigned int) * NUM_BINS));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // reset device histogram
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    histogram_shared_memory<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_histogram, N, NUM_BINS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Atomic Kernel Time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    bool correct = true;
    for(int i = 0; i < NUM_BINS; i++) {
        if(h_histogram[i] != h_hist_correct[i]) {
            printf("Mismatch at bin %d: %u != %u\n", 
                   i, h_histogram[i], h_hist_correct[i]);
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    printf("\nFirst 10 bins:\n");
    for(int i = 0; i < 10; i++) {
        printf("Bin %d: %u\n", i, h_histogram[i]);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_histogram));
    delete[] h_input;
    delete[] h_histogram;
    delete[] h_hist_correct;

    return 0;
}