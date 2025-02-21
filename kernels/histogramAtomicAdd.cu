#include <cuda_runtime.h>
#include <stdio.h>
#include "helpers.h"

__global__
void histogram_global_atomic(unsigned int* input, unsigned int* histogram, int n, int num_bins){
    // kernel using global atomics
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread will process one element
    if (tid < n){
        unsigned int item = input[tid];
        // atomic addition to appropriate bin
        // ensures the addition to histogram happens in one uninterrupted sequence
        atomicAdd(&histogram[item], 1);
    }
}

__global__
void histogram_incorrect(unsigned int* input, unsigned int* histogram, int n, int num_bins){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        unsigned int item = input[tid];
        // race condition
        // multiple threads try to do this at the same time. final result depends on order in which the threads execute
        histogram[item]++;
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
    histogram_global_atomic<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_histogram, N, NUM_BINS);
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