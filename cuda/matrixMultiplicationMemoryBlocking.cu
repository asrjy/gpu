#include <stdio.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include <iostream>
#include <vector>
#define CEIL_DIV(M, N) (((M) + (N)-1)/(N))

__global__
void sgemm_shared_memory_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    // the block of output matrix c, this block is responsible for
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // index of current thread within the block. 
    // using a single dimension for the thread index and calculating the row and column within the block
    // using division and modulo. 
    // this will ensure thread within a warp access memory in a contiguous manner (coalescing)
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // As and Bs are allocated in shared memory. 
    // As and Bs will hold a tile from A matrix and B matrix respectively
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE){
        // iterates through tiles of A and B that contribute to current block of C
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for( int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx){
            // actual dot product of A and B tiles
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        __syncthreads();
    }

    // applying the scaling factors
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main(){
    int M = 1024;
    int N = 1024;
    int K = 1024;
    const int BLOCKSIZE = 32;

    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N, 0.0f);
    
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    // scaling factors
    float alpha = 1.0f;
    float beta  = 0.0f;

    // allocate Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCKSIZE, BLOCKSIZE); // threads per block
    dim3 gridDim( (M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE); // blocks per grid

    sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaDeviceSynchronize()); 
    cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_C_cpu(M*N, 0.0f);

     for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                h_C_cpu[i * N + j] += h_A[i * K + k] * h_B[k * N + j];
            }
        }
    }
     float tolerance = 1e-5;
     bool match = true;
     for (int i = 0; i < M * N; ++i) {
        if (abs(h_C[i] - h_C_cpu[i]) > tolerance) {
            match = false;
            std::cout << "Mismatch at index " << i << ": GPU = " << h_C[i] << ", CPU = " << h_C_cpu[i] << std::endl;
            break;
        }
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "Matrix multiplication complete." << std::endl;

    return 0;
}