#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "helpers.h"

// block size in k dimension
const int BK = 32;
// block size in n dimension
const int BN = 32;  
// block size in m dimension
const int BM = 32; // Block size in M dimension
// tile multiple: tm results computed by each thread
const int TM = 4; 

__global__ void kernel4_1D_blocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // m dimension
    const uint blockRow = blockIdx.x; 
    // n dimension
    const uint blockCol = blockIdx.y; // N dimension

    // 1d thread id within the block
    const uint threadId = threadIdx.x;
    const uint threadRow = threadId / BN;
    const uint threadCol = threadId % BN;

    // calculate the row and column indices within the global C matrix
    const uint row = blockRow * BM + threadRow;
    const uint col = blockCol * BN + threadCol;

    // define shared memory for A and B tiles
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // allocate thread-local cache for results in register file
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // calculate inner row and column indices for A and B within the block
        uint innerRowA = threadRow;
        uint innerColA = threadCol;
        uint innerRowB = threadRow;
        uint innerColB = threadCol;

        // load data into shared memory (with bounds checking)
        if (innerRowA < BM && bkIdx + innerColA < K) {
            As[innerRowA * BK + innerColA] = A[row * K + (bkIdx + innerColA)];
        } else {
            As[innerRowA * BK + innerColA] = 0.0f; // Pad with 0 if out of bounds
        }
         if (bkIdx + innerRowB < K && innerColB < BN) {
            Bs[innerRowB * BN + innerColB] = B[(bkIdx + innerRowB) * N + col];
        } else {
            Bs[innerRowB * BN + innerColB] = 0.0f; // Pad with 0 if out of bounds
        }

        __syncthreads();

        // advance blocktile for outer loop
        // A += BK;
        // B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                if ((row+resIdx)<M){
                     threadResults[resIdx] +=
                        As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
                }

            }
        }
        __syncthreads();
    }
    // writing results back into global mem
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        if((row+resIdx)<M && col < N){
             C[(row+resIdx) * N + col] = alpha * threadResults[resIdx] + beta * C[(row+resIdx) * N + col];
        }

    }
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    // scaling factors
    float alpha = 1.0f;
    float beta  = 0.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // 1d threads per block
    dim3 blockDim(BN); 
    dim3 gridDim( (M + BM - 1) / BM, (N + BN - 1) / BN);

    kernel4_1D_blocktiling<<<gridDim, blockDim>>>(
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

    return 0;
}
