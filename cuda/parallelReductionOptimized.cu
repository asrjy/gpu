#include <iostream>
#include <cuda_runtime.h>
#include "helpers.h"

__global__ 
void reduce_warp(float *g_idata, float *g_odata, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // warp id and lane id
    // WARP_SIZE is 32, standard in nvidia GPUs
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    float val = (i < n) ? g_idata[i] : 0.0f; // Load data

   // warp level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // 0xffffffff is the mask indicated all threads in the warp should participate in the shuffle
        // each bit in the mask corrresponds to the shuffle, we are setting it to 1 indicating it's active
        // val is the value to be shuffled (partial sum of each thread)
        // source lan = laneId + offset. each thread recieves the value of val from thread whose laneId is laneId + offset
        float other = __shfl_sync(0xffffffff, val, laneId + offset);
        val += other;
    }

    // one thread per warp. writing warp sum to shared memory
    extern __shared__ float warpSums[];
    if (laneId == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads(); 

    // normal block level reduction using shared memory
    if (warpId == 0) {
        if (tid < blockDim.x/WARP_SIZE){
            int sdataIndex = tid;
           
            float sdata[blockDim.x/WARP_SIZE];
             sdata[sdataIndex] = warpSums[sdataIndex];
              __syncthreads();
            for (int s = (blockDim.x/WARP_SIZE) / 2; s > 0; s >>= 1) {

                if (sdataIndex < s) {
                     sdata[sdataIndex] += sdata[sdataIndex + s];

                }
                 __syncthreads();
            }
             if (tid == 0){
                g_odata[blockIdx.x] = sdata[0];
            }
        }
    }
}


int main(){
    int n = 1024 * 1024;
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;
    // number of warps per block
    int numWarpsPerBlock = blockSie / WARP_SIZE;

    std::vector<float> h_idata(n);
    for (int i = 0; i < n; ++i) {
        h_idata[i] = (float)i; // Initialize data
    }
    std::vector<float> h_odata(numBlocks);

    float *d_idata, *d_odata;
    CUDA_CHECK(cudaMalloc((void**)&d_idata, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_odata, numBlocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_idata, h_idata.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedMemSize = numWarpsPerBlock * sizeof(float); // One float per warp
    reduce_warp<<<numBlocks, blockSize, sharedMemSize>>>(d_idata, d_odata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_odata.data(), d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float totalSum = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        totalSum += h_odata[i];
    }

    float expectedSum = (float)(n - 1) * n / 2.0f;
    std::cout << "Sum: " << totalSum << std::endl;
    std::cout << "Expected Sum: " << expectedSum << std::endl;

    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));

    return 0;

}