#include "helpers.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__
void vectorAbsDiffKernel(float *d_Output, const float *d_A, const float *d_B, int size){
    // automatic variable
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
        float diff = d_A[i] - d_B[i];
        float absDiff;
        // if negative, multiply with -1, else keep same. 
        if (diff >= 0){
            absDiff = diff;
        } else {
            absDiff = -diff;
        }
        d_Output[i] = absDiff;
    }
}

int main(){
    int size = 10;
    // memory size in size of floats
    size_t memSize = size * sizeof(float);

    std::vector<float> A(size), B(size), Output(size);

    // initializing with sample data
    for (int i=0; i < size; ++i){
        A[i] = i + 1.0f;
        B[i] = size - i;
    }

    std::cout << "values in vector A:";
    for (float val : A) 
        std::cout << val << " ";
    std::cout << "values in vector B:";
    for (float val : B) 
        std::cout << val << " ";

    // allocate device memory. "stub"
    float *d_A, *d_B, *d_Output;
    // creating a variable to check for cuda errors
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_A, memSize);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaMalloc d_A failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&d_B, memSize);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaMalloc d_B failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        // freeing already allocated d_A because we crashin
        cudaFree(d_A);
        return 1;
    }
    
    cudaStatus = cudaMalloc((void**)&d_Output, memSize);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaMalloc d_Output failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        // freeing d_A and d_B
        cudaFree(d_B);
        cudaFree(d_A);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_A, A.data(), memSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){
        std::cerr << "copy to d_A failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Output);
        return 1;
    }
    

    cudaStatus = cudaMemcpy(d_B, B.data(), memSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess){
        std::cerr << "copy to d_B failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Output);
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1)/threadsPerBlock;
    // creating cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // recording 'start' event at 0
    cudaEventRecord(start, 0);
    // launching the kernel
    vectorAbsDiffKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_A, d_B, size);

    CHECK_KERNEL_ERROR();

    // recording 'stop' event
    cudaEventRecord(stop, 0);
    // wait for stop event to finish to be recorded and gpu to finish
    cudaEventSynchronize(stop);


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess){
        std::cerr << "kernel launch failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Output);
        return 1;
    }

    // wait for kernel execution to complete before cudaMemcpyDeviceToHost
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaDeviceSynchronize failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Output);
    }

    // copy output
    cudaStatus = cudaMemcpy(Output.data(), d_Output, memSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaMemCpy d_Output to Output failed! " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_Output);
        return 1;
    }

    std::cout << "absolute difference: ";
    for (float val: Output) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "verification (host): ";
    for (int i = 0; i < size; ++i){
        std::cout << std::abs(A[i] - B[i]) << " ";
    }
    std::cout << std::endl;

    cudaStatus = cudaFree(d_A);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaFree d_A failed! " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaStatus = cudaFree(d_B);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaFree d_B failed! " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    cudaStatus = cudaFree(d_Output);
    if (cudaStatus != cudaSuccess){
        std::cerr << "cudaFree d_Output failed! " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // calculating execution time in milliseconds
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // milliseconds = milliseconds/100;

    printf("\n Kernel Execution Time: %.3f milliseconds\n", milliseconds);

    return 0;

}