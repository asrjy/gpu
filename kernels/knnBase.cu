#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "helpers.h"

struct DistanceIndex {
    // structure for storing distance and index
    float distance;
    int index;
};

__device__
__host__
bool operator<(const DistanceIndex& a, const DistanceIndex& b) {
    // comparision operator for sorting
    return a.distance < b.distance;
}

__global__
void compute_distances(float* trainData, float* testPoint, DistanceIndex* distances, int numTrainingPoints, int numFeatures){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < numTrainingPoints){
        float distance = 0.0f;

        // compute the euclidean distance
        for(int i = 0; i < numFeatures; i++){
            float diff = trainData[tid * numFeatures + i] - testPoint[i];
            distance += diff * diff;
        }

        distances[tid].distance = sqrt(distance);
        distances[tid].index = tid;
    }
}

__global__
void compute_distances_shared(float* trainData, float* testPoint, DistanceIndex* distances, int numTrainingPoints, int numFeatures){
    extern __shared__ float sharedTest[];

    if(threadIdx.x < numFeatures){
        sharedTest[threadIdx.x] = testPoint[threadIdx.x];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < numTrainingPoints){
        float distance = 0.0f;

        for(int i  = 0; i < numFeatures; i++){
            float diff = trainData[tid * numFeatures + i] - sharedTest[i];
            distance += diff * diff;
        }

        distances[tid].distance = sqrt(distance);
        distances[tid].index = tid;
    }
}

__global__
void compute_distances_tiled(float* trainData, float* testPoint, DistanceIndex* distances, int numTrainingPoints, int numFeatures){
    extern __shared__ float shared[];
    float* sharedTest = shared;
    float* sharedTrain = &shared[numFeatures];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(threadIdx.x < numFeatures){
        sharedTest[threadIdx.x] = testPoint[threadIdx.x];
    }

    float distance = 0.0f;

    for(int tile = 0; tile < numFeatures; tile += blockDim.x){
        // load training data tile 
        if(tid < numTrainingPoints && (tile + threadIdx.x) < numFeatures){
            sharedTrain[threadIdx.x] = trainData[tid * numFeatures + tile + threadIdx.x];
        }
        __syncthreads();

        if(tid < numTrainingPoints){
            for(int i = 0; i < blockDim.x && (tile + i) < numFeatures; i++){
                float diff = sharedTrain[i] - sharedTest[tile + i];
                distance += diff * diff;
            }
        }
        __syncthreads();
    }
    if(tid < numTrainingPoints){
        distances[tid].distance = sqrt(distance);
        distances[tid].index = tid;
    }
}

__global__
void compute_distances_batch(float* traiNData, float* testData, DistanceIndex* distances, int numTrainingPoints, int numTestPoints, int numFeatures){
    int trainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int testIdx = blockIdx.y;

    if(trainIdx < numTrainingPoints){
        float distance = 0.0f;
        for(int i = 0; i < numFeatures; i++){
            float diff = trainData[trainIdx * numFeatures + i] - testData[testIdx * numFeatures + i];
            distance += diff * diff;
        }
        distances[testIdx * numTrainingPoints + trainIdx].distance = sqrt(distance);
        distances[testIdx * numTrainingPoints + trainIdx].indx = trainIdx;
    }
}

void knn_cuda(float* h_trainData, float* h_testPoint, int numTrainingPoints, int numFeatures, int k){
    float *d_trainData, *d_testPoint;
    DistanceIndex *d_distances;

    cudaMalloc(&d_trainData, numTrainingPoints * numFeatures * sizeof(float));
    cudaMalloc(&d_testPoint, numFeatures * sizeof(float));
    cudaMalloc(&d_distances, numTrainingPoints * sizeof(DistanceIndex));

    cudaMemcpy(d_trainData, h_trainData, numTrainingPoints * numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testPoint, h_testPoint, numFeatures * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numTrainingPoints + blockSize - 1) / blockSize;

    compute_distances<<<numBlocks, blockSize>>>(d_trainData, d_testPoint, d_distances, numTrainingPoints, numFeatures);
    DistanceIndex* h_distances = new DistanceIndex[numTrainingPoints];
    cudaMemcpy(h_distances, d_distances, numTrainingPoints * sizeof(DistanceIndex), cudaMemcpyDeviceToHost);
    
    // sort distances 
    std::sort(h_distances, h_distances + numTrainingPoints);
    
    printf("K nearest neighbors:\n");
    for (int i = 0; i < k; i++) {
        printf("Index: %d, Distance: %f\n", 
               h_distances[i].index, h_distances[i].distance);
    }

    cudaFree(d_trainData);
    cudaFree(d_testPoint);
    cudaFree(d_distances);
    delete[] h_distances;
}

void knn_cuda_thrust(float* h_trainData, float* h_testPoint, int numTrainingPoints, int numFeatures, int k){
    float *d_trainData, *d_testPoint;
    DistanceIndex *d_distances;

    cudaMalloc(&d_trainData, numTrainingPoints * numFeatures * sizeof(float));
    cudaMalloc(&d_testPoint, numFeatures * sizeof(float));
    cudaMalloc(&d_distances, numTrainingPoints * sizeof(DistanceIndex));

    cudaMemcpy(d_trainData, h_trainData, numTrainingPoints * numFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_testPoint, h_testPoint, numFeatures * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numTrainingPoints + blockSize - 1) / blockSize;

    compute_distances_shared<<<numBlocks, blockSize, (numFeatures + blockSize) * sizeof(float)>>>(d_trainData, d_testPoint, d_distances, numTrainingPoints, numFeatures);

    thrust::device_ptr<DistanceIndex> thrust_distances(d_distances);
    thrust::sort(thrust_distances, thrust_distances + numTrainingPoints);

    // copying only k nearest back to host 
    DistanceIndex* h_distances = new DistanceIndex[k];
    cudaMemcpy(h_distances, d_distances, k * sizeof(DistanceIndex), cudaMemcpyDeviceToHost);

    // sort distances 
    std::sort(h_distances, h_distances + numTrainingPoints);
    
    printf("K nearest neighbors:\n");
    for (int i = 0; i < k; i++) {
        printf("Index: %d, Distance: %f\n", 
               h_distances[i].index, h_distances[i].distance);
    }

    cudaFree(d_trainData);
    cudaFree(d_testPoint);
    cudaFree(d_distances);
    delete[] h_distances;
}

int main() {
    // setting random seed
    srand(time(0));

    const int numTrainingPoints = 1000000;  
    const int numFeatures = 128;            
    const int k = 5;                        

    float* h_trainData = new float[numTrainingPoints * numFeatures];
    float* h_testPoint = new float[numFeatures];

    // initializing training data with random values between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < numTrainingPoints * numFeatures; i++) {
        h_trainData[i] = dis(gen);
    }

    // random test point creation
    for (int i = 0; i < numFeatures; i++) {
        h_testPoint[i] = dis(gen);
    }

    // cpu verification; only for small datasets
    if (numTrainingPoints <= 10000) {  
        std::vector<DistanceIndex> cpu_distances(numTrainingPoints);
        
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        // compute distances on cpu
        for (int i = 0; i < numTrainingPoints; i++) {
            float distance = 0.0f;
            for (int j = 0; j < numFeatures; j++) {
                float diff = h_trainData[i * numFeatures + j] - h_testPoint[j];
                distance += diff * diff;
            }
            cpu_distances[i].distance = sqrt(distance);
            cpu_distances[i].index = i;
        }
        
        // sort
        std::sort(cpu_distances.begin(), cpu_distances.end());
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        
        printf("CPU K nearest neighbors:\n");
        for (int i = 0; i < k; i++) {
            printf("Index: %d, Distance: %f\n", 
                   cpu_distances[i].index, cpu_distances[i].distance);
        }
        printf("CPU Time: %lld ms\n", cpu_duration.count());
    }

    // gpu knn
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    knn_cuda_thrust(h_trainData, h_testPoint, numTrainingPoints, numFeatures, k);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
    
    printf("GPU Time: %lld ms\n", gpu_duration.count());

    delete[] h_trainData;
    delete[] h_testPoint;

    return 0;
}
