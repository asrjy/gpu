#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
using namespace std;

// error checking
inline cudaError_t checkCudaErrors(cudaError_t error){
    if(error != cudaSuccess){
        fprintf(stderr, "CUDA ERROR = %d: %s (%s:%d)\n", error, cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return error;
}


__global__
void imageBlurKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int kernelSize){
    // calculating thread's position in output image
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // checking if we are in boundaries
    if (row < height && col < width){
        float blurredPixelValue = 0.0f;
        int kernelRadius = kernelSize/2;
        // how much the value of each pixel should be multiplied by
        // if it's a 3x3 kernel, each value is multiplied by 1/9
        int kernelWeight = 1.0f / (kernelRadius * kernelRadius);

        for(int kernelRow = -kernelRadius; kernelRow <= kernelRadius; ++kernelRow){
            for(int kernelCol = -kernelRadius; kernelCol <= kernelRadius; ++kernelCol){
                int neighbourRow = row + kernelRow;
                int neighbourCol = col + kernelCol;

                int clampedRow = max(0, min(neighbourRow, height - 1));
                int clampedCol = max(0, min(neighbourCol, width - 1));

                // replication
                blurredPixelValue += inputImage[clampedRow * width + clampedCol] * kernelWeight;
            }
        }
        outputImage[row * width + col] = (unsigned char)blurredPixelValue;
    }
}

int main(){
    int width = 256;
    int height = 256;

    unsigned char* h_inputImage = new unsigned char[width * height];
    unsigned char* h_outputImage = new unsigned char[width * height];

    if(!h_inputImage || !h_outputImage){
        fprintf(stderr, "host memory allocation failed \n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < width * height; ++i){
        h_inputImage[i] = (i % 256);
    }

    unsigned char* d_inputImage;
    unsigned char* d_outputImage;

    checkCudaErrors(cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1)/ blockDim.x, (height + blockDim.y - 1)/blockDim.y);

    imageBlurKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, 3);
    checkCudaErrors(cudaGetLastError());
    // synchronizing after kernel launch helps catch kernel errors before host-device copy is done
    // ensures that kernel has completed
    // because cuda is asynchronous, this will help. 
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    printf("first few blurred pixel values are \n");
    for(int i =0; i < 10; ++i){
        printf("%d ", h_outputImage[i]);
    }
    printf("\n");

    // freeing memory
    checkCudaErrors(cudaFree(d_inputImage));
    checkCudaErrors(cudaFree(d_outputImage));
    delete[] h_inputImage;
    delete[] h_outputImage;

    return 0;
}