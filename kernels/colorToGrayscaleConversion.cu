#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_ERROR_CHECK(cmd)   \
do{                             \
    cudaError_t error = cmd;    \
    if(error != cudaSuccess){   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;  \
        exit(EXIT_FAILURE);     \
    }                           \
} while(0)

// creating constants to be multiplied with r, g and b values
constexpr float R_TO_GRAY = 0.299f;
constexpr float G_TO_GRAY = 0.587f;
constexpr float B_TO_GRAY = 0.114f;

__global__
void rgb_to_grayscale_kernel(const float* __restrict__ rgb_image, float* __restrict__ grayscale_image, int width, int height){
    // calculating the global thread indices in 2d
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // checking if thread is within bounds
    if ( x < width && y < height){
        // M[i][j] = j * _ + i
        int pixel_index = y * width + x;
        // rgb components are interleaved
        int rgb_index = pixel_index * 3;

        // accessing rgb values
        float r = rgb_image[rgb_index];
        float g = rgb_image[rgb_index + 1];
        float b = rgb_image[rgb_index + 2];

        // applying grayscale conversion formula
        float gray = (r * R_TO_GRAY) + (g * G_TO_GRAY) + (b * B_TO_GRAY);

        // storing grascale value 
        grayscale_image[pixel_index] = gray;
    }
}

void rgb_to_grayscale(const float* rgb_image, float* grayscale_image, int width, int height){
    // stub code
    // creating variables to store images in device memory
    float *d_rgb_image, *d_grayscale_image;
    size_t image_size_bytes = width * height * sizeof(float);
    // for rgb
    size_t rgb_size_bytes = image_size_bytes * 3;

    // allocate device memory
    CUDA_ERROR_CHECK(cudaMalloc(&d_rgb_image, rgb_size_bytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_grayscale_image, image_size_bytes));

    // copy image to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_rgb_image, rgb_image, rgb_size_bytes, cudaMemcpyHostToDevice));

    // setting grid and block dimensions for kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width + blockDim.x - 1)/blockDim.x,
        (height + blockDim.y - 1)/blockDim.y
    );

    // launch kernel
    rgb_to_grayscale_kernel<<<gridDim, blockDim>>>(d_rgb_image, d_grayscale_image, width, height);
    // error checking for kernel is different. kernel isnt launched inside CUDAERRORCHECK because it's not assignable value
    CUDA_ERROR_CHECK(cudaGetLastError());

    // copy grayscale back to host
    CUDA_ERROR_CHECK(cudaMemcpy(grayscale_image, d_grayscale_image, image_size_bytes, cudaMemcpyDeviceToHost));
    
    // free device memory
    CUDA_ERROR_CHECK(cudaFree(d_rgb_image));
    CUDA_ERROR_CHECK(cudaFree(d_grayscale_image));
}

int main(){
    int width = 512;
    int height = 512;
    int num_pixels = width * height;

    // creating a sample gradient
    std::vector<float> rgb_image(num_pixels * 3);
    for(int y = 0; y < height; ++y){
        for (int x = 0; x < width; ++x){
            int index = (y * width + x) * 3;
            rgb_image[index] = (float)x / width;
            rgb_image[index + 1] = (float)y / height;
            rgb_image[index + 2] = 0.5f;

        }
    }

    // allocate space for grayscale image on host
    std::vector<float> grayscale_image(num_pixels);

    // perform conversion
    // .data() returns a pointer to the vector
    rgb_to_grayscale(rgb_image.data(), grayscale_image.data(), width, height);

    // verifying 
    std::cout << "first 10 values of grayscale image " << std::endl;
    for (int i = 0; i < 10; ++i){
        std::cout << grayscale_image[i] << " ";
    }
    std::cout << std::endl;
    return 0;

}