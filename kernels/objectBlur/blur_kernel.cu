#include <stdio.h>

extern "C" { 
    __global__
    void blur_kernel(unsigned char *in, unsigned char *out, int width, int height, int blur_radius){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x<width && y<height){
            int sum = 0;
            int count = 0;
            for(int i = -blur_radius; i <= blur_radius; ++i){
                for(int j = -blur_radius; j <= blur_radius; ++j){
                    int nx = x + i;
                    int ny = y + j;

                    if(nx >= 0 && nx < width && ny >= 0 && ny < height){
                        sum += in[ny * width + nx];
                        count++;
                    }
                }
            }
            out[y * width + x] = (unsigned char)(sum/count);
        }
    }
}


