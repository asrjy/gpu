#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H
// include guards: prevent header from being included multiple times during compilation

#include <stdio.h>
#include <cuda_runtime.h>

// macro definition. takes one argument: call
// do while loop makes a multi statement macro and makes it behave like a single line statement
// executes the cuda function "call" and stores the return value in error
// if it is not a success, display the error and details in the standard error output stream
// EXIT_FAILURE terminates the program.
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t error = call;                                                           \
        if (error != cudaSuccess){                                                          \
            fprintf(stderr, "CUDA Error at %s: %d\n", __FILE__, __LINE__);                  \
            fprintf(stderr, "Error code: %d, %s\n", error, cudaGetErrorString(error));       \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)
#endif