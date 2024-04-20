#include <iostream>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#define BLOCK_SIZE 16

__global__ void kernel()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    printf("Hello from block %d, thread %d\n", i, j);
}

__device__ void kernelcall()
{
    kernel<<<10, 1>>>();
}

__global__ void mainkernel()
{
    kernelcall();
}

int main()
{
    mainkernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}