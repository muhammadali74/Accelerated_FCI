#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

#define blocksize 8

__global__ void nodiag_normalize(double *A, double *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == i && x != y)
        {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }
}

__global__ void diag_normalize(double *A, double *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == y && x == i)
        {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }
}

__global__ void gaussjordan(double *A, double *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
    {
        if (x != i)
        {
            I[x * n + y] -= I[i * n + y] * A[x * n + i];
            if (y != i)
            {
                A[x * n + y] -= A[i * n + y] * A[x * n + i];
            }
        }
    }
}

__global__ void set_zero(double *A, double *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
    {
        if (x != i)
        {
            if (y == i)
            {
                A[x * n + y] = 0;
            }
        }
    }
}

int main()
{
    const int n = 8;
    // creating input
    double *iL = new double[n * n];
    double L[] = {1.000000, 0.890520, 0.008822, -0.028730, 0.932269, 0.025409, 0.878241, -0.795243, 0.890520, 1.000000, 0.018756, -0.041665, 0.836217, 0.019652, 0.769136, -0.676597, 0.008822, 0.018756, 1.000000, -0.436597, -0.166942, 0.319497, 0.000330, 0.165078, -0.028730, -0.041665, -0.436597, 1.000000, 0.038607, -0.729447, -0.011744, 0.330830, 0.932269, 0.836217, -0.166942, 0.038607, 1.000000, -0.133347, 0.825897, -0.792844, 0.025409, 0.019652, 0.319497, -0.729447, -0.133347, 1.000000, -0.077962, -0.419188, 0.878241, 0.769136, 0.000330, -0.011744, 0.825897, -0.077962, 1.000000, -0.725739, -0.795243, -0.676597, 0.165078, 0.330830, -0.792844, -0.419188, -0.725739, 1.00000};

    // matrix_read(L, n);
    // savetofile(L, "L.txt", n, n);

    cout << "inv\n";
    double *d_A, *d_L, *I, *dI;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ddsize = n * n * sizeof(double);

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((n + blocksize - 1) / blocksize, (n + blocksize - 1) / blocksize);
    // memory allocation
    err = cudaMalloc((void **)&d_A, ddsize);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **)&dI, ddsize);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    I = new double[n * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
                I[i * n + i] = 1.0;
            else
                I[i * n + j] = 0.0;
        }
    }

    // copy data from CPU to GPU
    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    // timer start
    cudaEventRecord(start, 0);

    // L^(-1)
    for (int i = 0; i < n; i++)
    {
        nodiag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
        diag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
        gaussjordan<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
        set_zero<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy data from GPU to CPU
    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    cout << "Cuda Time - inverse: " << time << "ms\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << iL[i * n + j] << " ";
        }
        cout << endl;
    }
    // savetofile(I, "I.txt", n, n);
    cudaFree(d_A);
    cudaFree(dI);

    delete[] I;
    delete[] L;
    delete[] iL;

    system("Pause");
    return 0;
}