#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
// #include <device_functions.h>

__device__ void BINOM(int n, int k, int *out)
{
    int P, N1, R;
    // between n - k and k, N1 should be Max(n-k, k) and P should be Min(n-k, k);
    N1 = k;
    P = n - k;
    if (N1 <= P)
    {
        N1 = P;
        P = k;
    }
    if (P == 0)
    {
        R = 1;
    }
    else if (P == 1)
    {
        R = N1 + 1;
    }
    else
    {
        R = N1 + 1;
        for (int i = 2; i < (P + 1); i++)
        {
            R = (R * (N1 + i)) / i;
        }
    }
    *out = R;
}

__device__ void IthCombination(int out[], int N, int P, int L, int skip)
{
    // The out[p] can be calculated  using formula out[p] = out[p - 1] + L - K. note that out[p] is in 1-base indexing
    int P1 = P - 1;
    int R;
    int k = 0;
    for (int i = 0; i < P1; i++)
    {
        out[i] = 0;
        if (i > 0)
        {
            out[i] = out[i - 1];
        }
        while (k < L)
        {
            out[i] = out[i] + 1;
            BINOM(N - out[i], P - (i + 1), &R);
            k = k + R;
        }
        k = k - R;
    }
    out[P1] = out[P1 - 1] + L - k;
    for (int i = 0; i < P; i++)
    {
        if (out[i] >= skip)
            out[i] = out[i] + 1;
    }
}

__global__ void kernel(int *d_out, int N, int P, int L)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    IthCombination(d_out, 6, 1, idx + 1, 1);
}

int main()
{
    int N = 6; // total size to choose from it is actually N choose P
    int P = 1; // P is the size of combinations we want
    int L = 0; // controls the ith cobination from n choose p
    int *d_out;
    int *h_out = (int *)malloc(P * sizeof(int));
    cudaMalloc(&d_out, P * sizeof(int));
    kernel<<<1, 1>>>(d_out, N, P, L);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, P * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < P; i++)
    {
        printf("%d ", h_out[i]);
    }
    printf("\n");
    return 0;
}