#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
// Function to find the inverse of a matrix using cuBLAS
#define blocksize 8
#define numnodes 15

__global__ void nodiag_normalize(float *A, float *I, int n, int i)
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

__global__ void diag_normalize(float *A, float *I, int n, int i)
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

__global__ void gaussjordan(float *A, float *I, int n, int i)
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

__global__ void set_zero(float *A, float *I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
    {
        A[x * n + y] = -A[x * n + y];
        if (x != i)
        {
            if (y == i)
            {
                A[x * n + y] = 0;
            }
        }
    }
}

__device__ double norm_cdf(double x)
{
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

__constant__ float corr_coef[] = {1.00000000e+00, -4.83579373e-01, -2.10180536e-02, 8.94362611e-01,
                                  4.24590545e-02, -4.39537552e-02, -3.72566578e-02, 2.28322056e-01,
                                  -2.70536328e-02, 4.27293493e-02, 1.73825003e-02, -6.50802196e-01,
                                  5.91756144e-01, -2.06959582e-02, 2.70788093e-01, -4.83579373e-01,
                                  1.00000000e+00, -1.40257075e-02, -4.36597452e-01, -3.68239991e-02,
                                  1.87559381e-02, -7.29248800e-03, -4.84051959e-01, 8.82179475e-03,
                                  -1.66941916e-01, -3.12556246e-02, 3.19497360e-01, -5.70704583e-01,
                                  3.30163719e-04, 1.65078186e-01, -2.10180536e-02, -1.40257075e-02,
                                  1.00000000e+00, -3.79522885e-02, 2.21022082e-02, -4.55431387e-02,
                                  -6.09293492e-02, -4.23240270e-01, -4.86054643e-02, -5.61390005e-02,
                                  2.58086740e-02, 3.38029706e-01, -3.27965626e-01, -3.25053917e-01,
                                  -7.13265157e-02, 8.94362611e-01, -4.36597452e-01, -3.79522885e-02,
                                  1.00000000e+00, 3.39409193e-02, -4.16645390e-02, -4.80159297e-02,
                                  2.25022448e-01, -2.87299150e-02, 3.86072068e-02, 6.06648779e-03,
                                  -7.29446968e-01, 5.48572747e-01, -1.17443722e-02, 3.30830194e-01,
                                  4.24590545e-02, -3.68239991e-02, 2.21022082e-02, 3.39409193e-02,
                                  1.00000000e+00, -7.14733633e-01, -5.54449739e-03, 2.68961648e-02,
                                  -6.36551329e-01, -6.06022267e-01, 8.52044177e-01, -2.15715881e-02,
                                  2.66669065e-02, -5.53971922e-01, 4.87876124e-01, -4.39537552e-02,
                                  1.87559381e-02, -4.55431387e-02, -4.16645390e-02, -7.14733633e-01,
                                  1.00000000e+00, -2.48611305e-03, -1.70829275e-03, 8.90519938e-01,
                                  8.36217278e-01, -6.06831019e-01, 1.96516325e-02, 3.76421899e-05,
                                  7.69136293e-01, -6.76596960e-01, -3.72566578e-02, -7.29248800e-03,
                                  -6.09293492e-02, -4.80159297e-02, -5.54449739e-03, -2.48611305e-03,
                                  1.00000000e+00, 8.67387944e-03, -5.01083111e-02, 1.55556766e-01,
                                  2.96596364e-02, -4.99757117e-01, 1.44895213e-02, -1.35745663e-02,
                                  1.75474021e-01, 2.28322056e-01, -4.84051959e-01, -4.23240270e-01,
                                  2.25022448e-01, 2.68961648e-02, -1.70829275e-03, 8.67387944e-03,
                                  1.00000000e+00, 1.15323811e-02, 1.01365962e-01, 1.90223690e-02,
                                  -3.02449537e-01, 8.38801255e-01, 1.38380137e-01, -4.84940815e-02,
                                  -2.70536328e-02, 8.82179475e-03, -4.86054643e-02, -2.87299150e-02,
                                  -6.36551329e-01, 8.90519938e-01, -5.01083111e-02, 1.15323811e-02,
                                  1.00000000e+00, 9.32268543e-01, -5.57680706e-01, 2.54087157e-02,
                                  1.73894000e-02, 8.78240514e-01, -7.95243192e-01, 4.27293493e-02,
                                  -1.66941916e-01, -5.61390005e-02, 3.86072068e-02, -6.06022267e-01,
                                  8.36217278e-01, 1.55556766e-01, 1.01365962e-01, 9.32268543e-01,
                                  1.00000000e+00, -5.24494555e-01, -1.33347391e-01, 1.21034969e-01,
                                  8.25896782e-01, -7.92844330e-01, 1.73825003e-02, -3.12556246e-02,
                                  2.58086740e-02, 6.06648779e-03, 8.52044177e-01, -6.06831019e-01,
                                  2.96596364e-02, 1.90223690e-02, -5.57680706e-01, -5.24494555e-01,
                                  1.00000000e+00, -2.29352819e-02, 1.84045112e-02, -4.91222791e-01,
                                  4.29002421e-01, -6.50802196e-01, 3.19497360e-01, 3.38029706e-01,
                                  -7.29446968e-01, -2.15715881e-02, 1.96516325e-02, -4.99757117e-01,
                                  -3.02449537e-01, 2.54087157e-02, -1.33347391e-01, -2.29352819e-02,
                                  1.00000000e+00, -5.16421169e-01, -7.79618284e-02, -4.19188374e-01,
                                  5.91756144e-01, -5.70704583e-01, -3.27965626e-01, 5.48572747e-01,
                                  2.66669065e-02, 3.76421899e-05, 1.44895213e-02, 8.38801255e-01,
                                  1.73894000e-02, 1.21034969e-01, 1.84045112e-02, -5.16421169e-01,
                                  1.00000000e+00, 1.18242925e-01, 6.15757820e-02, -2.06959582e-02,
                                  3.30163719e-04, -3.25053917e-01, -1.17443722e-02, -5.53971922e-01,
                                  7.69136293e-01, -1.35745663e-02, 1.38380137e-01, 8.78240514e-01,
                                  8.25896782e-01, -4.91222791e-01, -7.79618284e-02, 1.18242925e-01,
                                  1.00000000e+00, -7.25739319e-01, 2.70788093e-01, 1.65078186e-01,
                                  -7.13265157e-02, 3.30830194e-01, 4.87876124e-01, -6.76596960e-01,
                                  1.75474021e-01, -4.84940815e-02, -7.95243192e-01, -7.92844330e-01,
                                  4.29002421e-01, -4.19188374e-01, 6.15757820e-02, -7.25739319e-01,
                                  1.00000000e+00};

__device__ float calculate_statistic(int num_records, int x, int y, int *zz, int n)
{
    float par_corr;

    // Extract correlation coefficients for readability
    float corr_xy = corr_coef[x * numnodes + y];
    float corr_xz = corr_coef[x * numnodes + zz[0]];
    float corr_yz = corr_coef[y * numnodes + zz[0]];

    if (n == 0) // never reached
    {
        if (corr_xy >= 1.0)
            //*result = 0;
            return 0;
        par_corr = corr_xy;
    }
    else if (n == 1)
    {
        if (corr_xz >= 1.0 || corr_yz >= 1.0)
            return 0;
        else
            par_corr = (corr_xy - corr_xz * corr_yz) / sqrtf((1 - pow(corr_xz, 2)) * (1 - pow(corr_yz, 2)));
    }
    else
    {
        // Calculate partial correlation coefficient

        // Combine indices: (x, y) + zz
        int *all_var_idx;
        cudaMalloc((void **)&all_var_idx, (n + 2) * sizeof(int));
        all_var_idx[0] = x;
        all_var_idx[1] = y;
        for (int i = 0; i < n; ++i)
        {
            all_var_idx[i + 2] = zz[i];
        }
        // printf("all_var_idx: %d %d %d %d %d %d %d %d\n", all_var_idx[0], all_var_idx[1], all_var_idx[2], all_var_idx[3], all_var_idx[4], all_var_idx[5], all_var_idx[6], all_var_idx[7]);

        // Allocate memory for correlation coefficients subset
        float *corr_coef_subset;
        cudaMalloc((void **)&corr_coef_subset, (n + 2) * (n + 2) * sizeof(float));

        // Copy relevant subset of correlation coefficients to device
        for (int i = 0; i < n + 2; ++i)
        {
            for (int j = 0; j < n + 2; ++j)
            {
                corr_coef_subset[i * (n + 2) + j] = corr_coef[all_var_idx[i] * numnodes + all_var_idx[j]];
                // printf("%f ", corr_coef_subset[i * (n + 2) + j]);
            }
        }

        // Calculate inverse of correlation coefficients subset
        float *inv_corr_coef;
        cudaMalloc((void **)&inv_corr_coef, (n + 2) * (n + 2) * sizeof(float));
        for (int i = 0; i < n + 2; i++)
        {
            for (int j = 0; j < n + 2; j++)
            {
                if (i == j)
                    inv_corr_coef[i * (n + 2) + i] = 1.0;
                else
                    inv_corr_coef[i * (n + 2) + j] = 0.0;
            }
        }
        dim3 threadsPerBlock(blocksize, blocksize);
        dim3 numBlocks((n + 2 + blocksize - 1) / blocksize, (n + 2 + blocksize - 1) / blocksize);

        for (int i = 0; i < n + 2; i++)
        {
            nodiag_normalize<<<numBlocks, threadsPerBlock>>>(corr_coef_subset, inv_corr_coef, n + 2, i);
            diag_normalize<<<numBlocks, threadsPerBlock>>>(corr_coef_subset, inv_corr_coef, n + 2, i);
            gaussjordan<<<numBlocks, threadsPerBlock>>>(corr_coef_subset, inv_corr_coef, n + 2, i);
            set_zero<<<numBlocks, threadsPerBlock>>>(corr_coef_subset, inv_corr_coef, n + 2, i);
        }

        // Calculate partial correlation coefficient
        par_corr = fabsf(inv_corr_coef[0 * (n + 2) + 1] / sqrtf(fabsf(inv_corr_coef[0 * (n + 2) + 0] * inv_corr_coef[1 * (n + 2) + 1])));
        printf("par_corr: %f\n", par_corr);

        // Free allocated memory
        cudaFree(corr_coef_subset);
        cudaFree(inv_corr_coef);

        // Assign result
        // *result = par_corr;
    }

    if (par_corr >= 1.0)
        return 0;
    else if (par_corr <= 0)
        return INFINITY;

    int deg_of_freedom = num_records - n - 2;
    float z = 0.5 * log1pf((2 * par_corr) / (1 - par_corr));
    float val_for_cdf = abs(sqrtf(deg_of_freedom - 1) * z);
    float statistics = 2 * (1 - norm_cdf(val_for_cdf));
    return statistics;
}

__global__ void calculate_kernel(int num_records, int x, int y, int *zz, int n, float *result)
{
    *result = calculate_statistic(num_records, x, y, zz, n);
}

int main()
{
    const int num_records = 1000; // Example number of records
    const int x = 8;              // Example index of x
    const int y = 5;              // Example index of y
    // const int m = 4;                        // Example value of m
    const int n = 6;                                    // Example size of correlation matrix
    const int zz[] = {1, 3, 9, 11, 13, 14}; // Example array for z

    // Allocate memory for correlation coefficients
    // float *corr_coef;
    // cudaMalloc((void **)&corr_coef, num_records * num_records * sizeof(float));

    // Fill correlation coefficients with dummy data
    // Example: Fill with random values for demonstration
    // float dummy_data[num_records * num_records];
    // for (int i = 0; i < num_records * num_records; ++i)
    // {
    //     dummy_data[i] = static_cast<float>(rand()) / RAND_MAX;
    // }
    // cudaMemcpy(corr_coef, dummy_data, num_records * num_records * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for zz on device and copy data
    int *zz_device;
    cudaMalloc((void **)&zz_device, sizeof(zz));
    cudaMemcpy(zz_device, zz, sizeof(zz), cudaMemcpyHostToDevice);

    // Allocate memory for result
    float *result;
    cudaMalloc((void **)&result, sizeof(float));

    // Launch kernel
    calculate_kernel<<<1, 1>>>(num_records, x, y, zz_device, n, result);

    // Copy result back to host
    float statistic;
    cudaMemcpy(&statistic, result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Statistic: " << statistic << std::endl;

    // Free allocated memory
    cudaFree(corr_coef);
    cudaFree(zz_device);
    cudaFree(result);

    return 0;
}