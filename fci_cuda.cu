#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <cublas_v2.h>
// input: pds_list
// a graph representation
// a sepset class attributes
// a ci_test class attributes

// output: the graph with the edges that are not significant removed

// Description: This function implements the FCI algorithm with the CUDA parallel programming model.
#define numnodes 15
#define numsamples 1000
#define maxlevel 7
#define blocksize 8
#define alpha 0.01

__constant__ float correlation_matrix[] = {1.00000000e+00, -4.83579373e-01, -2.10180536e-02, 8.94362611e-01,
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

__device__ int calculate_statistic(int x, int y, int *zz, int n)
{
  float par_corr;

  // Extract correlation coefficients for readability
  float corr_xy = correlation_matrix[x * numnodes + y];
  float corr_xz = correlation_matrix[x * numnodes + zz[0]];
  float corr_yz = correlation_matrix[y * numnodes + zz[0]];

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
      if (zz[i] < 0 || zz[i] >= numnodes)
      {
        printf("illegal index %d\n", zz[i]);
      }
      else
      {
        all_var_idx[i + 2] = zz[i];
      }
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
        printf("all_var_idx[i] = %d, all_var_idx[j] = %d, coeff = %f \n", all_var_idx[i], all_var_idx[j], correlation_matrix[all_var_idx[i] * numnodes + all_var_idx[j]]);
        // if (all_var_idx[i] < 0 || all_var_idx[j] < 0 || all_var_idx[i] >= numnodes || all_var_idx[j] >= numnodes)
        // {
        //   // corr_coef_subset[i * (n + 2) + j] = 0.0;
        //   printf("illegal index %d %d\n", all_var_idx[i], all_var_idx[j]);
        //   continue; // skip illegal index
        // }
        float k = correlation_matrix[all_var_idx[i] * numnodes + all_var_idx[j]];

        // corr_coef_subset[i * (n + 2) + j] = correlation_matrix[all_var_idx[i] * numnodes + all_var_idx[j]];
        corr_coef_subset[i * (n + 2) + j] = k;

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
    // printf("par_corr: %f\n", par_corr);

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

  int deg_of_freedom = numsamples - n - 2;
  float z = 0.5 * log1pf((2 * par_corr) / (1 - par_corr));
  float val_for_cdf = abs(sqrtf(deg_of_freedom - 1) * z);
  float statistics = 2 * (1 - norm_cdf(val_for_cdf));
  if (statistics < alpha)
    return 0;
  else
    return 1;
}

// __global__ void calculate_kernel(int x, int y, int *zz, int n, float *result)
// {
//   *result = calculate_statistic(x, y, zz, n);
// }

inline cudaError_t checkCudaErr(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime error at %s: %s\n", msg, cudaGetErrorString(err));
  }
  return err;
}

// generating combinations
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
  if (skip > 0)
  {
    for (int i = 0; i < P; i++)
    {
      if (out[i] >= skip)
        out[i] = out[i] + 1;
    }
  }
}

__device__ int factorial(int n)
{
  int result = 1;
  for (int i = 1; i <= n; ++i)
  {
    result *= i;
  }
  return result;
}

__device__ void delete_edge(int *adj_matrix, int node1, int node2)
{
  adj_matrix[node1 * numnodes + node2] = 0;
  adj_matrix[node2 * numnodes + node1] = 0;
}

__constant__ int pds_list[] = {3, 12, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 6, 8, 9, 10, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 8, 9, 10, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 8, 9, 10, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 6, 9, 10, 11, 13, 14, -1, -1, -1, -1, -1, -1, 1, 3, 6, 8, 10, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 6, 8, 9, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 6, 8, 9, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 6, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1};
// __constant__ int *pds_set_sizes;
// [numnodes] = {{1, 3, 5, 6, 8, 9, 11, 12, 14}, {0, 3, 5, 8, 11, 14}, {}, {0, 1, 5, 8, 11, 14}, {}, {0, 1, 3, 8, 11, 14}, {0,9,8}, {8, 12 , 13}, {0, 1, 3, 5, 6, 7, 9, 11, 13, 14}, {0,8,6}, {}, {1,3,5,8,14}, {0,7}, {8,7}, {0,1,3,5,8,11}};
// __constant__ int *adjnode;
__constant__ int graphnodes[] = {0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, -1};

__device__ int sepset[numnodes * numnodes * numnodes];

//__global__ void fci_zero_kernel(graph *g)
__global__ void fci_zero_kernel(int *adj_matrix)
{
  int index_i = blockIdx.x * blockDim.x + threadIdx.x;
  int i = graphnodes[index_i];
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < numnodes && j < numnodes && i < j)
  {
    if (correlation_matrix[i * numnodes + j] >= 1.0)
    {
      printf("setting %d, %d to 0 \n", i, j);
      adj_matrix[i * numnodes + j] = 0;
      adj_matrix[j * numnodes + i] = 0;
    }
  }
}

__global__ void fci_level_kernel(int *adj_matrix, int level)
{
  int index_i = blockIdx.y;
  int i = graphnodes[index_i];

  int indices[numnodes - 1];
  int condset[numnodes - 1];
  int node1, node2;
  if (i != -1)
  {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("i = %d, j = %d \n", i, j);
    int index_j = -1;
    int numedges = 0;
    __shared__ int pds_set[numnodes];
    // __shared__ int sepset[numnodes * numnodes]
    // int temp = 0;

    for (int k = 0; k < numnodes - 1; k++)
    {
      pds_set[k] = pds_list[i * (numnodes) + k]; // numnodes -1
      // printf("pds_set[%d] = %d \n", k, pds_set[k]);
      if (pds_set[k] == j)
      {
        index_j = k;
      }
    }
    __syncthreads();

    for (int k = 0; k < numnodes - 1; k++)
    {
      if (pds_set[k] != -1)
      {
        numedges++;
      }
    }

    if (index_j == -1)
    {
      // printf("nothing to remove from ods set \n at i = %d, j = %d \n", i, j);
    }
    else
    {
      printf("index_j = %d \n", index_j);
      numedges--;
      index_j++;
    }

    // printf("numedges in pdsset %d = %d \n", i, numedges);

    // int j = adjnode[i*numnodes + p];

    if (i > j)
    {
      node1 = i;
      node2 = j;
    }
    else
    {
      node1 = j;
      node2 = i;
    }

    int combinations = factorial(numedges) / (factorial(numedges - level) * factorial(level));
    // printf("yahan tk sahi chl rha \n");
    if (level <= numedges)
    {
      for (int t = threadIdx.y; t < combinations; t += blockDim.y)
      {
        if (adj_matrix[i * numnodes + j] == 1)
        {
          // printf("combination parameters %d, %d, %d, %d \n", numedges - 1, level, t, index_j + 1);
          IthCombination(indices, numedges, level, t + 1, index_j);
          for (int k = 0; k < level; k++)
          {
            condset[k] = pds_set[indices[k] - 1];
          }
          // for (int k = 0; k < numnodes - 1; k++)
          // {
          //   printf("%d ", condset[k]);
          // }
          // printf("level: %d \n", level);

          if (calculate_statistic(i, j, condset, level) == 1)
          { // cond indep condition here
            // g->delete_edge(i, j);
            printf("deleting edge %d, %d \n", i, j);
            adj_matrix[i * numnodes + j] = 0;
            adj_matrix[j * numnodes + i] = 0;

            for (int z = 0; z < level; z++)
            {
              sepset[(numnodes * numnodes * z) + (numnodes * node2) + node1] = condset[z];
            }
          }
        }
      }
    }
  }
}

__global__ void fci_parent_kernel(int *g)
{
  int level = blockIdx.x * blockDim.x + threadIdx.x;
  if (level <= maxlevel)
  {
    if (level == 0)
    {
      fci_zero_kernel<<<dim3(numnodes, numnodes), dim3(1, 1)>>>(g);
    }
    else
    {
      fci_level_kernel<<<dim3(numnodes, numnodes), dim3(1, 1)>>>(g, level);
    }
  }
}

int main()
{
  // graph g();
  int *adj_matrix;
  checkCudaErr(cudaMallocManaged(&adj_matrix, numnodes * numnodes * sizeof(int)), "alloc adj mat");
  int adj_matrix_h[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0};
  for (int i = 0; i < numnodes; i++)
  {
    for (int j = 0; j < numnodes; j++)
    {
      if (adj_matrix_h[i * numnodes + j] == 1)
        adj_matrix[i * numnodes + j] = 1;
      else
        adj_matrix[i * numnodes + j] = 0;
    }
  }

  for (int i = 0; i < numnodes; i++)
  {
    for (int j = 0; j < numnodes; j++)
    {
      printf("%d ", adj_matrix[i * numnodes + j]);
    }
    printf("\n ");
  }

  fci_parent_kernel<<<numnodes - 1, 1>>>(adj_matrix);
  cudaDeviceSynchronize();
  checkCudaErr(cudaGetLastError(), "error last check");
  // checkCudaErr(cudaMemcpy(adj_mat, g.adj_matrix, numnodes * numnodes * sizeof(int), cudaMemcpyDeviceToHost), "d to h");
  printf("printing adj matrix after algorithm \n");
  for (int i = 0; i < numnodes; i++)
  {
    for (int j = 0; j < numnodes; j++)
    {
      printf("%d ", adj_matrix[i * numnodes + j]);
    }
    printf("\n");
  }

  checkCudaErr(cudaFree(adj_matrix), "free error");
  int sepset_h[numnodes * numnodes * numnodes];
  checkCudaErr(cudaMemcpyFromSymbol(sepset_h, sepset, numnodes * numnodes * numnodes * sizeof(int)), "sepset copy");
  // for (int i = 0; i < numnodes; i++)
  // {
  //   for (int j = 0; j < numnodes; j++)
  //   {
  //     printf("sepset between %d and %d \n", i, j);
  //     for (int k = 0; k < numnodes; k++)
  //     {
  //       printf("%d ", sepset_h[(numnodes * numnodes * k) + (numnodes * i) + j]);
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }
  // print graph

  return 0;

  // int pds_list_host[numnodes * numnodes];
  // int adjnodes_host[numnodes * numnodes];

  // for (int i = 0; i < numnodes; i++)
  // {
  //   for (int j = 0; j < numnodes; j++)
  //   {
  //     pds_list_host[i * numnodes + j] = -1;
  //     adjnodes_host[i * numnodes + j] = -1;
  //   }
  // }
}