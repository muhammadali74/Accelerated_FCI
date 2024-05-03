import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

coreelation_mat_string = """__constant__ float correlation_matrix[] = """
pds_list_string = """__constant__ int pds_list[] = """
graphnodes_string = """__constant__ int graphnodes[] = """
cuda_string = """
__device__ int sepset[numnodes * numnodes * numnodes];


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
      // *result = 0;
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


    // Allocate memory for correlation coefficients subset
    float *corr_coef_subset;
    cudaMalloc((void **)&corr_coef_subset, (n + 2) * (n + 2) * sizeof(float));

    // Copy relevant subset of correlation coefficients to device
    for (int i = 0; i < n + 2; ++i)
    {
      for (int j = 0; j < n + 2; ++j)
      {
        
        int q = all_var_idx[i];
        int r = all_var_idx[j];
        int index = q * numnodes + r;

        float u = correlation_matrix[index];

     
        corr_coef_subset[i * (n + 2) + j] = u; // if k is assigned, then i get ilegal memory access errors.
       

        
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
    

    int index_j = -1;
    int numedges = 0;
    __shared__ int pds_set[numnodes];
    // __shared__ int sepset[numnodes * numnodes]
    // int temp = 0;

    for (int k = 0; k < numnodes - 1; k++)
    {
      pds_set[k] = pds_list[i * (numnodes) + k]; // numnodes -1

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
    }
    else
    {

      numedges--;
      index_j++;
    }





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

    if (level <= numedges)
    {
      for (int t = threadIdx.y; t < combinations; t += blockDim.y)
      {
        if (adj_matrix[i * numnodes + j] == 1)
        {
          IthCombination(indices, numedges, level, t + 1, index_j);
          for (int k = 0; k < level; k++)
          {
            condset[k] = pds_set[indices[k] - 1];
          }


          if (calculate_statistic(i, j, condset, level) == 1)
          { 

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
  // int level = blockIdx.x * blockDim.x + threadIdx.x;
  // if (level <= 3)
  for (int level = 0; level < maxlevelsss; level++)
  {
    if (level == 0)
    {
      fci_zero_kernel<<<dim3(numnodes, numnodes), dim3(1, 1)>>>(g);
    }
    else
    {
      fci_level_kernel<<<dim3(numnodes, numnodes/2), dim3(10, 2)>>>(g, level);
    }
  }
}

"""

def main(adj_matrix, correlation_matrix, pds_list, graphnodes, numnodes, numsamples, maxdepth, alpha):
    global cuda_string
    global coreelation_mat_string
    global pds_list_string
    global graphnodes_string

    blocksize = 8
    adj_matrix_gpu = cuda.mem_alloc(adj_matrix.nbytes)
    cuda.memcpy_htod(adj_matrix_gpu, adj_matrix)

    #copy in constant memory
    # correlation_matrix_gpu = cuda.mem_alloc(correlation_matrix.nbytes)
    # cuda.memcpy_htod(correlation_matrix_gpu, correlation_matrix)

    # pds_list_gpu = cuda.mem_alloc(pds_list.nbytes)
    # cuda.memcpy_htod(pds_list_gpu, pds_list)

    # graphnodes_gpu = cuda.mem_alloc(graphnodes.nbytes)
    # cuda.memcpy_htod(graphnodes_gpu, graphnodes)

    cuda_string = cuda_string.replace("numnodes", str(numnodes))
    cuda_string = cuda_string.replace("numsamples", str(numsamples))
    cuda_string = cuda_string.replace("blocksize", str(blocksize))
    cuda_string = cuda_string.replace("INFINITY", str(9999999))
    cuda_string = cuda_string.replace("alpha", str(alpha)) 
    cuda_string = cuda_string.replace("maxlevelsss", str(maxdepth))

    coreelation_mat_string = coreelation_mat_string +  "{" + ",".join([str(i) for i in correlation_matrix]) + "}; \n"
    # print(coreelation_mat_string)
    pds_list_string = pds_list_string + "{" + ",".join([str(i) for i in pds_list]) + "}; \n"
    # print(pds_list_string)
    graphnodes_string = graphnodes_string +  "{" + ",".join([str(i) for i in graphnodes]) + "}; \n"
    # print(graphnodes_string)
 
    cuda_string = coreelation_mat_string + pds_list_string + graphnodes_string + cuda_string

    print("launching kernel")
    mod = DynamicSourceModule(cuda_string)
    fci_kernel = mod.get_function("fci_parent_kernel")
    fci_kernel(adj_matrix_gpu, block=(1,1,1))

    cuda.memcpy_dtoh(adj_matrix, adj_matrix_gpu)
    # print(adj_matrix.reshape(numnodes, numnodes))
    return adj_matrix