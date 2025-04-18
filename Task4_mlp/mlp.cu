#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include "helper_cuda.h"
#include "matrix.h"

#define nthread 128
#define warpsize 32
#define tensor_m 8
#define tensor_n 32
#define tensor_k 16
#define FULL_MASK 0xffffffff

using namespace std;
using namespace nvcuda;

__global__ void cuda_gemm_tensor(int m, int n, int k, half *weight, float *bias, half *input, half *output) {
  //every warp in the block has tensor_m * tensor_n space to store tensor result
  __shared__ float tmp[tensor_m * tensor_n * nthread / warpsize];
  int wrapidx = blockIdx.x * (blockDim.x / warpsize) + threadIdx.x / warpsize;
  int line_a = (wrapidx / (n / tensor_n)) * tensor_m;
  int line_b = (wrapidx % (n / tensor_n)) * tensor_n;
  int rounds = k / tensor_k;
  wmma::fragment<wmma::matrix_a, tensor_m, tensor_n, tensor_k, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, tensor_m, tensor_n, tensor_k, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, tensor_m, tensor_n, tensor_k, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  for (int i = 0; i < rounds; ++i) {
    wmma::load_matrix_sync(a_frag, input + k * line_a + i * tensor_k, k);
    wmma::load_matrix_sync(b_frag, weight + k * line_b + i * tensor_k, k);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(tmp + tensor_m * tensor_n * (threadIdx.x / warpsize), c_frag, 
                          tensor_n, wmma::mem_row_major);
  for (int i = threadIdx.x % warpsize; i < tensor_m * tensor_n; i += warpsize) {
    int row_off = i / tensor_n;
    int col_off = i % tensor_n;
    int offset = tensor_m * tensor_n * (threadIdx.x / warpsize);
    tmp[offset + i] += bias[line_b + col_off];
    tmp[offset + i] = tmp[offset + i] > 0 ? tmp[offset + i] : 0; //relu
    output[(line_a + row_off) * n + line_b + col_off] = __float2half(tmp[offset + i]);
  }
}

__global__ void cuda_gemm(int m, int n, int k, half *weight, float *bias, half *input, half *output) {
  __shared__ half tmp_a_line[8192];
  int warpidx = blockIdx.x * (blockDim.x / warpsize) + threadIdx.x / warpsize;
  int line_a = warpidx / (n / (blockDim.x / warpsize) * (blockDim.x / warpsize));
  int line_b = warpidx % (n / (blockDim.x / warpsize) * (blockDim.x / warpsize));
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    tmp_a_line[i] = input[line_a * k + i];
  }
  __syncthreads();
  half sum = __float2half(.0f);
  for (int i = threadIdx.x % warpsize; i < k; i += warpsize) {
    sum += tmp_a_line[i] * weight[line_b * k + i];
  }
  for (int interval = 16; interval > 0; interval >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, interval);
  }
  if (threadIdx.x % warpsize == 0) {
    sum += bias[line_b];
    sum = sum > __float2half(.0f) ? sum : __float2half(.0f);
    output[line_a * n + line_b] = sum;  
  }
}

//mixed-precision
__global__ void cuda_gemm_mixed(int m, int n, int k, half *weight, float *bias, half *input, half *output) {
  __shared__ half tmp_a_line[8192];
  int warpidx = blockIdx.x * (blockDim.x / warpsize) + threadIdx.x / warpsize;
  int line_a = warpidx / (n / (blockDim.x / warpsize) * (blockDim.x / warpsize));
  int line_b = warpidx % (n / (blockDim.x / warpsize) * (blockDim.x / warpsize));
  //assume all threads in a block have the same line_a
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    tmp_a_line[i] = input[line_a * k + i];
  }
  __syncthreads();
  float sum = .0f;
  for (int i = threadIdx.x % warpsize; i < k; i += warpsize) {
    sum += __half2float(tmp_a_line[i] * weight[line_b * k + i]);
  }
  for (int interval = 16; interval > 0; interval >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, interval);
  }
  if (threadIdx.x % warpsize == 0) {
    sum += bias[line_b];
    sum = sum > .0f ? sum : .0f;
    output[line_a * n + line_b] = __float2half(sum);
  }
}

__global__ void cuda_gemm_left(int m, int n, int k, half *weight, float *bias, half *input, half *output) {
  __shared__ half tmp_a_line[8192];
  int warpidx = blockIdx.x * (blockDim.x / warpsize) + threadIdx.x / warpsize;
  int line_a = warpidx / (n % (nthread / warpsize));
  int line_b = warpidx % (n % (nthread / warpsize));
  int weight_off = n / (nthread / warpsize) * (nthread / warpsize);
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    tmp_a_line[i] = input[line_a * k + i];
  }
  __syncthreads();
  half sum = __float2half(.0f);
  for (int i = threadIdx.x % warpsize; i < k; i += warpsize) {
    sum += tmp_a_line[i] * weight[(weight_off + line_b) * k + i];
  }
  for (int interval = 16; interval > 0; interval >>= 1) {
    sum += __shfl_down_sync(FULL_MASK, sum, interval);
  }
  if (threadIdx.x % warpsize == 0) {
    sum += bias[weight_off + line_b];
    sum = sum > __float2half(.0f) ? sum : __float2half(.0f);
    output[line_a * n + weight_off + line_b] = sum;
  }
}

void run_layer(Matrix<half> weight, half *weight_d, Matrix<float> bias, float *bias_d,
              half *input, half *output, int pictures) {
  checkCudaErrors(cudaMemcpy(weight_d, &weight.data[0], weight.data.size() * sizeof(half),
                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bias_d, &bias.data[0], bias.data.size() * sizeof(float), 
                  cudaMemcpyHostToDevice));
  int blocknum = (pictures / tensor_m) * (weight.shape[0] / tensor_n) / (nthread / warpsize);
  if (blocknum >= 64) {
    cuda_gemm_tensor<<<blocknum, nthread>>>(pictures, weight.shape[0], weight.shape[1], weight_d,
                                            bias_d, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  else { 
    blocknum = pictures * (weight.shape[0] / (nthread / warpsize));
    cuda_gemm<<<blocknum, nthread>>>(pictures, weight.shape[0], weight.shape[1], weight_d,
                                    bias_d, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  if (weight.shape[0] % (nthread / warpsize)) {
    int threads = (weight.shape[0] % (nthread / warpsize)) * warpsize;
    blocknum = pictures;
    cuda_gemm_left<<<blocknum, threads>>>(pictures, weight.shape[0], weight.shape[1], weight_d, bias_d, 
                                          input, output);
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

int main() {
  auto fc1_bias = Matrix<float>("model/fc1.bias.txt");
  auto fc1_weight = Matrix<half>("model/fc1.weight.txt");
  auto fc2_bias = Matrix<float>("model/fc2.bias.txt");
  auto fc2_weight = Matrix<half>("model/fc2.weight.txt");
  auto fc3_bias = Matrix<float>("model/fc3.bias.txt");
  auto fc3_weight = Matrix<half>("model/fc3.weight.txt");
  auto input = Matrix<half>("dataset/bs-8192-image.txt");
  auto lable = Matrix<int>("dataset/bs-8192-label.txt");
  half *result = (half *)malloc(fc3_weight.shape[0] * input.shape[0] * sizeof(half));
  cerr << "[*]Matrix shape:" << endl;
  input.log_info();
  fc1_weight.log_info();
  fc2_weight.log_info();
  fc3_weight.log_info();

  half *fc1_weight_d, *fc2_weight_d, *fc3_weight_d;
  float *fc1_bias_d, *fc2_bias_d, *fc3_bias_d;
  half *input_d, *tmp_d;
  float *output_d;
  checkCudaErrors(cudaMalloc(&fc1_weight_d, fc1_weight.data.size() * sizeof(half)));
  checkCudaErrors(cudaMalloc(&fc2_weight_d, fc2_weight.data.size() * sizeof(half)));
  checkCudaErrors(cudaMalloc(&fc3_weight_d, fc3_weight.data.size() * sizeof(half)));
  checkCudaErrors(cudaMalloc(&fc1_bias_d, fc1_bias.data.size() * sizeof(float)));
  checkCudaErrors(cudaMalloc(&fc2_bias_d, fc2_bias.data.size() * sizeof(float)));
  checkCudaErrors(cudaMalloc(&fc3_bias_d, fc3_bias.data.size() * sizeof(float)));

  checkCudaErrors(cudaMalloc(&input_d, input.data.size() * sizeof(half)));
  checkCudaErrors(cudaMalloc(&tmp_d, input.shape[0] * fc1_weight.shape[0] * sizeof(float)));
  checkCudaErrors(cudaMalloc(&output_d, input.shape[0] * fc3_weight.shape[0] * sizeof(half)));

  checkCudaErrors(cudaMemcpy(input_d, &input.data[0], input.data.size() * sizeof(half), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaFuncSetCacheConfig(cuda_gemm_tensor, cudaFuncCachePreferShared));
  run_layer(fc1_weight, fc1_weight_d, fc1_bias, fc1_bias_d, input_d, tmp_d, input.shape[0]);
  run_layer(fc2_weight, fc2_weight_d, fc2_bias, fc2_bias_d, tmp_d, input_d, input.shape[0]);
  run_layer(fc3_weight, fc3_weight_d, fc3_bias, fc3_bias_d, input_d, tmp_d, input.shape[0]);

  checkCudaErrors(cudaMemcpy(result, tmp_d, 10 * input.shape[0] * sizeof(half), cudaMemcpyDeviceToHost));
  
  int diff = 0;
  for (int i = 0; i < input.shape[0]; ++i) {
    int idx = 0;
    float val = __half2float(result[i * fc3_weight.shape[0]]);
    for (int j = 0; j < fc3_weight.shape[0]; ++j) {
      if(val < __half2float(result[i * fc3_weight.shape[0] + j])) {
        val = __half2float(result[i * fc3_weight.shape[0] + j]);
        idx = j;
      }
    }
    if (idx != lable.data[i]) {
      ++diff;
    }
  }
  printf("%d out of %d labels are different!\n", diff, input.shape[0]);

  free(result);
  cudaFree(fc1_bias_d);
  cudaFree(fc1_weight_d);
  cudaFree(fc2_bias_d);
  cudaFree(fc2_weight_d);
  cudaFree(fc3_bias_d);
  cudaFree(fc3_weight_d);
  cudaFree(input_d);
  cudaFree(tmp_d);
  cudaFree(output_d);
}