#include <cuda_runtime.h>
#include "utils.hh"
#include "helper_cuda.h"
#include <stdlib.h>

using namespace std;
#define nthread 256
#define FULL_MASK 0xffffffff
#define warpSize 32

void verify(const int &len, float *x, float *y) {
  const static float eps = 1e-3;
  for (int i = 0; i < len; i++) {
    if (fabs(x[i] - y[i]) > eps) {
      printf("idx: %d, host: %.4f, device: %.4f", i, x[i], y[i]);
      return;
    }
  }
  printf("OK\n");
  return;
}

void matrix_reduction(const int &m, const int &n, float *mat, float *res) {
  for (int i = 0; i < m; ++i) {
    float sum = 0.f;
    for (int j = 0; j < n; ++j) {
      sum += mat[i * n + j];
    }
    res[i] = sum;
  }
}

__global__ void matrix_reduction_warp_cuda(const int m, const int n, 
                                          float *mat, float *res) {
  float val = 0;
  int idx = threadIdx.x;
  int line = blockIdx.x * (nthread / warpSize) + idx / warpSize;
  int off = idx % warpSize;
  // int stride = blockDim.x * (nthread / warpSize);
  for (int i = line * n + off; i < (line + 1) * n; i += warpSize) {
    val += mat[i];
  }
  for (int interval = 16; interval > 0; interval >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, interval);
  }
  if (off == 0) {
    res[line] = val;
  }
}

int main(int argc, char **argv) {
  int m{}, n{};
  if (argc != 3) {
    m = 131072;
    n = 1024;
    std::cerr << "lack of arguments, use default size" << std::endl;
  } else {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
  }
  std::cerr << "matrix size: " << m << " x " << n << std::endl;
  const int mat_len = m * n;

  auto mat = new float[mat_len];
  auto res_vec_host = new float[m];
  auto res_vec_device = new float[m];

  checkCudaErrors(cudaHostRegister(mat, sizeof(float) * mat_len, cudaHostRegisterDefault));

  RandomizeSP(mat_len, mat);

  float *mat_d, *res_vec_d;
  checkCudaErrors(cudaMalloc(&mat_d, sizeof(float) * m * n));
  checkCudaErrors(cudaMalloc(&res_vec_d, sizeof(float) * m));
  checkCudaErrors(cudaMemcpy(mat_d, mat, sizeof(float) * m * n, cudaMemcpyHostToDevice));

  auto host_launch = [&]() {
    Timer timer("host execution");
    matrix_reduction(m, n, mat, res_vec_host);
  };

  host_launch();

  auto device_launch = [&]() {
    Timer timer("device execution 1 - a line per warp");
    matrix_reduction_warp_cuda<<<m / (nthread / warpSize), nthread>>>
                                (m, n, mat_d, res_vec_d);
    checkCudaErrors(cudaDeviceSynchronize()); 
  };

  device_launch();

  checkCudaErrors(cudaMemcpy(res_vec_device, res_vec_d, sizeof(float) * m, cudaMemcpyDeviceToHost));
  // PrintArray(m, res_vec);
  verify(m, res_vec_host, res_vec_device);

  delete[] mat;
  delete[] res_vec_host;
  delete[] res_vec_device;
  return 0;
}