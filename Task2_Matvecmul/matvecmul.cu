#include <climits>
#include <cstdlib>
#include <driver_types.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "utils.hh"

void cpu_mat_times_vec_row(const int m, const int k, const float *A, const float *x, float *B) {
    for (int i = 0; i < m; ++i) {
        float res = 0;
        for (int j = 0; j < k; ++j) {
            res += A[i * k + j] * x[j];
        }
        B[i] = res;
    }
}

void cpu_mat_times_vec_col(const int m, const int k, const float *A, const float *x, float *B) {
    for (int i = 0; i < m; ++i) {
        float res = 0;
        for (int j = 0; j < k; ++j) {
            res += A[j * m + i] * x[j];
        }
        B[i] = res;
    }
}

__global__ void gpu_mat_times_vec_row(const int m, const int k, const float *A, const float *x, float *B) {
    __shared__ float vec[4096];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        vec[i] = x[i];
    }
    __syncthreads();
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row <= m) {
        float res = 0;
        for (int i = 0; i < k; ++i) {
            res += A[row * k + i] * vec[i];
        }
        B[row] = res;
    }
}

__global__ void gpu_mat_times_vec_col(const int m, const int k, const float *A, const float *x, float *B) {
    __shared__ float vec[4096];
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        vec[i] = x[i];
    }
    __syncthreads();
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row <= m) {
        float res = 0;
        for (int i = 0; i < k; ++i) {
            res += A[i * m + row] * vec[i];
        }
        B[row] = res;
    }
}

template<class T>
bool verify(const T *arr1, const T *arr2, const int len){
    const static float eps = 1e-3;
    for (int i = 0; i < len; i++) {
        if (std::fabs(arr1[i] - arr2[i]) > eps){
            return false;
        }
    }
    return true;
}

int main(int argc,char *argv[]) {
    int m, k;
    if (argc == 2) {
        m = atoi(argv[0]);
        k = atoi(argv[1]);
    }
    else {
        m = 131072;
        k = 4096;
    }
    
    float *A = new float[m * k];
    float *x = new float[k];
    float *B = new float[m];
    float *c_A, *c_x, *c_B;
    float *B_from_gpu = new float[m];
    
    RandomizeSP(m * k, A);
    RandomizeSP(k, x);
    
    cudaMalloc(&c_A, m * k * sizeof(float));
    cudaMalloc(&c_x, k * sizeof(float));
    cudaMalloc(&c_B, m * sizeof(float));
    cudaMemset(c_B, 0, m * sizeof(float));
    checkCudaErrors(cudaMemcpy(c_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_x, x, k * sizeof(float), cudaMemcpyHostToDevice));
    
    {
        Timer time_cpu("cpu_mat_times_vec");
        cpu_mat_times_vec_row(m, k, A, x, B);
    }
    {
        int nthreads = 128;
        int nblocks = (m + nthreads + 1) / nthreads;
        Timer time_gpu1("gpu_mat_times_vec_row");
        gpu_mat_times_vec_row<<<nblocks, nthreads>>>(m, k, c_A, c_x, c_B);
    }
    checkCudaErrors(cudaMemcpy(B_from_gpu, c_B, m * sizeof(float), cudaMemcpyDeviceToHost));
    if (verify(B, B_from_gpu, m) == true)
        std::cout<<"correct!"<<std::endl;
    else
        std::cout<<"incorrect!"<<std::endl;


    {
        Timer time_cpu("cpu_mat_times_vec");
        cpu_mat_times_vec_col(m, k, A, x, B);
    }
    {
        int nthreads = 128;
        int nblocks = (m + nthreads + 1) / nthreads;
        Timer time_gpu1("gpu_mat_times_vec_col");
        gpu_mat_times_vec_col<<<nblocks, nthreads>>>(m, k, c_A, c_x, c_B);
    }

    checkCudaErrors(cudaMemcpy(B_from_gpu, c_B, m * sizeof(float), cudaMemcpyDeviceToHost));
    if (verify(B, B_from_gpu, m) == true)
        std::cout<<"correct!"<<std::endl;
    else
        std::cout<<"incorrect!"<<std::endl;

    cudaFree(c_A);
    cudaFree(c_x);
    cudaFree(c_B);
    
    delete[] A;
    delete[] x;
    delete[] B;
    delete[] B_from_gpu;
    return 0;
}