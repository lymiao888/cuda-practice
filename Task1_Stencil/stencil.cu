#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include <ratio>
#include <vector>
#include <random>
#include <chrono>
#include "helper_cuda.h"

#define M 131072
#define N 4096
void init_float(float (*arr)[N], int m) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < N; j++){
            arr[i][j] = (std::rand() % 100000) / 1000;
        }
    }
}
void init_bool(bool (*arr)[N], int m) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < N; j++){
            if (std::rand()%2 == 0) arr[i][j] = true;
            else arr[i][j] = false;
        }
    }
}
__global__ void masked_9_point_stencil_from_gpu(float (*mat)[N], bool (*mask)[N], float (*result)[N], int m) {
    memset(result, 0.f, sizeof(float) * m * N);
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float res = 0.f;
            res += mask[i][j] ? mat[i][j] : 0.f;
            res += mask[i - 1][j] ? mat[i - 1][j] : 0.f;
            res += mask[i][j - 1] ? mat[i][j - 1] : 0.f;
            res += mask[i + 1][j] ? mat[i + 1][j] : 0.f;
            res += mask[i][j + 1] ? mat[i][j + 1] : 0.f;
            res += mask[i + 1][j + 1] ? mat[i + 1][j + 1] : 0.f;
            res += mask[i + 1][j - 1] ? mat[i + 1][j - 1] : 0.f;
            res += mask[i - 1][j + 1] ? mat[i - 1][j + 1] : 0.f;
            res += mask[i - 1][j - 1] ? mat[i - 1][j - 1] : 0.f;
            result[i][j] = res;
        }
    }
}
__host__ void masked_9_point_stencil_from_cpu(float (*mat)[N], bool (*mask)[N], float (*result)[N], int m) {
    memset(result, 0.f, sizeof(float) * m * N);
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float res = 0.f;
            res += mask[i][j] ? mat[i][j] : 0.f;
            res += mask[i - 1][j] ? mat[i - 1][j] : 0.f;
            res += mask[i][j - 1] ? mat[i][j - 1] : 0.f;
            res += mask[i + 1][j] ? mat[i + 1][j] : 0.f;
            res += mask[i][j + 1] ? mat[i][j + 1] : 0.f;
            res += mask[i + 1][j + 1] ? mat[i + 1][j + 1] : 0.f;
            res += mask[i + 1][j - 1] ? mat[i + 1][j - 1] : 0.f;
            res += mask[i - 1][j + 1] ? mat[i - 1][j + 1] : 0.f;
            res += mask[i - 1][j - 1] ? mat[i - 1][j - 1] : 0.f;
            result[i][j] = res;
        }
    }
}

bool verify(float (*mat1)[N], float (*mat2)[N], int m){
    bool ret = true;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < N; j++){
            if (mat1[i][j] != mat2[i][j])   ret = false;
        }
    }
    return ret;
}

int main(){

    // define array
    float (*mat)[N] = new float[M][N];
    bool (*mask)[N] = new bool[M][N];
    float (*res)[N] = new float[M][N];
    float (*res_from_gpu)[N] = new float[M][N];

    float (*c_mat)[N];
    bool (*c_mask)[N];
    float (*c_res)[N];

    // initialize mat and mask
    init_float(mat, M);
    init_bool(mask, M);

    // we have to allocate memory before memcpy ,don't forget c_res
    cudaMalloc(&c_mat, M * N * sizeof(float));
    cudaMalloc(&c_mask, M * N * sizeof(bool));
    cudaMalloc(&c_res, M * N * sizeof(float));
    
    checkCudaErrors(cudaMemcpy(c_mat, mat, M * N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_mask, mask, M * N * sizeof(bool), cudaMemcpyHostToDevice));

    // Compute mat on gpu
    auto t1 = std::chrono::steady_clock::now();
    masked_9_point_stencil_from_gpu<<<1, 8>>>(c_mat, c_mask, c_res, M);
    checkCudaErrors(cudaDeviceSynchronize());
    auto t2 = std::chrono::steady_clock::now();
    double d1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
    std::cout<<"time on gpu : "<<d1<<"ms\n";
    
    // Compute mat on cpu
    auto t3 = std::chrono::steady_clock::now();
    masked_9_point_stencil_from_cpu(mat, mask, res, M);
    auto t4 = std::chrono::steady_clock::now();
    double d2 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t4 - t3).count();
    std::cout<<"time on cpu : "<<d2<<"ms\n";

    // Verify the answer
    checkCudaErrors(cudaMemcpy(res_from_gpu, c_res, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    bool equel = verify(res_from_gpu, res, M);
    if (equel == true)
        std::cout<<"the result is correct!\n";
    else
        std::cout<<"the result is incorrect!\n";

    //free memory
    cudaFree(c_mat);
    cudaFree(c_mask);
    cudaFree(c_res);
    
    delete[] mat;
    delete[] mask;
    delete[] res;
    delete[] res_from_gpu;

    return 0;
}
