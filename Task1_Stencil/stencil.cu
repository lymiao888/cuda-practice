#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "helper_cuda.h"
#include "CudaAllocator.h"
// 请你使用CUDA改写上述伪代码，使其能在GPU上运行。具体流程可能如下：
//   - 为几个数组分别在gpu和cpu上分配内存
//   - 在cpu上初始化数组，填入随机数
//   - 将cpu上的数组拷贝到gpu上
//   - 启动计算kernel
//   - 将数据从gpu拷贝回cpu
//   - 在cpu上运行参考实现，对比答案
//   - 释放所有资源
/*
void masked_9_point_stencil(float mat[m][n], bool mask[m][n], float result[m][n]) {
    memset(result, 0.f, sizeof(float) * m * n);
    for (int i = 1; i < m - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
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
*/

void init(std::vector<std::vector<float>, CudaAllocator<std::vector<float>>> &mat, std::vector<std::vector<bool>, CudaAllocator<std::vector<bool>>> &mask, int m, int n){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat[i][j] = rand() % 1000;
            mask[i][j] = rand() % 2 ? true : false;
        }
    }
}

int main(){
    // define 
    std::vector<std::vector<float>, CudaAllocator<std::vector<float>>> mat;
    std::vector<std::vector<bool>, CudaAllocator<std::vector<bool>>> mask;
    std::vector<std::vector<float>, CudaAllocator<std::vector<float>>> result;
    
    // Initialization
    int n = 20, m = 20;
    init(mat, mask, n, m);
}
