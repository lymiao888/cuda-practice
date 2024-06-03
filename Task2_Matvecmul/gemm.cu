#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define M 131072	
#define K 4096

// void mat_times_vec(const float A[m][k], const float x[k], float B[m]) {
//     for (int i = 0; i < m; ++i) {
//         float res = 0;
//         for (int j = 0; j < k; ++j) {
//             res += A[i][j] * x[j];
//         }
//         B[i] = res;
//     }
// }

int main() {

    return 0;
}