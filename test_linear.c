#include <stdio.h>
#include <stdlib.h>
#include "linear.h"

int main()
{
    size_t m = 3, n = 2, k = 3;

    // 分配测试矩阵
    float *A = (float *)malloc(m * k * sizeof(float));
    float *B = (float *)malloc(n * k * sizeof(float));
    float *C = (float *)malloc(m * n * sizeof(float));
    float *BT = (float *)malloc(n * k * sizeof(float));

    // 初始化数据
    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 3.0f;
    A[3] = 4.0f;
    A[4] = 5.0f;
    A[5] = 6.0f;
    A[6] = 7.0f;
    A[7] = 8.0f;
    A[8] = 9.0f;


    B[0] = 1.0f;
    B[1] = 2.0f;
    B[2] = 3.0f;
    B[3] = 4.0f;
    B[4] = 5.0f;
    B[5] = 6.0f;

    // 执行重定义的矩阵点积：mxk @ nxk -> mxn
    mmdot(m, n, k, A, B, C);

    // 打印结果验证
    printf("Matrix C result:\n");
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    mtrans(n, k, B, BT);
    mmtdot(m, k, n, A, BT, C);
    printf("Matrix C result:\n");
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            printf("%f ", C[i * n + j]);
        }
        printf("\n");
    }

    float *sina = (float *)malloc(m * k * sizeof(float));
    vmf(sinf, A, sina, m*k);

    printf("Matrix sina result:\n");
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            printf("%f ", sina[i * k + j]);
        }
        printf("\n");
    }
    // 释放资源
    free(A);
    free(B);
    free(C);
    free(BT);

    printf("Verification finished successfully!\n");
    return 0;
}