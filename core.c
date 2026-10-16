#include <stddef.h>

// 定义一个包含 8 个 float 的向量类型（对应 256 位 SIMD，如 AVX2 或 Neon 8-wide）
// 编译器会根据目标平台自动将其映射为最优的硬件向量寄存器
typedef float v8sf __attribute__((vector_size(32)));

/**
 * 1. 向量加法：c = a + b
 */
void vvadd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float); // 8 个单精度浮点数
    
    // SIMD 主循环
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va + vb; // 编译器自动生成向量加法指令
    }
    
    // 尾部标量（处理不能被 8 整除的部分）
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

/**
 * 2. 向量减法：c = a - b
 */
void vvsub(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);
    
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va - vb;
    }
    
    for (; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

/**
 * 3. 向量对位乘法：c = a * b（Hadamard 乘积 / 标量展开后的矩阵乘法基石）
 */
void vvmul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);
    
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va * vb;
    }
    
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

/**
 * 4. 高性能向量点积：sum = a · b（四路并行累加，打破 FMA 延迟链）
 */
float vvdot(const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);       // 8
    size_t unroll_step = step * 4;                    // 32 (4个向量寄存器并行)
    
    // 初始化 4 个独立的向量累加器，彻底打破流水线延迟依赖
    v8sf sum0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    v8sf sum1 = sum0;
    v8sf sum2 = sum0;
    v8sf sum3 = sum0;
    
    // 四路展开循环
    for (; i + unroll_step <= n; i += unroll_step) {
        sum0 = *(const v8sf*)(a + i) * *(const v8sf*)(b + i) + sum0;
        sum1 = *(const v8sf*)(a + i + step) * *(const v8sf*)(b + i + step) + sum1;
        sum2 = *(const v8sf*)(a + i + step * 2) * *(const v8sf*)(b + i + step * 2) + sum2;
        sum3 = *(const v8sf*)(a + i + step * 3) * *(const v8sf*)(b + i + step * 3) + sum3;
    }
    
    // 合并 4 个累加向量
    v8sf total_vec = (sum0 + sum1) + (sum2 + sum3);
    
    // 处理剩余的单向量块 (步长为 8)
    for (; i + step <= n; i += step) {
        total_vec = *(const v8sf*)(a + i) * *(const v8sf*)(b + i) + total_vec;
    }
    
    // 水平规约：将向量内部的 8 个分量相加
    float temp[8];
    *(v8sf*)temp = total_vec;
    float result = temp[0] + temp[1] + temp[2] + temp[3] + 
                   temp[4] + temp[5] + temp[6] + temp[7];
                   
    // 尾部标量
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}


/**
 * 1. 矩阵加法：C = A + B (支持任意 m x n 连续矩阵)
 */
void mmadd(size_t m, size_t n, const float* A, const float* B, float* C) {
    // 连续矩阵加法可以直接降维成一维长向量加法
    vvadd(A, B, C, m * n);
}

/**
 * 2. 矩阵对位乘法 (Hadamard 积)：C = A ⊙ B
 */
void mmmul(size_t m, size_t n, const float* A, const float* B, float* C) {
    vvmul(A, B, C, m * n);
}

/**
 * 3. 矩阵乘法主力引擎：C = A(mxk) * B(nxk) -> (mxn)
 */
void mmdot(size_t m, size_t n, size_t k,
           const float* __restrict__ A, 
           const float* __restrict__ B, 
           float* __restrict__ C) {
    
    // 动态多线程调度（大规模并行，小规模单线程避免开销）[cite: 1, 2]
    #pragma omp parallel for collapse(2) if (m * n * k > 65536)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // A 的第 i 行与 B 的第 j 行进行重定义点积
            C[i * n + j] = vvdot(A + i * k, B + j * k, k);
        }
    }
}
