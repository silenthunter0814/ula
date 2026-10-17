#include <stddef.h>
#include <stdlib.h>

#define STACK_BUF_SIZE 4096

typedef float v8sf __attribute__((vector_size(32)));

/* ----- Vector OP ----- */
/* c = a + b */
void vvadd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float); // 8 
    
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va + vb; 
    }
    
    for (; i < n; ++i) 
        c[i] = a[i] + b[i];
}

/* c = a - b */
void vvsub(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);
    
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va - vb;
    }
    
    for (; i < n; ++i) 
        c[i] = a[i] - b[i];
}

/* c = a * b */
void vvmul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);
    
    for (; i + step <= n; i += step) {
        v8sf va = *(const v8sf*)(a + i);
        v8sf vb = *(const v8sf*)(b + i);
        *(v8sf*)(c + i) = va * vb;
    }
    
    for (; i < n; ++i) 
        c[i] = a[i] * b[i];
}

/* sum = a @ b */
float vvdot(const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float);       
    size_t unroll_step = step * 4;              
    
    v8sf sum0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    v8sf sum1 = sum0;
    v8sf sum2 = sum0;
    v8sf sum3 = sum0;
    
    for (; i + unroll_step <= n; i += unroll_step) {
        sum0 = *(const v8sf*)(a + i) * *(const v8sf*)(b + i) + sum0;
        sum1 = *(const v8sf*)(a + i + step) * *(const v8sf*)(b + i + step) + sum1;
        sum2 = *(const v8sf*)(a + i + step * 2) * *(const v8sf*)(b + i + step * 2) + sum2;
        sum3 = *(const v8sf*)(a + i + step * 3) * *(const v8sf*)(b + i + step * 3) + sum3;
    }
    
    v8sf total_vec = (sum0 + sum1) + (sum2 + sum3);
    
    for (; i + step <= n; i += step) 
        total_vec = *(const v8sf*)(a + i) * *(const v8sf*)(b + i) + total_vec;
    
    float temp[8];
    *(v8sf*)temp = total_vec;
    float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
                   
    for (; i < n; ++i) 
        result += a[i] * b[i];
    
    return result;
}

/* sum = sum(a[0...n-1]) */
float vsum(const float *__restrict__ a, size_t n)
{
    size_t i = 0;
    size_t step = sizeof(v8sf) / sizeof(float); 

    v8sf sum_vec = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (; i + step <= n; i += step)
        sum_vec = *(const v8sf *)(a + i) + sum_vec;

    float temp[8];
    *(v8sf *)temp = sum_vec;
    float total = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < n; ++i)
        total += a[i];

    return total;
}

/* ----- Matrix OP ----- */
/* C = A + B */
void mmadd(size_t m, size_t n, const float* A, const float* B, float* C) {
    vvadd(A, B, C, m * n);
}

/* C = A * B */
void mmmul(size_t m, size_t n, const float* A, const float* B, float* C) {
    vvmul(A, B, C, m * n);
}

/* C = A(m x k) @ B( n x k) -> (m x n)*/
void mmdot(size_t m, size_t n, size_t k, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {    
    #pragma omp parallel for collapse(2) if (m * n * k > 65536)
    for (size_t i = 0; i < m; ++i) 
        for (size_t j = 0; j < n; ++j) 
            C[i * n + j] = vvdot(A + i * k, B + j * k, k);  /* C[i, j] = A[i] @ B[j] */
}

/* Matrix Transpose */
void mtrans(size_t rows, size_t cols, const float* __restrict__ src, float* __restrict__ dst) {
    for (size_t r = 0; r < rows; ++r) 
        for (size_t c = 0; c < cols; ++c) 
            dst[c * rows + r] = src[r * cols + c];
}


/* ----- Matrix - Scale OP ----- */
/* C = A + scalar */
void msadd(size_t m, size_t n, const float* __restrict__ A, float scalar, float* __restrict__ C) {
    size_t total = m * n;
    for (size_t i = 0; i < total; ++i) 
        C[i] = A[i] + scalar;
}

/* C = A * scalar */
void msmul(size_t m, size_t n, const float* __restrict__ A, float scalar, float* __restrict__ C) {
    size_t total = m * n;
    for (size_t i = 0; i < total; ++i) 
        C[i] = A[i] * scalar;
}


/* ----- C = A(m x k) @ B.T( k x n) -> (m x n) ----- */
void mmtdot(size_t m, size_t n, size_t k, const float* A, const float* B, float* C) {
    size_t total_elements = k * n;
    float local_stack_buf[STACK_BUF_SIZE];
    
    float* worker = NULL;
    int onheap = 0;

    if (total_elements <= STACK_BUF_SIZE) 
        worker = local_stack_buf;
    else {
        worker = (float*)malloc(total_elements * sizeof(float));
        // error
        onheap = 1;
    }
    mtrans(k, n, B, worker);
    mmdot(m, n, k, A, worker, C);
    if (onheap) 
        free(worker);
}


/* ----- Matrix Sum ----- */
/* matrix sum all */
float msum(size_t m, size_t n, const float *__restrict__ A) {
    return vsum(A, m * n);
}

/* sum(axis=1)) */
void msum_row(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y) {
    for (size_t i = 0; i < m; ++i)
        y[i] = vsum(A + i * n, n);
}

/* sum(axis=0)) */
void msum_col(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y)
{
    size_t total_elements = m * n;
    float local_stack_buf[STACK_BUF_SIZE];

    float *worker = NULL;
    int onheap = 0; 

    if (total_elements <= STACK_BUF_SIZE)
        worker = local_stack_buf;
    else {
        worker = (float *)malloc(total_elements * sizeof(float));
            // error
        onheap = 1;
    }

    mtrans(m, n, A, worker);
    for (size_t j = 0; j < n; ++j)
        y[j] = vsum(worker + j * m, m);

    if (onheap)
        free(worker);
}