#include <stddef.h>
#include <stdlib.h>

#define STACK_BUF_SIZE 4096

typedef float v8sf __attribute__((vector_size(32)));

void mmadd(size_t m, size_t n, const float *A, const float *B, float *C);
void mmmul(size_t m, size_t n, const float *A, const float *B, float *C);
void mmdot(size_t m, size_t n, size_t k, const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C);

void msadd(size_t m, size_t n, const float *__restrict__ A, float scalar, float *__restrict__ C);
void msmul(size_t m, size_t n, const float *__restrict__ A, float scalar, float *__restrict__ C);

void mtrans(size_t rows, size_t cols, const float *__restrict__ src, float *__restrict__ dst);

void mmtdot(size_t m, size_t n, size_t k, const float *A, const float *B, float *C);

float msum(size_t m, size_t n, const float *__restrict__ A);
void msum_row(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y);
void msum_col(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y);