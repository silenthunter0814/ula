#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STACK_BUF_SIZE 4096

typedef float (*FUNC)(float);

void mtrans(size_t rows, size_t cols, const float *__restrict__ src, float *__restrict__ dst);

void mmadd(size_t m, size_t n, const float *A, const float *B, float *C);
void mmsub(size_t m, size_t n, const float *A, const float *B, float *C); 
void mmmul(size_t m, size_t n, const float *A, const float *B, float *C);
void mmdot(size_t m, size_t n, size_t k, const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C);
void mmtdot(size_t m, size_t k, size_t n, const float *A, const float *B, float *C);

void msadd(size_t m, size_t n, const float *__restrict__ A, float scalar, float *__restrict__ C);
void msmul(size_t m, size_t n, const float *__restrict__ A, float scalar, float *__restrict__ C);

float msum(size_t m, size_t n, const float *__restrict__ A);
void msum_row(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y);
void msum_col(size_t m, size_t n, const float *__restrict__ A, float *__restrict__ y);

float SE(size_t m, size_t n, const float *e);
float MSE(size_t m, size_t n, const float *e);
float SE2(size_t m, size_t n, const float *y, const float *t);
float MSE2(size_t m, size_t n, const float *y, const float *t);

void vmf(FUNC f, const float *__restrict__ x, float *__restrict__ y, size_t n) ;

float converge(size_t m, size_t n, const float *e0, const float *e, float toc);