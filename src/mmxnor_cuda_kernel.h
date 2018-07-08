#ifndef _MMXNOR_CUDA_KERNEL
#define _MMXNOR_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void MatrixMul_Xnor(float *a,float *b,int a_rows,int a_cols,int b_cols,float *result,cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
