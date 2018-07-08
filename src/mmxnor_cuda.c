#include <THC/THC.h>
#include "mmxnor_cuda_kernel.h"

extern THCState *state;

int mmxnor(THCudaTensor *a_tensor, THCudaTensor *b_tensor, THCudaTensor *c_tensor, int x, int n, int y){
    float *a = THCudaTensor_data(state, a_tensor);
    float *b = THCudaTensor_data(state, b_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

//  	THCudaTensor *c_tensor=THCudaTensor_newWithStorage1d(
//  		state,THCudaTensor_storage(state, a_tensor),THCudaTensor_storageOffset(state, a_tensor),x*y,0);
  	float *c = THCudaTensor_data(state, c_tensor);
    MatrixMul_Xnor(a, b, x, n, y, c, stream);
    return 1;
}