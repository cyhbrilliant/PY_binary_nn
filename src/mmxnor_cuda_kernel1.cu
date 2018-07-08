#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "mmxnor_cuda_kernel.h"


//horizontal
__global__ void AMatrix2Bin(float *a,int *a_bin,int a_rows,int pitch_a,int pitch_a_bin,int MaxBS,int BINSIZE){
    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;


    int maxThreads=MaxBS*a_rows;
    for(int id = bix*bdx+tix; id < maxThreads; id+=gdx*bdx) {
        int rid=id/MaxBS;
        int cid=id%MaxBS;

        int Integer=0;
        int base=1;
        for (int i=0;i<BINSIZE;i++){
            if (a[rid*pitch_a+(cid+1)*BINSIZE-1-i]==1.f){
                Integer+=base;
            }
            base=base<<1;
        }

        a_bin[rid*pitch_a_bin+cid]=Integer;
    }

}
//vetical
__global__ void BMatrix2Bin(float *b,int *b_bin,int b_cols,int pitch_b,int pitch_b_bin,int MaxBS,int BINSIZE){
    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;

    int maxThreads=MaxBS*b_cols;
    for(int id = bix*bdx+tix; id < maxThreads; id+=gdx*bdx) {
        int cid=id/MaxBS;
        int rid=id%MaxBS;

        int Integer=0;
        int base=1;
        for (int i=0;i<BINSIZE;i++){
            if (b[((rid+1)*BINSIZE-1-i)*pitch_b+cid]==1.f){
                Integer+=base;
            }
            base=base<<1;
        }

        b_bin[rid*pitch_b_bin+cid]=Integer;
    }

}

__device__ unsigned char __popcount_tab_device[256];//__constant__ is slower than __device__
__device__ int popcount (int x) {
  return __popcount_tab_device[(x >>  0) & 0xff]  
  + __popcount_tab_device[(x >>  8) & 0xff]  
  + __popcount_tab_device[(x >> 16) & 0xff] 
  + __popcount_tab_device[(x >> 24) & 0xff];
}

__global__ void MatrixMulXnor(int *a,int *b,int a_rows,int a_cols,
    int b_cols,float *result,int pitch_a,int pitch_b,
    int pitch_result,int BINSIZE,int RealMidSize){

    int tix=threadIdx.x;
    // int tiy=threadIdx.y;
    int bix=blockIdx.x;
    // int biy=blockIdx.y;
    int bdx=blockDim.x;
    // int bdy=blockDim.y;
    int gdx=gridDim.x;
    // int gdy=gridDim.y;

    int rest=(BINSIZE*a_cols-RealMidSize);
    for(int i=bix;i<a_rows;i+=gdx){
        for(int j=tix;j<b_cols;j+=bdx){
            // printf("i=%d ; j=%d\n",i,j);
            int sum=0;
            for(int k=0;k<a_cols;k++){
                int bin=(a[i*pitch_a+k]^b[k*pitch_b+j]);
                int negnum=popcount(bin);
                int posnum=BINSIZE-negnum;
                //calculate ignores the rest of BINSIZE if the Matsize cant devided by BINSIZE ,it can cause err
                //(10/00)'(01/00) should be 0000 but it is 0011,so 1+1 is trash in the result.and it can cause a_rows*b_cols times. 
                sum+=(posnum-negnum);
            }
            result[i*pitch_result+j]=sum-rest;
        }
    }


}


void MatrixMul_Xnor(float *a,float *b,int a_rows,int a_cols,int b_cols,float *result,cudaStream_t stream){

    int BINSIZE=30;//size of bin2int, 32 means 0000 0000 0000 0000 0000 0000 0000 0000
    int MaxBS=(a_cols-1)/BINSIZE+1;
    int a_cols_Copysize=MaxBS*BINSIZE;
    dim3 BS_BIN(512,1,1);
    dim3 GS_BIN(6,1,1);
    // fprintf(stderr, "MaxBS : %d\n", MaxBS);

    float *a_device;//a_rows * a_cols_Copysize
    float *b_device;//a_cols_Copysize * b_cols
    size_t pitch_a_device, pitch_b_device;
    cudaMallocPitch((void**)&a_device , &pitch_a_device , sizeof(float) *a_cols_Copysize , a_rows);
    cudaMallocPitch((void**)&b_device , &pitch_b_device , sizeof(float) *b_cols , a_cols_Copysize);
    cudaMemset(a_device, 0, pitch_a_device * a_rows);
    cudaMemset(b_device, 0, pitch_b_device * a_cols_Copysize);
    cudaMemcpy2D(a_device,pitch_a_device,a,sizeof(float) *a_cols ,sizeof(float) *a_cols, a_rows,cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(b_device,pitch_b_device,b,sizeof(float) *b_cols ,sizeof(float) *b_cols, a_cols,cudaMemcpyDeviceToDevice);


    int *a_device_bin;
    int *b_device_bin;
    size_t pitch_a_device_bin, pitch_b_device_bin;
    cudaMallocPitch((void**)&a_device_bin , &pitch_a_device_bin , sizeof(int) *MaxBS , a_rows);
    cudaMallocPitch((void**)&b_device_bin , &pitch_b_device_bin , sizeof(int) *b_cols , MaxBS);

    AMatrix2Bin<<<GS_BIN,BS_BIN>>>(a_device , a_device_bin , a_rows , 
        pitch_a_device/sizeof(float) , pitch_a_device_bin/sizeof(int) , MaxBS , BINSIZE);
    BMatrix2Bin<<<GS_BIN,BS_BIN>>>(b_device , b_device_bin , b_cols , 
        pitch_b_device/sizeof(float) , pitch_b_device_bin/sizeof(int) , MaxBS , BINSIZE);


    float *result_device;//a_rows * b_cols
    size_t pitch_result_device;
    cudaMallocPitch((void**)&result_device , &pitch_result_device , sizeof(float) *b_cols , a_rows);

    const unsigned char __popcount_tab[] = {
      0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
      3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8,
    };
    cudaMemcpyToSymbol(__popcount_tab_device, __popcount_tab, sizeof(__popcount_tab));


    dim3 BS_MM(1024,1,1);
    dim3 GS_MM(120,1,1);
    MatrixMulXnor<<<GS_MM,BS_MM>>>(a_device_bin , b_device_bin , a_rows , MaxBS , b_cols ,
     result_device , pitch_a_device_bin/sizeof(int) , pitch_b_device_bin/sizeof(int) , 
     pitch_result_device/sizeof(float) , BINSIZE , a_cols);


    cudaMemcpy2D(result,sizeof(float) *b_cols, result_device,pitch_result_device,sizeof(float) *b_cols , a_rows ,cudaMemcpyDeviceToDevice);

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(a_device_bin);
    cudaFree(b_device_bin);
    cudaFree(result_device);


    cudaError_t err;
    err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}