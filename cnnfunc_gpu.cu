
/* cnnfunc_gpu.cu */

#include <cuda_runtime.h>

#include "header.h"

__global__ void convolution_gpu_naive(
    float* devInput, int isize, int ichan,
    float* devOutput, int osize, int ochan,
    float* devWeight, float* devBias,
    int ksize, int stride)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;
    
    int kcol;
    int krow;
    int kch;

    int outputIdx = och * osize * osize + orow * osize + ocol;
    int ochOffset = och * ichan * ksize * ksize;
    int strideOffset = orow * stride * isize + ocol * stride;

    float* pWeight = devWeight + ochOffset;
    float* pInput = devInput + strideOffset;
    
    if (ocol >= osize || orow >= osize || och >= ochan)
        return;

    devOutput[outputIdx] = 0.0f;

    for (krow = 0; krow < ksize; ++krow)
        for (kcol = 0; kcol < ksize; ++kcol)
            for (kch = 0; kch < ichan; ++kch)
                devOutput[outputIdx] +=
                    pWeight[kch * ksize * ksize + krow * ksize + kcol] *
                    pInput[kch * isize * isize + krow * isize + kcol];
    
    devOutput[outputIdx] += devBias[och];
}

__global__ void maxpooling_gpu_naive(
    float* devInput, int isize, int ichan,
    float* devOutput, int osize,
    int ksize, int stride)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;

    int kcol;
    int krow;
    float max;
    float tmp;

    int outputIdx = och * osize * osize + orow * osize + ocol;
    int inputOffset = och * isize * isize +
                      (orow * stride) * isize +
                      (ocol * stride);
    
    if (ocol >= osize || orow >= osize || och >= ichan)
        return;

    max = -256.0f;

    for (krow = 0; krow < ksize; ++krow) {
        for (kcol = 0; kcol < ksize; ++kcol) {
            tmp = devInput[inputOffset + krow * isize + kcol];

            if (max < tmp)
                max = tmp;
        }
    }

    devOutput[outputIdx] = max;
}

__global__ void relu_gpu_naive(
    float* devInput, int isize, int ichan)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;
    
    int inputIdx = och * isize * isize + orow * isize + ocol;
    
    if (ocol >= isize || orow >= isize || och >= ichan)
        return;

    devInput[inputIdx] *= (devInput[inputIdx] > 0.0f);
}

__global__ void classifier_gpu_naive(
    float* devInput, int isize,
    float* devOutput, int osize,
    float* devWeight, float* devBias)
{
    int orow = threadIdx.x + blockIdx.x * blockDim.x;
    int irow;
    
    int weightOffset = orow * isize;

    if (orow >= osize)
        return;

    devOutput[orow] = 0.0f;

    for (irow = 0; irow < isize; ++irow)
        devOutput[orow] += devWeight[weightOffset + irow] * devInput[irow];

    devOutput[orow] += devBias[orow];
}

__global__ void classifier_gpu_blocked(
    float* devInput, int isize,
    float* devOutput, int osize,
    float* devWeight, float* devBias)
{
    int i;
    int j;
    int k;

    int weightIdxBegin = isize * (32 * blockIdx.y);
    int weightIdxEnd = weightIdxBegin + isize;
    int weightIdxStep = 32;
    int weightIdx;

    int inputIdxBegin = 0;
    int inputIdxStep = 32;
    int outputIdx;

    float tmp = 0.0f;
        
    __shared__ float subWeight[32][32];
    __shared__ float subInput[32];

    for (i = weightIdxBegin, j = inputIdxBegin;
         i < weightIdxEnd;
         i += weightIdxStep, j += inputIdxStep) {

        weightIdx = i + isize * threadIdx.y + threadIdx.x;
        
        if (weightIdx < isize * osize)
            subWeight[threadIdx.y][threadIdx.x] = devWeight[weightIdx];

        /*
        #pragma unroll
        for (k = 0; k < 32; ++k)
            if (weightIdx + k < isize * osize)
                subWeight[threadIdx.y][k] = devWeight[weightIdx + k];
        */
        
        if (threadIdx.x == 0 && j + threadIdx.y < isize)
            subInput[threadIdx.y] = devInput[j + threadIdx.y];

        __syncthreads();
        
        if (threadIdx.x == 0) {
            #pragma unroll
            for (k = 0; k < 32; ++k)
                if (weightIdx + k < isize * osize && j + k < isize)
                    tmp += subWeight[threadIdx.y][k] * subInput[k];
        }

        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        outputIdx = 32 * blockIdx.y + threadIdx.y;

        if (outputIdx < osize)
            devOutput[outputIdx] = tmp;
    }
}

