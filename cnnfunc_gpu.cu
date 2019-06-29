
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
    float sum;
    
    if (ocol >= osize || orow >= osize || och >= ochan)
        return;

    sum = devBias[och];
    
    for (krow = 0; krow < ksize; ++krow)
        for (kcol = 0; kcol < ksize; ++kcol)
            for (kch = 0; kch < ichan; ++kch)
                sum += pWeight[kch * ksize * ksize + krow * ksize + kcol] *
                       pInput[kch * isize * isize + krow * isize + kcol];

    devOutput[outputIdx] = sum;
}

__host__ void im2col(
    float* input, int isize, int ichan,
    float* output,
    int ksize, int stride)
{
    int orow;
    int ocol;
    int kch;
    int krow;
    int kcol;

    int osize = (isize - ksize) / stride + 1;
    
    for (orow = 0; orow < osize; ++orow)
        for (ocol = 0; ocol < osize; ++ocol)
            for (kch = 0; kch < ichan; ++kch)
                for (krow = 0; krow < ksize; ++krow)
                    for (kcol = 0; kcol < ksize; ++kcol)
                        *(output++) = *(input + kch * isize * isize + 
                                        (krow + orow * stride) * isize +
                                        (kcol + ocol * stride));
}

__global__ void im2col_gpu(
    float* devInput, int isize, int ichan,
    float* devOutput, int osize,
    int ksize, int stride)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int kch = blockIdx.z;
    int krow;
    int kcol;

    for (krow = 0; krow < ksize; ++krow)
        for (kcol = 0; kcol < ksize; ++kcol)
            *(devOutput + (orow * osize + ocol) * ksize * ksize * ichan +
              kch * ksize * ksize + krow * ksize + kcol) =
                *(devInput + kch * isize * isize +
                  (krow + orow * stride) * isize +
                  (kcol + ocol * stride));
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

__global__ void maxpooling_gpu_kernel_2x2(
    float* devInput, int isize, int ichan,
    float* devOutput, int osize,
    int stride)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;

    float tmp0;
    float tmp1;
    float tmp2;
    float tmp3;
    float tmp4;
    float tmp5;

    int outputIdx = och * osize * osize + orow * osize + ocol;
    int inputOffset = och * isize * isize +
                      (orow * stride) * isize +
                      (ocol * stride);
    
    if (ocol >= osize || orow >= osize || och >= ichan)
        return;

    tmp0 = devInput[inputOffset];
    tmp1 = devInput[inputOffset + 1];
    tmp2 = devInput[inputOffset + isize];
    tmp3 = devInput[inputOffset + isize + 1];

    tmp4 = max(tmp0, tmp1);
    tmp5 = max(tmp2, tmp3);

    devOutput[outputIdx] = max(tmp4, tmp5);
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
    int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;

    float tmp = 0.0f;

    __shared__ float subInput[32];
    
    #pragma unroll
    for (i = weightIdxBegin, j = 0; i < weightIdxEnd; i += 32, j += 32) {
        if (j + threadIdx.y < isize)
            subInput[threadIdx.y] = devInput[j + threadIdx.y];
        else
            subInput[threadIdx.y] = 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (k = 0; k < 32; ++k)
            tmp += devWeight[i + isize * threadIdx.y + k] * subInput[k];

        __syncthreads();
    }

    if (outputIdx < osize)
        devOutput[outputIdx] = tmp;
}

__global__ void classifier_gpu_blocked_and_relu(
    float* devInput, int isize,
    float* devOutput, int osize,
    float* devWeight, float* devBias)
{
    int i;
    int j;
    int k;

    int weightIdxBegin = isize * (32 * blockIdx.y);
    int weightIdxEnd = weightIdxBegin + isize;
    int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;

    float tmp = 0.0f;

    __shared__ float subInput[32];
    
    #pragma unroll
    for (i = weightIdxBegin, j = 0; i < weightIdxEnd; i += 32, j += 32) {
        if (j + threadIdx.y < isize)
            subInput[threadIdx.y] = devInput[j + threadIdx.y];
        else
            subInput[threadIdx.y] = 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (k = 0; k < 32; ++k)
            tmp += devWeight[i + isize * threadIdx.y + k] * subInput[k];

        __syncthreads();
    }

    if (outputIdx < osize)
        devOutput[outputIdx] = tmp * (tmp > 0.0f);
}

