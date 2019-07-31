
/* lenet_old.cu */

template <int InputSize, int InputChannels,
          int OutputSize, int OutputChannels,
          int KernelSize>
__global__ void convolution_gpu_shared_memory(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias)
{
    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;

    int icol;
    int irow;
    
    int kcol;
    int krow;
    int kch;
    
    int outputIdx = och * OutputSize * OutputSize + orow * OutputSize + ocol;
    int ochOffset = och * InputChannels * KernelSize * KernelSize;

    float* pWeight = devWeight + ochOffset;
    float sum;

    __shared__ float sharedInput[InputChannels][InputSize][InputSize];

    if (ocol >= OutputSize || orow >= OutputSize || och >= OutputChannels)
        return;

    icol = ocol;
    irow = orow;
    
    for (kch = 0; kch < InputChannels; ++kch)
        sharedInput[kch][irow][icol] =
            devInput[kch * InputSize * InputSize + irow * InputSize + icol];

    icol = ocol + KernelSize;
    irow = orow;

    if (icol < InputSize)
        for (kch = 0; kch < InputChannels; ++kch)
            sharedInput[kch][irow][icol] =
                devInput[kch * InputSize * InputSize + irow * InputSize + icol];
    
    icol = ocol;
    irow = orow + KernelSize;

    if (irow < InputSize)
        for (kch = 0; kch < InputChannels; ++kch)
            sharedInput[kch][irow][icol] =
                devInput[kch * InputSize * InputSize + irow * InputSize + icol];
    
    icol = ocol + KernelSize;
    irow = orow + KernelSize;

    if (icol < InputSize && irow < InputSize)
        for (kch = 0; kch < InputChannels; ++kch)
            sharedInput[kch][irow][icol] =
                devInput[kch * InputSize * InputSize + irow * InputSize + icol];

    __syncthreads();
    
    sum = devBias[och];

    for (kch = 0; kch < InputChannels; ++kch)
        for (krow = 0; krow < KernelSize; ++krow)
            for (kcol = 0; kcol < KernelSize; ++kcol)
                sum += pWeight[kch * KernelSize * KernelSize + krow * KernelSize + kcol] *
                       sharedInput[kch][orow + krow][ocol + kcol];

    devOutput[outputIdx] = sum;
}

template <int BlockSize,
          int InputSize, int InputChannels,
          int OutputSize, int OutputChannels,
          int KernelSize>
__global__ void convolution_gpu_shared_memory_2(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias)
{
    int i;

    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;
    int ich = threadIdx.z;

    int icol;
    int irow;
    
    int kcol;
    int krow;
    
    const int outputIdx = och * OutputSize * OutputSize + orow * OutputSize + ocol;
    const int ochOffset = och * InputChannels * KernelSize * KernelSize;
    const int inputOffset = ich * InputSize * InputSize;
    const int kernelOffset = ich * KernelSize * KernelSize;
    
    float* pWeight = devWeight + ochOffset;
    float sum;

    const int KernelRadius = KernelSize / 2;
    const int SharedInputSize = BlockSize + KernelRadius * 2;

    __shared__ float sharedInput[InputChannels][SharedInputSize][SharedInputSize];
    __shared__ float sharedWeight[InputChannels][KernelSize][KernelSize];
    __shared__ float sharedResult[InputChannels][BlockSize][BlockSize];

    if (ocol >= OutputSize || orow >= OutputSize)
        return;

    icol = ocol;
    irow = orow;

    sharedInput[ich][threadIdx.y][threadIdx.x] =
        devInput[inputOffset + irow * InputSize + icol];

    icol = ocol + KernelRadius * 2;
    irow = orow;

    if (icol < InputSize)
        sharedInput[ich][threadIdx.y][threadIdx.x + KernelRadius * 2] =
            devInput[inputOffset + irow * InputSize + icol];
    
    icol = ocol;
    irow = orow + KernelRadius * 2;

    if (irow < InputSize)
        sharedInput[ich][threadIdx.y + KernelRadius * 2][threadIdx.x] =
            devInput[inputOffset + irow * InputSize + icol];
    
    icol = ocol + KernelRadius * 2;
    irow = orow + KernelRadius * 2;

    if (icol < InputSize && irow < InputSize)
        sharedInput[ich][threadIdx.y + KernelRadius * 2][threadIdx.x + KernelRadius * 2] =
            devInput[inputOffset + irow * InputSize + icol];
    
    /*
     * Hack: this code works because KernelSize is 5,
     * blockDim.x is 4, and blockDim.y is also 4
     */
    sharedWeight[ich][threadIdx.y][threadIdx.x] =
        pWeight[kernelOffset + threadIdx.y * KernelSize + threadIdx.x];
    sharedWeight[ich][threadIdx.y][threadIdx.x + 1] =
        pWeight[kernelOffset + threadIdx.y * KernelSize + threadIdx.x + 1];
    sharedWeight[ich][threadIdx.y + 1][threadIdx.x] =
        pWeight[kernelOffset + (threadIdx.y + 1) * KernelSize + threadIdx.x];
    sharedWeight[ich][threadIdx.y + 1][threadIdx.x + 1] =
        pWeight[kernelOffset + (threadIdx.y + 1) * KernelSize + threadIdx.x + 1];

    __syncthreads();
    
    sharedResult[ich][threadIdx.y][threadIdx.x] = 0.0f;
    
    for (krow = 0; krow < KernelSize; ++krow)
        for (kcol = 0; kcol < KernelSize; ++kcol)
            sharedResult[ich][threadIdx.y][threadIdx.x] +=
                sharedWeight[ich][krow][kcol] *
                sharedInput[ich][threadIdx.y + krow][threadIdx.x + kcol];
    
    __syncthreads();

    if (ich == 0) {
        sum = devBias[och];

        for (i = 0; i < InputChannels; ++i)
            sum += sharedResult[i][threadIdx.y][threadIdx.x];

        devOutput[outputIdx] = sum;
    }
}

template <int InputSize, int InputChannels,
          int OutputSize, int Stride>
__global__ void maxpooling_gpu_kernel_2x2_template(
    float* devInput, float* devOutput)
{
    int ocol = threadIdx.x;
    int orow = threadIdx.y;
    int och = blockIdx.z;

    float tmp0;
    float tmp1;
    float tmp2;
    float tmp3;
    float tmp4;
    float tmp5;

    int outputIdx = och * OutputSize * OutputSize + orow * OutputSize + ocol;
    int inputOffset = och * InputSize * InputSize +
                      (orow * Stride) * InputSize +
                      (ocol * Stride);
    
    if (ocol >= OutputSize || orow >= OutputSize || och >= InputChannels)
        return;

    tmp0 = devInput[inputOffset];
    tmp1 = devInput[inputOffset + 1];
    tmp2 = devInput[inputOffset + InputSize];
    tmp3 = devInput[inputOffset + InputSize + 1];

    tmp4 = fmaxf(tmp0, tmp1);
    tmp5 = fmaxf(tmp2, tmp3);

    devOutput[outputIdx] = fmaxf(tmp4, tmp5);
}

template <int BlockSize, int InputSize, int OutputSize>
__global__ void classifier_gpu_blocked_and_relu_template(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias)
{
    int i;
    int j;
    int k;

    int weightIdxBegin = InputSize * (BlockSize * blockIdx.y);
    int weightIdxEnd = weightIdxBegin + InputSize;
    int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;

    float tmp = 0.0f;

    __shared__ float subInput[BlockSize];
    
    for (i = weightIdxBegin, j = 0; i < weightIdxEnd;
         i += BlockSize, j += BlockSize) {
        if (j + threadIdx.y < InputSize)
            subInput[threadIdx.y] = devInput[j + threadIdx.y];
        else
            subInput[threadIdx.y] = 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (k = 0; k < BlockSize; ++k)
            tmp += devWeight[i + InputSize * threadIdx.y + k] * subInput[k];

        __syncthreads();
    }

    if (outputIdx < OutputSize)
        if (tmp > 0.0f)
            devOutput[outputIdx] = tmp;
        else
            devOutput[outputIdx] = 0.0f;
}

template <int BlockSize, int InputSize, int OutputSize>
__global__ void classifier_gpu_blocked_and_softmax_template(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias)
{
    int i;
    int k;

    // int weightIdxBegin = InputSize * (BlockSize * blockIdx.y);
    // int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;
    int weightIdxBegin = 0;
    int outputIdx = threadIdx.y;
    
    // float* pWeight = devWeight + weightIdxBegin + InputSize * threadIdx.y;
    float* pWeight = devWeight + InputSize * threadIdx.y;
    float tmp = 0.0f;
    float sum = 0.0f;

    __shared__ float subInput[BlockSize];
    __shared__ float subOutput[OutputSize];
    
    for (i = 0; i < InputSize; i += BlockSize) {
        if (i + threadIdx.y < InputSize)
            subInput[threadIdx.y] = devInput[i + threadIdx.y];
        else
            subInput[threadIdx.y] = 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (k = 0; k < BlockSize; ++k)
            tmp += pWeight[i + k] * subInput[k];
    }
    
    if (outputIdx < OutputSize)
        subOutput[outputIdx] = expf(tmp);

    __syncthreads();
    
    #pragma unroll
    for (k = 0; k < OutputSize; ++k)
        sum += subOutput[k];
    
    if (outputIdx < OutputSize)
        devOutput[outputIdx] = subOutput[outputIdx] / sum;
}

