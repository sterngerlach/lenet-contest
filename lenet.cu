
/* lenet.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "header.h"

#define IMAGE_FILE       "./txt/image1000/"
#define CHECK_PARAMS    (0)

#define IMAGE_SIZE      (1 * 28 * 28)

#define CONV1_W_SIZE    (20 * 1 * 5 * 5)
#define CONV1_B_SIZE    (20)
#define CONV1_OUT_SIZE  (20 * 24 * 24)

#define POOL1_OUT_SIZE  (20 * 12 * 12)

#define CONV2_W_SIZE    (50 * 20 * 5 * 5)
#define CONV2_B_SIZE    (50)
#define CONV2_OUT_SIZE  (50 * 8 * 8)

#define POOL2_OUT_SIZE  (50 * 4 * 4)

#define FC1_W_SIZE      (500 * 800)
#define FC1_B_SIZE      (500)
#define FC1_OUT_SIZE    (500)

#define FC2_W_SIZE      (10 * 500)
#define FC2_B_SIZE      (10)
#define FC2_OUT_SIZE    (10)

#define CUDA_SAFE_CALL(call)                                                \
    do {                                                                    \
        cudaError_t err = (call);                                           \
                                                                            \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Error (%s:%d), code: %d, reason: %s\n",        \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

void check_result(float* hostResult, float* gpuResult, int size)
{
    int i;

    for (i = 0; i < size; ++i) {
        if (fabs(hostResult[i] - gpuResult[i]) > 5.0e-2) {
            printf("check_result() failed at index %d\n", i);
            printf("GPU result: %f, Host result: %f\n",
                   gpuResult[i], hostResult[i]);
            printf("\n");
            
            printf("GPU result: \n");
            print_all_params(gpuResult, size);
            printf("\n");

            printf("Host result: \n");
            print_all_params(hostResult, size);

            exit(EXIT_FAILURE);
        }
    }
}

template <int BlockSize,
          int InputSize, int InputChannels,
          int OutputSize, int OutputChannels,
          int KernelSize,
          int PoolOutputSize, int Stride>
__global__ void convolution_gpu_shared_memory_2_maxpooling_2x2(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias,
    float* devPoolOutput)
{
    /* Assumptions: blockDim.x == 4, blockDim.y == 4 */
    /* Assumptions: KernelSize == 5, Stride == 2 */

    int i;

    int ocol = threadIdx.x + blockIdx.x * blockDim.x;
    int orow = threadIdx.y + blockIdx.y * blockDim.y;
    int och = blockIdx.z;
    int ich = threadIdx.z;

    int icol;
    int irow;
    
    int kcol;
    int krow;
    
    /* const int outputIdx = och * OutputSize * OutputSize + orow * OutputSize + ocol; */
    const int ochOffset = och * InputChannels * KernelSize * KernelSize;
    const int inputOffset = ich * InputSize * InputSize;
    const int kernelOffset = ich * KernelSize * KernelSize;
    const int tmpOffset = inputOffset + orow * InputSize + ocol;
    
    float* pWeight = devWeight + ochOffset;
    float tmp = 0.0f;
    float sum;

    const int KernelRadius = KernelSize / 2;
    const int SharedInputSize = BlockSize + KernelRadius * 2;

    __shared__ float sharedInput[InputChannels][SharedInputSize][SharedInputSize];
    __shared__ float sharedWeight[InputChannels][KernelSize][KernelSize];
    __shared__ float sharedResult[InputChannels][BlockSize][BlockSize];

    if (ocol >= OutputSize || orow >= OutputSize)
        return;

    sharedInput[ich][threadIdx.y][threadIdx.x] =
        // devInput[inputOffset + irow * InputSize + icol];
        devInput[tmpOffset];

    icol = ocol + KernelRadius * 2;

    if (icol < InputSize)
        sharedInput[ich][threadIdx.y][threadIdx.x + KernelRadius * 2] =
            // devInput[inputOffset + irow * InputSize + icol];
            devInput[tmpOffset + KernelRadius * 2];
    
    irow = orow + KernelRadius * 2;

    if (irow < InputSize)
        sharedInput[ich][threadIdx.y + KernelRadius * 2][threadIdx.x] =
            // devInput[inputOffset + irow * InputSize + icol];
            devInput[tmpOffset + InputSize * KernelRadius * 2];
    
    icol = ocol + KernelRadius * 2;
    irow = orow + KernelRadius * 2;

    if (icol < InputSize && irow < InputSize)
        sharedInput[ich][threadIdx.y + KernelRadius * 2][threadIdx.x + KernelRadius * 2] =
            // devInput[inputOffset + irow * InputSize + icol];
            devInput[tmpOffset + InputSize * KernelRadius * 2 + KernelRadius * 2];
    
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
    
    // sharedResult[ich][threadIdx.y][threadIdx.x] = 0.0f;
    
    #pragma unroll
    for (krow = 0; krow < KernelSize; ++krow)
        #pragma unroll
        for (kcol = 0; kcol < KernelSize; ++kcol)
            // sharedResult[ich][threadIdx.y][threadIdx.x] +=
            tmp += 
                sharedWeight[ich][krow][kcol] *
                sharedInput[ich][threadIdx.y + krow][threadIdx.x + kcol];
    
    sharedResult[ich][threadIdx.y][threadIdx.x] = tmp;

    __syncthreads();

    if (ich == 0) {
        sum = devBias[och];
        
        #pragma unroll
        for (i = 0; i < InputChannels; ++i)
            sum += sharedResult[i][threadIdx.y][threadIdx.x];

        /* devOutput[outputIdx] = sum; */
        sharedResult[0][threadIdx.y][threadIdx.x] = sum;
    }

    /* Max pooling */
    if (ich == 0 && threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0) {
        float tmp[6];

        tmp[0] = sharedResult[0][threadIdx.y][threadIdx.x];
        tmp[1] = sharedResult[0][threadIdx.y][threadIdx.x + 1];
        tmp[2] = sharedResult[0][threadIdx.y + 1][threadIdx.x];
        tmp[3] = sharedResult[0][threadIdx.y + 1][threadIdx.x + 1];

        tmp[4] = fmaxf(tmp[0], tmp[1]);
        tmp[5] = fmaxf(tmp[2], tmp[3]);
        
        devPoolOutput[och * PoolOutputSize * PoolOutputSize +
                      (orow / 2) * PoolOutputSize +
                      (ocol / 2)] = fmaxf(tmp[4], tmp[5]);
    }

}

template <int BlockSize, int InputSize, int OutputSize>
__global__ void classifier_gpu_blocked_and_relu_template_2(
    float* devInput, float* devOutput,
    float* devWeight, float* devBias)
{
    int i;
    int k;

    int weightIdxBegin = InputSize * (BlockSize * blockIdx.y);
    int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;
    
    float* pWeight = devWeight + weightIdxBegin;
    float tmp = 0.0f;

    __shared__ float subInput[BlockSize][BlockSize];
    __shared__ float subWeight[BlockSize][BlockSize];
    
    for (i = 0; i < InputSize; i += BlockSize) {
        /* This implementation wastes so many threads */
        // if (threadIdx.x == 0)
            if (i + threadIdx.y < InputSize)
                subInput[0][threadIdx.y] = devInput[i + threadIdx.y];
            else
                subInput[0][threadIdx.y] = 0.0f;

        subWeight[threadIdx.y][threadIdx.x] =
            // devWeight[i + InputSize * threadIdx.y + threadIdx.x];
            pWeight[i + InputSize * threadIdx.y + threadIdx.x];

        __syncthreads();
        
        // if (threadIdx.x == 0)
        #pragma unroll
        for (k = 0; k < BlockSize; ++k)
            tmp += subWeight[threadIdx.y][k] * subInput[0][k];
    }
    
    if (threadIdx.x == 0 && outputIdx < OutputSize)
        if (tmp > 0)
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

    int weightIdxBegin = InputSize * (BlockSize * blockIdx.y);
    int outputIdx = threadIdx.y + blockDim.y * blockIdx.y;
    
    float* pWeight = devWeight + weightIdxBegin + InputSize * threadIdx.y;
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
    
    devOutput[outputIdx] = subOutput[outputIdx] / sum;
}

int main()
{
    int imageCount = 0;
    char imageFileName[64];

    float* hostImage;

    float* hostConv1Weight;
    float* hostConv1Bias;
    float* hostConv1Out;
    float* hostPool1Out;
  
    float* hostConv2Weight;
    float* hostConv2Bias;
    float* hostConv2Out;
    float* hostPool2Out;

    float* hostFc1Weight;
    float* hostFc1Bias;
    float* hostFc1Out;

    float* hostFc2Weight;
    float* hostFc2Bias;
    float* hostFc2Out;
    
    float* devImage;

    float* devConv1Weight;
    float* devConv1Bias;
    float* devConv1Out;
    float* devPool1Out;

    float* devConv2Weight;
    float* devConv2Bias;
    float* devConv2Out;
    float* devPool2Out;

    float* devFc1Weight;
    float* devFc1Bias;
    float* devFc1Out;

    float* devFc2Weight;
    float* devFc2Bias;
    float* devFc2Out;
    
    float* gpuFc2Out;

    dim3 block;
    dim3 grid;

    dim3 blockConv1(4, 4, 1);
    dim3 gridConv1(8, 8, 20);

    dim3 blockConv2(4, 4, 20);
    dim3 gridConv2(4, 4, 50);

    dim3 blockFc1(16, 16, 1);
    dim3 gridFc1(1, (500 + blockFc1.y - 1) / blockFc1.y, 1);

    dim3 blockFc2(1, 16, 1);
    dim3 gridFc2(1, 1, 1);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    float elapsedTime;
    double gpuTimeSum = 0.0;
    double hostTimeSum = 0.0;
    
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    printf("/// LeNet ///\n");
    fflush(stdout);
    
    printf("Allocating host memory ...\n");
    fflush(stdout);

    hostImage = (float*)malloc(sizeof(float) * IMAGE_SIZE);

    hostConv1Weight = (float*)malloc(sizeof(float) * CONV1_W_SIZE);
    hostConv1Bias = (float*)malloc(sizeof(float) * CONV1_B_SIZE);
    hostConv1Out = (float*)malloc(sizeof(float) * CONV1_OUT_SIZE);
    hostPool1Out = (float*)malloc(sizeof(float) * POOL1_OUT_SIZE);
    
    hostConv2Weight = (float*)malloc(sizeof(float) * CONV2_W_SIZE);
    hostConv2Bias = (float*)malloc(sizeof(float) * CONV2_B_SIZE);
    hostConv2Out = (float*)malloc(sizeof(float) * CONV2_OUT_SIZE);
    hostPool2Out = (float*)malloc(sizeof(float) * POOL2_OUT_SIZE);

    hostFc1Weight = (float*)malloc(sizeof(float) * FC1_W_SIZE);
    hostFc1Bias = (float*)malloc(sizeof(float) * FC1_B_SIZE);
    hostFc1Out = (float*)malloc(sizeof(float) * FC1_OUT_SIZE);

    hostFc2Weight = (float*)malloc(sizeof(float) * FC2_W_SIZE);
    hostFc2Bias = (float*)malloc(sizeof(float) * FC2_B_SIZE);
    hostFc2Out = (float*)malloc(sizeof(float) * FC2_OUT_SIZE);

    gpuFc2Out = (float*)malloc(sizeof(float) * FC2_OUT_SIZE);
    
    printf("Reading parameters ...\n");
    
    /* Read Conv1 layer parameters */
    read_params("./txt/conv1_w.txt", hostConv1Weight, CONV1_W_SIZE);
    print_params("CONV1_W", hostConv1Weight, CONV1_W_SIZE);
    read_params("./txt/conv1_b.txt", hostConv1Bias, CONV1_B_SIZE);
    print_params("CONV1_B", hostConv1Bias, CONV1_B_SIZE);
    
    /* Read Conv2 layer parameters */
    read_params("./txt/conv2_w.txt", hostConv2Weight, CONV2_W_SIZE);
    print_params("CONV2_W", hostConv2Weight, CONV2_W_SIZE);
    read_params("./txt/conv2_b.txt", hostConv2Bias, CONV2_B_SIZE);
    print_params("CONV2_B", hostConv2Bias, CONV2_B_SIZE);
    
    /* Read Fc1 layer parameters */
    read_params("./txt/fc1_w.txt", hostFc1Weight, FC1_W_SIZE);
    print_params("FC1_W", hostFc1Weight, FC1_W_SIZE);
    read_params("./txt/fc1_b.txt", hostFc1Bias, FC1_B_SIZE);
    print_params("FC1_B", hostFc1Bias, FC1_B_SIZE);
    
    /* Read Fc2 layer parameters */
    read_params("./txt/fc2_w.txt", hostFc2Weight, FC2_W_SIZE);
    print_params("FC2_W", hostFc2Weight, FC2_W_SIZE);
    read_params("./txt/fc2_b.txt", hostFc2Bias, FC2_B_SIZE);
    print_params("FC2_B", hostFc2Bias, FC2_B_SIZE);
    
    printf("Allocating device memory ...\n");
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&devImage,
                              IMAGE_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv1Weight,
                              CONV1_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv1Bias,
                              CONV1_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv1Out,
                              CONV1_OUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devPool1Out,
                              POOL1_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv2Weight,
                              CONV2_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv2Bias,
                              CONV2_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devConv2Out,
                              CONV2_OUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devPool2Out,
                              POOL2_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc1Weight,
                              FC1_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc1Bias,
                              FC1_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc1Out,
                              FC1_OUT_SIZE * sizeof(float)));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc2Weight,
                              FC2_W_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc2Bias,
                              FC2_B_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&devFc2Out,
                              FC2_OUT_SIZE * sizeof(float)));
    
    printf("Transferring weight and bias data from host ...\n");
    
    CUDA_SAFE_CALL(cudaMemcpy(devConv1Weight, hostConv1Weight,
                              CONV1_W_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devConv1Bias, hostConv1Bias,
                              CONV1_B_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    
    CUDA_SAFE_CALL(cudaMemcpy(devConv2Weight, hostConv2Weight,
                              CONV2_W_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devConv2Bias, hostConv2Bias,
                              CONV2_B_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    
    CUDA_SAFE_CALL(cudaMemcpy(devFc1Weight, hostFc1Weight,
                              FC1_W_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devFc1Bias, hostFc1Bias,
                              FC1_B_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    
    CUDA_SAFE_CALL(cudaMemcpy(devFc2Weight, hostFc2Weight,
                              FC2_W_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(devFc2Bias, hostFc2Bias,
                              FC2_B_SIZE * sizeof(float),
                              cudaMemcpyHostToDevice));

    printf("\n");

    for (imageCount = 0; imageCount < 1000; ++imageCount) {
        sprintf(imageFileName, "%simage%03d.txt", IMAGE_FILE, imageCount);

        if (imageCount % 100 == 0) {
            printf("file: %s\n", imageFileName);
            fflush(stdout);
        }

        read_params(imageFileName, hostImage, IMAGE_SIZE);
        norm_image(hostImage, IMAGE_SIZE);

        /* Feed-forward (CPU) */
        cudaEventRecord(startEvent, 0);

        convolution(hostImage, 28, 1, hostConv1Out, 24, 20,
                    hostConv1Weight, hostConv1Bias, 5, 1);
        maxpooling(hostConv1Out, 24, 20, hostPool1Out, 12, 2, 2);

        convolution(hostPool1Out, 12, 20, hostConv2Out, 8, 50,
                    hostConv2Weight, hostConv2Bias, 5, 1);
        maxpooling(hostConv2Out, 8, 50, hostPool2Out, 4, 2, 2);

        classifier(hostPool2Out, 800, hostFc1Out, 500,
                   hostFc1Weight, hostFc1Bias);
        relu(hostFc1Out, 1, 500);

        classifier(hostFc1Out, 500, hostFc2Out, 10,
                   hostFc2Weight, hostFc2Bias);
        softmax(hostFc2Out, 10);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        hostTimeSum += (double)elapsedTime;

        /* Feed-Forward (GPU) */
        CUDA_SAFE_CALL(cudaMemcpy(devImage, hostImage,
                                  IMAGE_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));
        cudaEventRecord(startEvent, 0);

        convolution_gpu_shared_memory_2_maxpooling_2x2
            <4, 28, 1, 24, 20, 5, 12, 2><<<gridConv1, blockConv1>>>(
                devImage, NULL, devConv1Weight, devConv1Bias, devPool1Out);

        convolution_gpu_shared_memory_2_maxpooling_2x2
            <4, 12, 20, 8, 50, 5, 4, 2><<<gridConv2, blockConv2>>>(
                devPool1Out, NULL, devConv2Weight, devConv2Bias, devPool2Out);

        classifier_gpu_blocked_and_relu_template_2
            <16, 800, 500><<<gridFc1, blockFc1>>>(
                devPool2Out, devFc1Out, devFc1Weight, devFc1Bias);

        classifier_gpu_blocked_and_softmax_template
            <16, 500, 10><<<gridFc2, blockFc2>>>(
                devFc1Out, devFc2Out, devFc2Weight, devFc2Bias);

        CUDA_SAFE_CALL(cudaMemcpy(gpuFc2Out, devFc2Out,
                                  FC2_OUT_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        gpuTimeSum += elapsedTime;

        check_result(hostFc2Out, gpuFc2Out, 10);
    }

    printf("GPU implementation is %f times faster than CPU\n",
           hostTimeSum / gpuTimeSum);
    printf("Average processing time: CPU: %f ms, GPU: %f ms\n",
           hostTimeSum / 1000.0f, gpuTimeSum / 1000.0f);

    /* Free device memory */
    CUDA_SAFE_CALL(cudaFree(devImage));

    CUDA_SAFE_CALL(cudaFree(devConv1Weight));
    CUDA_SAFE_CALL(cudaFree(devConv1Bias));
    CUDA_SAFE_CALL(cudaFree(devConv1Out));
    CUDA_SAFE_CALL(cudaFree(devPool1Out));
    
    CUDA_SAFE_CALL(cudaFree(devConv2Weight));
    CUDA_SAFE_CALL(cudaFree(devConv2Bias));
    CUDA_SAFE_CALL(cudaFree(devConv2Out));
    CUDA_SAFE_CALL(cudaFree(devPool2Out));

    CUDA_SAFE_CALL(cudaFree(devFc1Weight));
    CUDA_SAFE_CALL(cudaFree(devFc1Bias));
    CUDA_SAFE_CALL(cudaFree(devFc1Out));

    CUDA_SAFE_CALL(cudaFree(devFc2Weight));
    CUDA_SAFE_CALL(cudaFree(devFc2Bias));
    CUDA_SAFE_CALL(cudaFree(devFc2Out));

    /* Free host memory */
    free(hostImage);

    free(hostConv1Weight);
    free(hostConv1Bias);
    free(hostConv1Out);
    free(hostPool1Out);
    
    free(hostConv2Weight);
    free(hostConv2Bias);
    free(hostConv2Out);
    free(hostPool2Out);

    free(hostFc1Weight);
    free(hostFc1Bias);
    free(hostFc1Out);
    
    free(hostFc2Weight);
    free(hostFc2Bias);
    free(hostFc2Out);

    free(gpuFc2Out);

    /* Reset device */
    CUDA_SAFE_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}

