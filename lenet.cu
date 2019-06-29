
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
        if (fabs(hostResult[i] - gpuResult[i]) > 1.0e-1) {
            printf("check_result() failed at index %d\n", i);
            printf("GPU result: %f, Host result: %f\n",
                   gpuResult[i], hostResult[i]);
            exit(EXIT_FAILURE);
        }
    }
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

        /* Print pixel values */
        /* print_params("IMAGE", hostImage, IMAGE_SIZE); */

        /* Show image */
        /* show_image(hostImage, 28); */
        /* printf("\n"); */
        
        /* Feed-forward (CPU) */
        /* printf("Feed forward ...\n"); */
        /* fflush(stdout); */

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

        /* printf("CPU: time: %f ms\n", elapsedTime); */
        
        /* Print result */
        /* print_all_params(hostFc2Out, 10); */
        /* printf("\n"); */

        /* Feed-Forward (GPU) */
        /* printf("Feed forward (GPU) ...\n"); */
        /* fflush(stdout); */

        cudaEventRecord(startEvent, 0);

        CUDA_SAFE_CALL(cudaMemcpy(devImage, hostImage,
                                  IMAGE_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));
        
        block.x = 32;
        block.y = 32;
        block.z = 1;

        grid.x = (24 + block.x - 1) / block.x;
        grid.y = (24 + block.y - 1) / block.y;
        grid.z = 20;

        convolution_gpu_naive<<<grid, block>>>(
            devImage, 28, 1, devConv1Out, 24, 20,
            devConv1Weight, devConv1Bias, 5, 1);
        CUDA_SAFE_CALL(cudaGetLastError());
        
        block.x = 32;
        block.y = 32;
        block.z = 1;

        grid.x = (12 + block.x - 1) / block.x;
        grid.y = (12 + block.y - 1) / block.y;
        grid.z = 20;

        maxpooling_gpu_kernel_2x2<<<grid, block>>>(
            devConv1Out, 24, 20, devPool1Out, 12, 2);
        CUDA_SAFE_CALL(cudaGetLastError());
        
        block.x = 32;
        block.y = 32;
        block.z = 1;

        grid.x = (8 + block.x - 1) / block.x;
        grid.y = (8 + block.y - 1) / block.y;
        grid.z = 50;

        convolution_gpu_naive<<<grid, block>>>(
            devPool1Out, 12, 20, devConv2Out, 8, 50,
            devConv2Weight, devConv2Bias, 5, 1);
        
        block.x = 32;
        block.y = 32;
        block.z = 1;

        grid.x = (4 + block.x - 1) / block.x;
        grid.y = (4 + block.y - 1) / block.y;
        grid.z = 50;

        maxpooling_gpu_kernel_2x2<<<grid, block>>>(
            devConv2Out, 8, 50, devPool2Out, 4, 2);
        CUDA_SAFE_CALL(cudaGetLastError());
        
        /*
        block.x = 32;
        block.y = 1;
        block.z = 1;

        grid.x = (500 + block.x - 1) / block.x;
        grid.y = 1;
        grid.z = 1;

        classifier_gpu_naive<<<grid, block>>>(
            devPool2Out, 800, devFc1Out, 500, devFc1Weight, devFc1Bias);
        */
        
        /*
        block.x = 1;
        block.y = 32;
        block.z = 1;

        grid.x = 1;
        grid.y = (500 + block.y - 1) / block.y;
        grid.z = 1;

        classifier_gpu_blocked<<<grid, block>>>(
            devPool2Out, 800, devFc1Out, 500, devFc1Weight, devFc1Bias);
        CUDA_SAFE_CALL(cudaGetLastError());

        block.x = 32;
        block.y = 32;
        block.z = 1;

        grid.x = (1 + block.x - 1) / block.x;
        grid.y = (1 + block.y - 1) / block.y;
        grid.z = 500;

        relu_gpu_naive<<<grid, block>>>(devFc1Out, 1, 500);
        CUDA_SAFE_CALL(cudaGetLastError());
        */

        block.x = 1;
        block.y = 32;
        block.z = 1;

        grid.x = 1;
        grid.y = (500 + block.y - 1) / block.y;
        grid.z = 1;

        classifier_gpu_blocked_and_relu<<<grid, block>>>(
            devPool2Out, 800, devFc1Out, 500, devFc1Weight, devFc1Bias);
        CUDA_SAFE_CALL(cudaGetLastError());
        
        /*
        block.x = 32;
        block.y = 1;
        block.z = 1;

        grid.x = 1;
        grid.y = 1;
        grid.z = 1;

        classifier_gpu_naive<<<grid, block>>>(
            devFc1Out, 500, devFc2Out, 10, devFc2Weight, devFc2Bias);
        */
        
        block.x = 1;
        block.y = 32;
        block.z = 1;

        grid.x = 1;
        grid.y = 1;
        grid.z = 1;

        classifier_gpu_blocked<<<grid, block>>>(
            devFc1Out, 500, devFc2Out, 10, devFc2Weight, devFc2Bias);
        CUDA_SAFE_CALL(cudaGetLastError());

        CUDA_SAFE_CALL(cudaMemcpy(gpuFc2Out, devFc2Out,
                                  FC2_OUT_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        softmax(gpuFc2Out, 10);

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        gpuTimeSum += elapsedTime;

        /* printf("GPU: time: %f ms\n", elapsedTime); */

        check_result(hostFc2Out, gpuFc2Out, 10);
        
        /* Print result */
        /* print_all_params(gpuFc2Out, 10); */
        /* printf("\n"); */
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

