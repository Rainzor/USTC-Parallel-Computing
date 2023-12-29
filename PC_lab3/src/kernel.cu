#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
void __syncthreads() {}
#endif


// CUDA 核函数：elementwise_add_kernel
__global__ void elementwise_add_kernel(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    // 分配主机内存
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // 初始化输入数组
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * i;
    }

    // 分配设备内存
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置执行参数并调用 CUDA 核函数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    elementwise_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // 复制结果到主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 检查结果
    for (int i = 0; i < N; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
