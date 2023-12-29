// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <time.h>
#include <chrono>
#include <iostream>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
//#include <cutil_inline.h>


using namespace std::chrono;

// Thread block size, block 中线程的数目
#define BLOCK_WIDTH 16
#define MATRIX_SIZE 16
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (10 * BLOCK_WIDTH * MATRIX_SIZE) // Matrix A width
#define HA (10 * BLOCK_WIDTH * MATRIX_SIZE) // Matrix A height
#define WB (10 * BLOCK_WIDTH * MATRIX_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height


//sequential code implemented on cpu
void matrixMulOnHost(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB) {
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

// Initialize a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

//Compare the cpu's result with gpu's 
void printDiff(float* data1, float* data2, int width, int height) {
    int i, j, k;
    int error_count = 0;
    for (j = 0; j < height; j++) {
        for (i = 0; i < width; i++) {
            k = j * width + i;
            if (std::abs(data1[k] - data2[k])>0.0001) {
                printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", i, j, data1[k], data2[k]);
                error_count++;
            }
        }
    }
    printf("nTotal Errors = %d", error_count);
}

//the function is executed on gpu,属于GPU的函数,前缀__device__
__device__  float* GetSubMatrix(float* matrix, int sub_x, int sub_y, int width) {
    return  matrix + width * BLOCK_WIDTH * sub_y + BLOCK_WIDTH * sub_x;
}

//Kernel code
__global__ void matrixMulKernel(float* C, float* A, float* B, int wA, int wB) {
    // Declaration of the shared memory array As used to
    //store the sub-matrix of A
    // 同一个线程块内的线程共享内存
    __shared__ float As[BLOCK_WIDTH][BLOCK_WIDTH];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_WIDTH][BLOCK_WIDTH];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index，子矩阵内标号
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix

    for (int m = 0; m < wA / BLOCK_WIDTH; m++) {//wA==hB 子矩阵的个数
        //get the address of submatrixA
        //float *subA=A+wA*BLOCK_WIDTH*by+BLOCK_WIDTH*m;
        float* subA = GetSubMatrix(A, m, by, wA);
        //get the address of submatrixB
        //float *subB=B+wB*BLOCK_WIDTH*m+BLOCK_WIDTH*bx;
        float* subB = GetSubMatrix(B, bx, m, wB);
        //统一线程块给As,Bs赋值
        As[ty][tx] = *(subA + wA * ty + tx);
        Bs[ty][tx] = *(subB + wB * ty + tx);

        // Synchronize to make sure the matrices are loaded
        //实现同一块内线程同步，实现子矩阵赋值同步
        __syncthreads();//虽然visual stuido报错，但可以运行

        // 计算线程块中tx,ty处对应的元素，
        // Csub是一个局部值，需要外循环结束，遍历A的所有列子矩阵，B的所有行子矩阵，才能得到最终的Csub
        for (int k = 0; k < BLOCK_WIDTH; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    //float *subC = C+wB * BLOCK_WIDTH * by + BLOCK_WIDTH * bx;
    float* subC = GetSubMatrix(C, bx, by, wB);
    *(subC + wB * ty + tx) = Csub;
}


int main(int argc, char** argv) {
    //----------------------------------------------------
    /*step1: host执行代码*/

    // set seed for rand()
    srand((unsigned)time(NULL));

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    //std::cout << "Generate Matrix A and B......" << std::endl;
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    printf("Matrix A is %d by %d, Matrix B is %d by %d\n", HA,WA,HB,WB);
    //--------------------------------------------------------
    /*step2: 传数据到GPU device*/

    // allocate device memory
    float* d_A;
    //cudaMalloc: 分配线性存储空间
    //void** 表示的才是真正的数组指针
    cudaMalloc((void**)&d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**)&d_B, mem_size_B);

    // copy host memory to device
    //cudaMemcpy: 从主机内存复制数据到设备内存
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**)&d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*)malloc(mem_size_C);


    //-------------------------------------------------------
    /*step3: GPU执行kernel函数*/

    // cuda记录时间的内置器：
    // create and start timer
    unsigned int timer = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // setup execution parameters
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid(WC / dimBlock.x, HC / dimBlock.y);

    // execute the kernel
    //<<<DimGrid, DimBlock, MemSize>>>指定kernel的执行配置:
    //DimGrid:指定grid的维度，类型是dim3
    //DimBlock:指定block的维度，类型是dim3
    //MemSize:指定动态分配的共享内存的大小，类型是unsigned int,可省略
    matrixMulKernel <<< dimGrid, dimBlock >>> (d_C, d_A, d_B, WA, WB);//这个报错不用管
    cudaThreadSynchronize();//CPU等待GPU执行完毕，保证串行
     
    // stop and destroy timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU Processing time: %f (ms)\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //-------------------------------------------------------
    /*step4: 将数据传回host代码*/

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);


    //-------------------------------------------------------
    /*step5: 继续执行host*/

    // compute reference solution
    //进行CPU计算，得到结果reference
    float* reference = (float*)malloc(mem_size_C);
    auto t1 = high_resolution_clock::now();
    matrixMulOnHost(reference, h_A, h_B, HA, WA, WB);
    auto t2 = high_resolution_clock::now();
    std::chrono::duration<double, std::milli>  dura = (t2 - t1);
    std::cout << "CPU Processing time : " << dura.count() << "(ms)\n";

    // check result
    float error_norm = 0;
    float ref_norm = 0;
    for (int i = 0; i < WC * HC; ++i) {
        float diff = reference[i] - h_C[i];
        error_norm += diff * diff;
        ref_norm += reference[i] * reference[i];
    }
    error_norm = sqrt(error_norm);
    ref_norm = sqrt(ref_norm);
    if (fabs(ref_norm) < 1e-7) {
        printf("Test %s \n", (error_norm < 1e-5) ? "PASSED" : "FAILED");
    }
    else {
        printf("Test %s \n", ((error_norm / ref_norm) < 1e-6) ? "PASSED" : "FAILED");
    }
    if (error_norm >= 1e-5f&&WC*HC<100) printDiff(reference, h_C, WC, HC);

    //-------------------------------------------------------
    /*清理分配的内存，结束*/

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaThreadExit();
    //cutilExit(argc, argv);
}

