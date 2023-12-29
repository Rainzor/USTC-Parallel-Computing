#include<iostream>
#include<random>
#include <chrono>

#include<cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std::chrono;
#define THREAD_NUM 512
#define VECTOR_SIZE 100000000
//设置
int block_num = std::min(65535, (VECTOR_SIZE + THREAD_NUM - 1) / THREAD_NUM);

std::default_random_engine generator(system_clock::now().time_since_epoch().count());
std::uniform_real_distribution<float> distribution(0, 1);

//#define BLOCK_NUM 
void vectorAddOnHost(float*c, const float* a,const float* b,const unsigned int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

__global__ void vectorAddOnKernel(float*c, float*a, float* b,unsigned int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < size) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


void computeAcceleration(int size) {
	float* a, * b, * c, * host_c;
	//分配数据
	a = (float*)malloc(size * sizeof(float));
	b = (float*)malloc(size * sizeof(float));
	c = (float*)malloc(size * sizeof(float));
	host_c = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		a[i] = distribution(generator);
		b[i] = distribution(generator);
	}
	float* dev_a, * dev_b, * dev_c;
	cudaMalloc((void**)&dev_a, size * sizeof(float));
	cudaMalloc((void**)&dev_b, size * sizeof(float));
	cudaMalloc((void**)&dev_c, size * sizeof(float));
	//向设备输入数据
	cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

	//计时
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//根据size设置block数目
	int new_block_num = std::min(65535, (size + THREAD_NUM - 1) / THREAD_NUM);
	//调用kernel函数
	vectorAddOnKernel <<<new_block_num, THREAD_NUM >>> (dev_c, dev_a, dev_b, size);
	cudaThreadSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//将结果从设备拷贝到主机
	cudaMemcpy(host_c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	auto t0 = high_resolution_clock::now();
	vectorAddOnHost(c, a, b, size);
	auto t1 = high_resolution_clock::now();
	std::chrono::duration<double, std::milli>  dura = (t1 - t0);
	float cputime = dura.count();
	std::cout <<"Thread num:"<< new_block_num * THREAD_NUM <<"\tVector size:"<< size << "\tAcceleration ratio: " << cputime / elapsedTime << "\n";

	//释放内存
	free(a);
	free(b);
	free(c);
	free(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}


int main() {
	int test_num[6] = { 100000,200000,1000000,2000000,10000000,20000000 };
	for (int i=0;i<6;i++) {
		computeAcceleration(test_num[i]);
	}

	//10000W数据测试和验证，与computeAcceleration函数功能相同
	/*float *a, *b, *c, *host_c;
	a = (float*)malloc(VECTOR_SIZE * sizeof(float));
	b = (float*)malloc(VECTOR_SIZE * sizeof(float));
	c = (float*)malloc(VECTOR_SIZE * sizeof(float));
	host_c = (float*)malloc(VECTOR_SIZE * sizeof(float));
	std::cout << "Vector is genrating..... \n";
	for(int i=0; i<VECTOR_SIZE;i++){
		a[i] = distribution(generator);
		b[i] = distribution(generator);
	}
	std::cout<<"Vector size: "<<VECTOR_SIZE<<std::endl;

	float * dev_a, *dev_b, *dev_c;
	cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(float));
	cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(float));
	cudaMalloc((void**)&dev_c, VECTOR_SIZE * sizeof(float));

	cudaMemcpy(dev_a, a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	vectorAddOnKernel << <block_num, THREAD_NUM >> > (dev_c, dev_a, dev_b, VECTOR_SIZE);
	cudaThreadSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU ADD Processing time: %f (ms)\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(host_c, dev_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	auto t0 = high_resolution_clock::now();
	vectorAddOnHost(c, a, b, VECTOR_SIZE);
    auto t1 = high_resolution_clock::now();
	std::chrono::duration<double, std::milli>  dura =(t1 - t0);
	float cputime = dura.count();
	std::cout << "CPU ADD Processing time : " << cputime << "(ms)\n";
	std::cout << "Acceleration ratio: " << cputime / elapsedTime << "\n";
	
	int i;
	for(i=0;i<VECTOR_SIZE;i++){
		if (c[i] != host_c[i]) {
			printf("Error: %f != %f\n", c[i], host_c[i]);
			break;
		}
	}
	if (i == VECTOR_SIZE)
		printf("Test PASSED\n");

	free(a);
	free(b);
	free(c);
	free(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);*/
	return 0;
}