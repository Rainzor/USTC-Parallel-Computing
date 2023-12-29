#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
using namespace std::chrono;
static long num_steps = 1000000000;
double step;
#define NUM_THREADS 8
int main() {
    int i;
    double x, pi, sum[NUM_THREADS];
	double paded_sum[NUM_THREADS][8]; // 64字节对齐
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  // 设置2线程
    auto start = high_resolution_clock::now();

#pragma omp parallel private(i)        // 并行域开始，每个线程(0和1)都会执行该代码
    {
        double x;
        int id;
        id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS) {
            x = (i + 0.5) * step;
            // sum[id] += 4.0 / (1.0 + x * x);//可能会出现阻塞现象，因为两个内存域在一起，会一次性锁64字节cache
			paded_sum[id][0] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++){
        // pi += sum[i] * step;
		pi += paded_sum[i][0] * step;
	}
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << std::endl
              << "Time Cost:" << duration.count() << " ms" << std::endl;
    printf("%lf\n", pi);
}
// 共2个线程参加计算，其中线程0进行迭代步0,2,4,...线程1进行迭代步1,3,5,....
