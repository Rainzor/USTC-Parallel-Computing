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
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  // 设置2线程
    auto start = high_resolution_clock::now();

#pragma omp parallel                   // 并行域开始，每个线程(0和1)都会执行该代码
    {
        double x;
        int id;
        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for  // 未指定chunk，迭代平均分配给各线程（0和1），连续划分
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++)
        pi += sum[i] * step;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << std::endl
              << "Time Cost:" << duration.count() << " ms" << std::endl;
    printf("%lf\n", pi);
}  // 