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
    double pi = 0.0;
    double sum = 0.0;

    double x = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);  // 设置2线程
    auto start = high_resolution_clock::now();

#pragma omp parallel private(i,x, sum)   // 该子句表示i,x,sum变量对于每个线程是私有的,性能更好
    {
        int id;
        id = omp_get_thread_num();
        for (i = id, sum = 0.0; i < num_steps; i = i + NUM_THREADS) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp critical  // 指定代码段在同一时刻只能由一个线程进行执行
//#pragma omp atomic
        pi += sum * step;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << std::endl
              << "Time Cost:" << duration.count() << " ms" << std::endl;
    printf("%lf\n", pi);
}  // 共2个线程参加计算，其中线程0进行迭代步0,2,4,...线程1进行迭代步1,3,5,....当被指定为critical的代码段	正在被0线程执行时，1线程的执行也到达该代码段，则它将被阻塞知道0线程退出临界区