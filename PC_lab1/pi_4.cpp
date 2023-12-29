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
    
    // 每个线程保留一份私有拷贝sum，x为线程私有，最后对线程中所以sum进行+规约，并更新sum的全局值
#pragma omp parallel for reduction(+ \
                                   : sum) private(x)
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    pi = sum * step;
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << std::endl
              << "Time Cost:" << duration.count() << " ms" << std::endl;
    printf("%lf\n", pi);
}  // 共2个线程参加计算，其中线程0进行迭代步0~49999，线程1进行迭代步50000~99999