#include <omp.h>
#include <stdio.h>
#define NUM_THREADS 4
int main() {
    int sum = 0;
    int n = 100;
    int i;
    omp_set_num_threads(NUM_THREADS);  
// 使用OpenMP并行计算
#pragma omp parallel
    {
#pragma omp master
        {
            // 只有一个线程执行加法运算
            for(i = 0; i < n*100; i++)
                sum = sum + i;
            printf("Thread %d is adding %d to sum\n", omp_get_thread_num(), sum);
        }

// 所有其他线程执行循环语句
#pragma omp for schedule(dynamic)
        for (i = 0; i < n; i++) {
            printf("Thread %d executing iteration %d\n", omp_get_thread_num(), i);
        }
    }

    printf("Sum = %d\n", sum);

    return 0;
}
