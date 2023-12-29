#include <omp.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using namespace std;
using namespace std::chrono;
#define NUM_THREADS 3
void PSRS(std::vector<int>& a);
void PSRS2(std::vector<int>& a);
void print(std::vector<int>& a) {
    int n = a.size();
    int i;
    if (n < 50) {
        std::cout << "Array data:\n";
#pragma omp parallel for ordered private(i)
        for (i = 0; i < n; i++) {
#pragma omp ordered
            std::cout << a[i] << " ";
        }
    }
    std::cout << std::endl;
}

//void mergeArrays(std::vector<int>& arr1, std::vector<int>& arr2) {
//    std::vector<int> result(arr1.size() + arr2.size());
//    std::merge(arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin());
//    arr1 = result;
//}

int main() {
    // 创建一个随机数生成器对象，使用当前时间作为种子
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

    // 创建一个均匀分布的整数分布器，范围为[0, 99]
    std::uniform_int_distribution<int> distribution(0, 99);
    int n, i;  // 数组大小
    std::cout << "Input the size of array: ";
    std::cin >> n;
    std::vector<int> a(n);
    
    //cout << "The random arrary data:" << endl;
    for (i = 0; i < n; i++) {
        a[i] = distribution(generator);
        //cout << a[i] << " ";
    }
    //std::cout << endl;

    print(a);
    std::vector<int> b(a), c(a);


    auto start = high_resolution_clock::now();
    std::cout << "\nPSRS processing by " << NUM_THREADS << " threads.......\n";
    PSRS(a);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << "PSRS Time Cost:" << duration.count() << " ms" << std::endl;

    print(a);

    start = high_resolution_clock::now();
    std::cout << "\nPSRS2 processing.......\n";
    PSRS2(b);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    std::cout << "PSRS2 Time Cost:" << duration.count() << " ms" << std::endl;

    print(b);

    start = high_resolution_clock::now();
    std::cout << "\nstd::sort processing.......\n";
    std::sort(c.begin(), c.end());
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    std::cout << "std::sort Time Cost:" << duration.count() << " ms" << std::endl;

    print(c);

    return 0;
}

void PSRS(std::vector<int>& data) {
    int n = data.size();
    int i, j;
    omp_set_num_threads(NUM_THREADS);  // 设置线程
    std::vector<int> sample_data(NUM_THREADS * NUM_THREADS);
    std::vector<int> backup(n);
    int pivot[NUM_THREADS - 1];
    int step = n / NUM_THREADS / NUM_THREADS;
    int divide[NUM_THREADS][NUM_THREADS];
    int len[NUM_THREADS][8];
    int offset[NUM_THREADS];
    int start, end, id;
#pragma omp parallel private(i, j,id, start, end)  // 局部排序
    {
        id = omp_get_thread_num();
        start = (id)*n / NUM_THREADS;
        end = id == (NUM_THREADS - 1) ? n : (id + 1) * n / NUM_THREADS;
        std::sort(data.begin() + start, data.begin() + end);

        for (i = 0; i < NUM_THREADS; i++) {  // 正则采样
            sample_data[id * NUM_THREADS + i] = data[start + i * step];
        }

        std::copy(data.begin() + start, data.begin() + end, backup.begin() + start);

#pragma omp barrier//同步障
#pragma omp master
        {
            std::sort(sample_data.begin(), sample_data.end());  // 采样排序
            for (i = 0; i < NUM_THREADS - 1; i++) {             // 选取主元
                pivot[i] = sample_data[(i + 1) * NUM_THREADS];
            }
        }
#pragma omp barrier //同步障
        //分割
        id = omp_get_thread_num();
        for (i = 0,j = start; i < NUM_THREADS - 1; i++) {
            while(data[j]<pivot[i]&&j<end)
                j++;
            divide[id][i] = j;
            // auto it = std::upper_bound(data.begin() + start, data.begin() + end, pivot[i]);
            // divide[id][i] = std::distance(data.begin(), it);
        }
        divide[id][NUM_THREADS - 1] = end;
#pragma omp barrier //同步障
        // 计算长度
        len[id][0] = 0;
        for (i = 0; i < NUM_THREADS; i++) {
            auto s = (i)*n / NUM_THREADS;
            len[id][0] += divide[i][id] - (id == 0 ? s : divide[i][id - 1]);
        }
#pragma omp barrier //同步障

#pragma omp master // 计算偏移量 与 备份
        {
            for (i = 0; i < NUM_THREADS; i++) {  
                offset[i] = i == 0 ? 0 : offset[i - 1] + len[i - 1][0];
            }
        }
#pragma omp barrier
        // 全局交换
        start = offset[id];
        end = id == NUM_THREADS - 1 ? n : offset[id + 1];
        for (i = 0; i < NUM_THREADS; i++) {
            int divide_start = id == 0 ? (i)*n / NUM_THREADS : divide[i][id - 1];
            int divide_end = divide[i][id];
            for (j = divide_start; j < divide_end && start < end; j++, start++) {
                data[start] = backup[j];
            }
        }
        // 再次局部排序
        start = offset[id];
        std::sort(data.begin() + start, data.begin() + end);
    }

}

void PSRS2(std::vector<int>& data){
    int n = data.size();
    int i, j;
    omp_set_num_threads(NUM_THREADS);  // 设置线程
    std::vector<int> sample_data(NUM_THREADS * NUM_THREADS);
    std::vector<int> backup(n);
    int pivot[NUM_THREADS - 1];
    int step = n / NUM_THREADS / NUM_THREADS;
    int divide[NUM_THREADS][NUM_THREADS];
    int len[NUM_THREADS][8];
    int offset[NUM_THREADS];
    int start, end, id;
    int debug;
#pragma omp parallel private(i, j, id, start, end)  // 局部排序
    {
        id = omp_get_thread_num();
        start = (id)*n / NUM_THREADS;
        end = id == (NUM_THREADS - 1) ? n : (id + 1) * n / NUM_THREADS;
        std::sort(data.begin() + start, data.begin() + end);

        for (i = 0; i < NUM_THREADS; i++) {  // 正则采样
            sample_data[id * NUM_THREADS + i] = data[start + i * step];
        }

        std::copy(data.begin() + start, data.begin() + end, backup.begin() + start);

    }
        std::sort(sample_data.begin(), sample_data.end());  // 采样排序

        //std::cout << "\nChoose Pivot: ";
        for (i = 0; i < NUM_THREADS - 1; i++) {  // 选取主元
            pivot[i] = sample_data[(i + 1) * NUM_THREADS];
             //std::cout << pivot[i] << " ";
        }
        // std::cout << std::endl;

         debug = 1;
    #pragma omp parallel private(i, id, start, end)  // 分割
        {
            id = omp_get_thread_num();
            start = (id)*n / NUM_THREADS;
            end = id == (NUM_THREADS - 1) ? n : (id + 1) * n / NUM_THREADS;
            for (i = 0; i < NUM_THREADS - 1; i++) {
                auto it = std::upper_bound(data.begin() + start, data.begin() + end, pivot[i]);
                divide[id][i] = std::distance(data.begin(), it);
            }
            divide[id][NUM_THREADS - 1] = end;
        }
         //    //debug = 1;
         //    std::cout << "\nAfter Local Sort:\n";
         //#pragma omp parallel private(i,id, start,end)//输出
         //    {
         //        id = omp_get_thread_num();
         //        start = (id)*n / NUM_THREADS;
         //        end = id == (NUM_THREADS - 1) ? n : (id + 1) * n / NUM_THREADS;
         //        std::cout << "Thread " << id << ": ";
         //        for (i = start; i < end; i++) {
         //            std::cout << data[i] << " ";
         //        }
         //        std::cout << std::endl;
         //    }

    #pragma omp parallel shared(divide, len) private(i, id, start, end)  // 计算长度
        {
            id = omp_get_thread_num();
            len[id][0] = 0;
            for (i = 0; i < NUM_THREADS; i++) {
                start = (i)*n / NUM_THREADS;
                len[id][0] += divide[i][id] - (id == 0 ? start : divide[i][id - 1]);
            }
        }

         for (i = 0; i < NUM_THREADS; i++) {  // 计算偏移
             offset[i] = i == 0 ? 0 : offset[i - 1] + len[i - 1][0];
         }
    //     debug = 1;



    #pragma omp parallel shared(offset,divide,backup,data) private(i, j, id, start, end)
        { 
            std::vector<std::vector<int>> arrays(NUM_THREADS);
            id = omp_get_thread_num();
            start = offset[id];
            end = id == NUM_THREADS - 1 ? n : offset[id + 1];
            for (i = 0; i < NUM_THREADS; i++) {
                int divide_start = id == 0 ? (i)*n / NUM_THREADS : divide[i][id - 1];
                int divide_end = divide[i][id];
                //arrays[i].assign(backup.begin() + divide_start, backup.begin() + divide_end);
                for (j = divide_start; j < divide_end && start < end; j++, start++) {
                    data[start] = backup[j];
                }
            }
            //while (arrays.size()>1) {
            //    std::vector<int> arr1 = arrays.back();
            //    arrays.pop_back();
            //    std::vector<int> arr2 = arrays.back();
            //    arrays.pop_back();
            //    mergeArrays(arr1, arr2);
            //    arrays.push_back(arr1);
            //}

            start = offset[id];

            //std::copy(arrays[0].begin(), arrays[0].end(), data.begin()+start);
            std::sort(data.begin() + start, data.begin() + end);

        }
     //    std::cout << "\nAfter Divide and Alloc:\n";
     //#pragma omp parallel private(i,id, start,end)//输出
     //    {
     //        id = omp_get_thread_num();
     //        start = offset[id];
     //        end = id == NUM_THREADS - 1 ? n : offset[id + 1];
     //        std::cout << "Thread " << id << ": ";
     //        for (i = start; i < end; i++) {
     //            std::cout << data[i] << " ";
     //        }
     //        std::cout << std::endl;
     //    }
     //    debug = 1;
}