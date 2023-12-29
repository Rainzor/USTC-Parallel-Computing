#include<mpi.h>
#include<iostream>
#include <algorithm>
#include<vector>
#include<limits.h>
#include<random>
#include <chrono>
using namespace std::chrono;
using std::vector;
using std::cout;
using std::endl;
int INF = INT_MAX;//无穷大
void print(vector<int>& a,int n) {
    int i;
    if (n < 50) {
        std::cout << "Array data:\n";
        for (i = 0; i < n; i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }
}

vector<int> PSRS(vector<int>& data);

int main(int argc, char* argv[]) {
	int nz;
    int pid, pNum;
    int size,n, n_extra;
    auto start = high_resolution_clock::now();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    vector<int> rawdata,local_data;
	// 创建一个随机数生成器对象，使用当前时间作为种子
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	// 创建一个均匀分布的整数分布器，范围为[0, 99]
	std::uniform_int_distribution<int> distribution(0, 99);
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &pNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);


    if (pid == 0) {
        std::cout << "Input the size of array: ";
        std::cin >> nz;
    }
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n_extra = nz % pNum == 0 ? 0 : pNum - nz % pNum;
    size = (nz+n_extra) / pNum;
    n = size * pNum;


    if (pid == 0) {
        rawdata.resize(n+n_extra);
    }
    local_data.resize(size);
    if (pid < pNum - 1) {
        for (int i = 0; i < size; i++) {
            local_data[i] = distribution(generator);
        }
    }
    else {
        for (int i = 0; i < size - n_extra; i++)
            local_data[i] = distribution(generator);
        if(n_extra>0)//处理不均匀数组
            for(int i=size-n_extra;i<size;i++)
			        local_data[i] = INF;
    }
    MPI_Gather(local_data.data(), size, MPI_INT, rawdata.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        cout << "\nThe random Data Generate......" << endl;
        print(rawdata, nz);
        std::cout << "\nstd::sort processing.......\n";
        start = high_resolution_clock::now();
        std::sort(rawdata.begin(), rawdata.begin()+ nz);
        end = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(end - start);

        print(rawdata, nz);
        std::cout << "std::sort Time Cost:" << duration.count() << " ms" << std::endl;
        std::cout << "\nPSRS processing by " << pNum << " process.......\n";
        start = high_resolution_clock::now();

    }
    MPI_Barrier(MPI_COMM_WORLD);
    vector<int> data = PSRS(local_data);

    if (pid == 0) {
        end = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(end - start);
        print(data,nz);
        std::cout << "PSRS Time Cost:" << duration.count() << " ms" << std::endl;
    }

    MPI_Finalize();
    
}


vector<int> PSRS(vector<int>& local_data) {
    /*----------------------------------------------------------------------------------*/
    //PSRS

    int pid, pNum;
    int i, j;
    int start, end;
    int size, n;
    MPI_Comm_size(MPI_COMM_WORLD, &pNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    size = local_data.size();
    n = size * pNum;
    start = pid * size;
    end = (pid + 1) * size;

    vector<int> data(n);
    vector<int> local_sample(pNum);
    vector<int> sample_data(pNum * pNum);
    vector<int> pivot(pNum);
    vector<int> local_divide(pNum);
    vector<int> divide(pNum * pNum);
    vector<int> len(pNum);
    vector<int> offset(pNum);


    //局部排序
    std::sort(local_data.begin(), local_data.end());

    //DEBUG
    //if (pid == 0)
    //    cout << "Loacl Sorted------------------------------" << endl;
    //cout << "pid" << pid << ": ";
    //for (auto& d : local_data)
    //    cout << d << " ";
    //cout << endl;
    
    // 正则采样
    int sample_step = size / pNum;
    for (i = 0; i < pNum; i++) {
        local_sample[i] = local_data[i * sample_step];
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    //分配局部排序结果广播到全局，在data中接受
    MPI_Allgather(local_data.data(), size, MPI_INT, data.data(), size, MPI_INT, MPI_COMM_WORLD);

    //Debug
    //if (pid == 1) {
    //    printf("\nData(pid%d):--------------------------------\n",pid);
    //    for (i = 0; i < n; i++)
    //        cout << data[i] << " ";
    //    cout << endl << "Data All Gather Pass-------------------------\n";
    //}

    //分配采样数据到0号进程sample_data中接收
    MPI_Gather(local_sample.data(), pNum, MPI_INT, sample_data.data(), pNum, MPI_INT, 0, MPI_COMM_WORLD);


    //DEBUG
    //if (pid == 0) {
    //    cout << "\nSample:--------------------------" << endl;
    //    for(i=0;i<pNum*pNum;i++)
    //        cout<<sample_data[i] << " ";
    //    cout << endl<<"Sample Pass-------------------------------"<<endl;
    //}

    //挑选主元
    if (pid == 0) {
        std::sort(sample_data.begin(), sample_data.end());
        for (i = 0; i < pNum - 1; i++) {
            pivot[i] = sample_data[(i + 1) * pNum];
        }
        pivot[pNum - 1] = INF;
    }
    //分配主元到全局
    MPI_Bcast(pivot.data(), pNum, MPI_INT, 0, MPI_COMM_WORLD);

    /*
    //Debug
    if (pid == 0) {
        printf("\nPivot(pid%d):--------------------------------\n", pid);
        for (i = 0; i < pNum; i++)
            cout << pivot[i] << " ";
        cout << "\nPivot Pass---------------------------------\n";
    }
    */
    
    //选取划分区间
    for (i = 0, j = start; i < pNum - 1; i++) {
        while (data[j] < pivot[i] && j < end)
            j++;
        local_divide[i] = j;
    }
    local_divide[pNum - 1] = end;

    //全局交换划分区间
    MPI_Allgather(local_divide.data(), pNum, MPI_INT, divide.data(), pNum, MPI_INT, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);

    local_data.clear();
    local_data.reserve(pNum);
    //数据再交换
    for (i = 0; i < pNum; i++) {
        int divide_start = pid == 0 ? (i)*n / pNum : divide[i * pNum + pid - 1];
        int divide_end = divide[i * pNum + pid];
        for (j = divide_start; j < divide_end; j++)
            local_data.push_back(data[j]);
    }

    //Debug
    //if (pid == 0)
    //    printf("\nAfter divided:--------------------------------\n");
    //cout << "pid" << pid << ": ";
    //for (auto& d : local_data)
    //    cout << d << " ";
    //cout << endl;

    //局部再排序
    std::sort(local_data.begin(), local_data.end());

    //Debug
    //if (pid == 0)
    //    cout << "\nAfter loacl sorted------------------------------" << endl;
    //cout << "pid" << pid << ": ";
    //for (auto& d : local_data)
    //    cout << d << " ";
    //cout << endl;

    
    size = local_data.size();
    MPI_Gather(&size, 1, MPI_INT, len.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    //根进程收集数据
    if (pid == 0) {
        offset[0] = 0;
        for (i = 1; i < pNum; i++) {
            offset[i] = offset[i - 1] + len[i - 1];
        }
    }

    // 使用MPI_Gatherv收集不同大小的数组
    MPI_Gatherv(local_data.data(), local_data.size(), MPI_INT, data.data(), len.data(), offset.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return data;
    //PSRS
    /*-------------------------------------------------------------------------------------*/
}