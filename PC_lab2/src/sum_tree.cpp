#include<mpi.h>
#include<iostream>
#include<random>
#include<chrono>
using namespace std::chrono;
int main() {
	int pid, pNum;
	int data;
	int recvdata;
	MPI_Status status;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &pNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	// 创建一个随机数生成器对象，使用当前时间作为种子
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	// 创建一个均匀分布的整数分布器，范围为[0, 99]
	std::uniform_int_distribution<int> distribution(0, 99);
	data = distribution(generator);
	std::cout << "pid: " << pid << ", local data: " << data << ", ";
	//gather
	for (int i = 1,flag=1; i < pNum&&flag; i = i << 1) {//i是每层相邻结点差值
		int tag = i;
		int diff = pid & tag;//按位与得到父节点
		int dest = pid ^ tag;//按位异或得到相邻传输结点

		if (diff) {
			MPI_Send(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
			flag = 0;
		}
		else {
			MPI_Recv(&recvdata, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &status);
			data += recvdata;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	//scatter
	for (int i = pNum; i >= 2; i = i >> 1) {
		int tag = i;
		int dest  = pid ^ (i >> 1);//按位异或得到相邻传输结点 
		if (pid % i == 0) {
			MPI_Send(&data,1, MPI_INT, dest, tag, MPI_COMM_WORLD);
		}
		else if (pid % (i >> 1) == 0) {//如果当前是没有传到的
			MPI_Recv(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &status);
		}
	}
	std::cout << "sum data: " << data;
	MPI_Finalize();
	return 0;
}