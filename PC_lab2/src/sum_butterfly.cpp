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

	int dest;
	for (int i = 1; i < pNum; i = i << 1) {
		//按位异或找邻居，比如i=2, 那么0的邻居是2，用到了异或的性质：a^b^b=a；
		//其实也可以通过pid % (i >> 1) == 0来找
		dest = pid ^ i;
		MPI_Send(&data, 1, MPI_INT, dest, i, MPI_COMM_WORLD);
		MPI_Recv(&recvdata, 1, MPI_INT, dest, i, MPI_COMM_WORLD, &status);
		data += recvdata;
	}
	std::cout << "sum data: " << data;
	MPI_Finalize();
	return 0;
}