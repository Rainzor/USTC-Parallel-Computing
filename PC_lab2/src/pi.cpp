#include <stdio.h>
#include <mpi.h>
#include <math.h>
long n, //number of slices 
i; // slice counter 
double sum, // running sum 
pi, // approximate value of pi 
mypi,
x, // independent var. 
h; // base of slice 
int group_size, my_rank;

int main(int argc, char* argv[])
{
	int group_size, my_rank;
	MPI_Status status;
	MPI_Init(&argc, &argv);//从这里开始fork出一系列进程
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &group_size);
	n = 2000;
	// Broadcast n to all other nodes 
	MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);//从0号进程处向所有进程广播，保证一致性，但它不是必须的
	h = 1.0 / (double)n;
	sum = 0.0;
	for (i = my_rank + 1; i <= n; i += group_size) {
		x = h * (i - 0.5);
		sum = sum + 4.0 / (1.0 + x * x);
	}
	mypi = h * sum;
	//Global sum 
	MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);//归约到0号线程
	if (my_rank == 0) { // Node 0 handles output 
		printf("pi is approximately : %.16lf\n", pi);
	}
	MPI_Finalize();
}