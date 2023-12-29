#include <stdio.h>
#include <chrono>
#include <iostream>
using namespace std::chrono;

static long num_steps = 1000000000;  // 越大值越精确
double step;
int main() {
	int i;
	double x, pi, sum = 0.0;
	step = 1.0 / (double)num_steps;
	auto start = high_resolution_clock::now();

	for (i = 1; i <= num_steps; i++) {
		x = (i - 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}
	pi = step * sum;
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    std::cout << std::endl
              << "Time Cost:" << duration.count() << " ms" << std::endl;
	printf("%lf\n", pi);
	return 0;
}
