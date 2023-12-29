# CUDA 编程

## 1 配置环境

配置Windows11系统下的CUDA环境主要有以下步骤

1. 下载CUDA工具包：访问[NVIDIA CUDA Toolkit下载页面](https://developer.nvidia.com/cuda-downloads)，选择相应的Windows版本，然后下载CUDA Toolkit安装程序。
2. 安装CUDA工具包：运行刚刚下载的安装程序。安装过程中，选择"Express"（快速安装）或"Custom"（自定义安装）。"Express"选项将在默认位置安装所有组件，而"Custom"选项允许您选择要安装的组件以及安装位置。对于大多数用户，建议使用"Express"选项。
3. 环境变量配置：安装完成后，CUDA Toolkit和cuDNN库应该已经被添加到系统环境变量中。如果没有，您需要手动添加以下环境变量：
   - 将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\bin`添加到`Path`环境变量中（`<version>`为您安装的CUDA版本号）。
   - 将`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\libnvvp`添加到`Path`环境变量中。
4. 安装cuDNN库（可选）：cuDNN库是一个用于深度神经网络的GPU加速库，可以与CUDA一起使用。要安装cuDNN，请按照以下步骤操作：
   - 注册并登录[NVIDIA Developer网站](https://developer.nvidia.com/developer-program/signup)，然后访问[cuDNN下载页面](https://developer.nvidia.com/cudnn)。
   - 选择适用于您安装的CUDA版本的cuDNN版本。
   - 下载适用于Windows的cuDNN库。
   - 将下载的cuDNN文件解压缩，并将其中的`bin`、`include`和`lib`文件夹的内容复制到CUDA Toolkit的相应文件夹中（例如，`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>`）。
5. 验证CUDA安装：要验证CUDA环境是否已正确安装和配置，请使用命令提示符运行以下命令：

```sh
nvcc --version
```

如果看到输出中显示了安装的CUDA版本信息，那么您已成功安装和配置CUDA环境。

## 2 代码工程创建

### 2.1 Visual Studio 2022

1. 创建新项目： 打开 Visual Studio，点击 "文件" > "新建" > "项目"，或者在开始页面点击 "创建新项目"。
2. 选择项目类型： 在 "新建项目" 窗口中，选择 "已安装" > "Visual C++" > "NVIDIA" > "CUDA"，然后选择 "CUDA Runtime" 项目模板。请注意，如果您没有看到这个选项，请确保已经正确安装了 CUDA Toolkit。
3. 设置项目名称和位置： 为项目设置名称和保存位置，然后点击 "创建"。
4. 编写 CUDA 代码： Visual Studio 将为您生成一个包含默认 "kernel.cu" 文件的新 CUDA 项目。在这个文件中，您可以看到一个简单的 CUDA 核函数示例。您可以修改这个文件，或者添加新的 ".cu" 文件来编写您自己的 CUDA 代码。
5. 编译和运行项目： 在 Visual Studio 工具栏上，选择适当的解决方案平台（例如，x64）和配置（例如，Debug 或 Release）。然后点击 "生成" > "生成解决方案" 或按下 F7 键来编译项目。编译完成后，按 F5 或点击 "调试" > "开始调试" 来运行项目。

### 2.2 Cmake

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # 更新 CMake 版本要求
project(cuda_example LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 NEW) # 设置新的 CMP0104 策略

set(CMAKE_CUDA_ARCHITECTURES 75) # 设置 CUDA 架构（这里是 75，根据 GPU 调整）

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)#配置Include库

# 添加可执行文件
add_executable(cuda_example src/kernel.cu)#根据需要修改目标文件地址

# 链接CUDA库
target_link_libraries(cuda_example PRIVATE ${CUDA_LIBRARIES})
```

## 3 Linux系统配置

> WSL2 Ubuntu20.02

### 安装 NVIDIA Driver

首先检查系统显卡驱动，宿主win11安装英伟达驱动NVIDIA Driver

```bash
nvidia-smi
```

传统上，安装 NVIDIA Driver 和 CUDA Toolkit 的步骤是分开的，但实际上我们可以直接安装 CUDA Toolkit，系统将自动安装与其版本匹配的 NVIDIA Driver。

### 安装 CUDA Toolkit

#### 预准备

在安装 CUDA Toolkit 前，要确保系统安装了 gcc 和 make。如果希望使用 C++ 进行 CUDA 编程，需要安装 g++。如果想要运行 CUDA 例程序，需要安装相应的依赖库。

```bash
sudo apt update # 更新 apt
sudo apt install gcc g++ make # 安装 gcc g++ make
sudo apt install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev # 安装依赖库
```

#### 下载 Toolkit

在 [CUDA Toolkit](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cuda-toolkit) 的下载页面选择系统版本和安装方式

![](/C:/Users/Lenovo/AppData/Roaming/Typora/typora-user-images/image-20230517125318479.png)

CUDA Toolkit 下载

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.debsudo 
cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

> **坑点：**可能由于wget无法下载，那么就手动点到链接里，在主机中下载完，在把文件移到wsl环境中

#### 设置环境变量

```bash
sudo vim ~/.bashrc
sudo vim ~/.zshrc
```

文件最后追加

```bash
export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

文件生效

```bash
source ~/.bashrc
source ~/.zshrc
```

即可完成 CUDA 的配置。

**注意事项：**

1. 环境变量 PATH 设置可执行程序的搜索路径，LD_LIBRARY_PATH 设置动态链接库的搜索路径
2. CUDA, cuRAND 等动态库均位于 /usr/local/cuda-10.1/lib64 路径中。在 CUDA 10.0 以前，cuBLAS 也位于此路径下，但在 CUDA 10.1 中，cuBLAS 被[迁移](https://link.zhihu.com/?target=https%3A//devtalk.nvidia.com/default/topic/1047981/cuda-setup-and-installation/cublas-for-10-1-is-missing/)到了 /usr/lib/x86_64-linux-gnu 中。可以通过运行

```bash
sudo find / -iname libcublas*
```

  来寻找 cuBLAS 动态库的路径。

3. 使用 Anaconda 安装的 CUDA Toolkit 不位于 lib64 路径中，也不会产生冲突。

#### 验证cuda是否安装成功

```bash
cat /usr/local/cuda/version.json
```

![image-20230517130038910](/C:/Users/Lenovo/AppData/Roaming/Typora/typora-user-images/image-20230517130038910.png)

### 安装cudnn

若是toolkit用wsl-Ubuntu版本，默认已经安装了

**验证cudnn是否安装**

```bash
nvcc -V
```

### 测试样例程序

可以在路径

```text
/usr/local/cuda-10.1/extras/demo_suite
```

路径下找到一些样例程序。deviceQuery 将输出 CUDA 的相关信息：

```txt
 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce GTX 1650"
  CUDA Driver Version / Runtime Version          12.1 / 12.1
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 4096 MBytes (4294639616 bytes)
  (14) Multiprocessors, ( 64) CUDA Cores/MP:     896 CUDA Cores
  GPU Max Clock rate:                            1515 MHz (1.51 GHz)
  Memory Clock rate:                             6001 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.1, CUDA Runtime Version = 12.1, NumDevs = 1, Device0 = NVIDIA GeForce GTX 1650
Result = PASS
```

CUDA 的各种特性：纹理内存 (texture memory)、常量内存 (constant memory)、共享内存 (shared  memory)、块 (block)、线程 (thread)、统一寻址 (unified addressing)  都包含在以上信息中。了解这些特性是进行 CUDA C/C++ 编程的基础

### 配置编译器

#### 命令

nvcc 是 CUDA C/C++ 的编译器，可以直接编译包含 C++ 语法的 (.cu) 源文件，语法和 gcc 类似。nvcc 的路径位于：

```bash
/usr/local/cuda-10.1/bin
```

在命令行中输入

```bash
nvcc --version
```

可查看 CUDA C/C++ 编译器 nvcc 的版本，本机结果如下

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

使用 nvcc 编译包含了 CUDA Library 的源文件时，需要在 nvcc 命令中添加相应的 flag。例如，cuRAND 的 flag 为 -lcurand，cuBLAS 的 flag 为 -lcublas。如果不希望每次编译时都加上这些动态库的话，可以在 .bashrc/.zshrc 中写入

```bash
alias nvc="nvcc -std=c++11 -lcurand -lcublas"
```

然后使用 nvc 来进行 C/C++ 文件的编译。

#### 使用 nvcc 进行 CUDA C/C++ 编程

为了体验 GPU 编程，测试一个简单的 CUDA C++ 程序：*两个整型向量的加法*

```cpp
#include <iostream>
#include <cuda.h>

using namespace std;

__global__ void add(int *a, const int *b){
    int i = blockIdx.x;
    a[i] += b[i];
}

int main(){
    const int N = 10; // number of elements
    int *a, *b, *temp, i;
    // malloc HOST memory for temp
    temp = new int [N];
    // malloc DEVICE memory for a, b
    cudaMalloc(&a, N*sizeof(int));
    cudaMalloc(&b, N*sizeof(int));
    // set a's values: a[i] = i
    for(i=0;i<N;i++) temp[i] = i;
    cudaMemcpy(a, temp, N*sizeof(int), cudaMemcpyHostToDevice);
    // set b's values: b[i] = 2*i
    for(i=0;i<N;i++) temp[i] = 2*i;
    cudaMemcpy(b, temp, N*sizeof(int), cudaMemcpyHostToDevice);
    // calculate a[i] += b[i] in GPU
    add<<<N,1>>>(a, b);
    // show a's values
    cudaMemcpy(temp, a, N*sizeof(int), cudaMemcpyDeviceToHost);
    for(i=0;i<N;i++){
        cout << temp[i] << endl;
    }
    // free HOST & DEVICE memory
    delete [] temp;
    cudaFree(a);
    cudaFree(b);
}
```

上述代码使用的 CUDA 函数包括：

- cudaMalloc:  为指针申请 GPU 中的内存
- cudaMemcpy:  CPU 和 GPU 之间的内存拷贝
- cudaFree: 释放指针指向的 GPU 内存

将上述代码保存为文件 test.cu，并在命令行里输入

```bash
nvcc -o test test.cu
```

即可生成名为 test 的可执行文件。打开这个文件，屏幕上将输出

```bash
0
3
6
9
12
15
18
21
24
27
```

> **注意：**上述代码仅为测试 CUDA  C/C++ 程序之用，不具有运行效率上的参考性。
