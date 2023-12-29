## OpenMP环境配置

> Windows11

简介：

OpenMP是一个共享存储并行系统上的应用程序接口。它规范了一系列的编译制导、运行库例程和环境变量。

它提供了C/C++和FORTRAN等的应用编程接口，已经应用到UNIX、Windows NT等多种平台上。

OpenMP使用FORK-JOIN并行执行模型。所有的OpenMP程序开始于一个单独的主线程（Master Thread）。主线程会一直串行地执行，直到遇到第一个并行域（Parallel Region）才开始并行执行。

FORK：主线程创建一队并行的线程，然后，并行域中的代码在不同的线程队中并行执行；②JOIN：当诸线程在并行域中执行完之后，它们或被同步或被中断，最后只有主线程在执行。

**可以理解为线程级并行**

每个线程有共享内存与独立内存 

## 编译和运行：

编译：

```sh
gcc a.c –fopenmp –o a
```

运行：

```sh
./a
```

### CMake使用

在Cmake文件中添加以下内容即可使用 `OpenMP`

```cmake
find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
```

