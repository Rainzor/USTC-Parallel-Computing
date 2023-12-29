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

include_directories(src)

find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(cuda_example src/kernel.cu)
add_executable(matrix_multiplication src/MatrixMulti.cu)
add_executable(vector_addition src/VectorAdd.cu)
# 链接CUDA库
target_link_libraries(cuda_example PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(matrix_multiplication PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(vector_addition PRIVATE ${CUDA_LIBRARIES})



