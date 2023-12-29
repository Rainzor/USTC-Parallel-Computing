cmake_minimum_required(VERSION 3.10)

project(mpi)

# 将生成的 exe, dll, lib 等放置在 bin, lib 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/lib")


# 寻找MPI库
find_package(MPI REQUIRED)

# 将MPI_INCLUDE_PATH添加到包含目录
include_directories(${MPI_INCLUDE_PATH})

# 创建一个名为my_mpi_program的可执行文件，将main.cpp编译为可执行文件
add_executable(${PROJECT_NAME}_TEST src/test.cpp)
add_executable(${PROJECT_NAME}_pi src/pi.cpp)
add_executable(${PROJECT_NAME}_PSRS src/PSRS.cpp)
add_executable(${PROJECT_NAME}_SumButterfly src/sum_butterfly.cpp)
add_executable(${PROJECT_NAME}_SumTree src/sum_tree.cpp)

# 将MPI库链接到可执行文件
target_link_libraries(${PROJECT_NAME}_TEST ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_pi ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_PSRS ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_SumButterfly ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_SumTree ${MPI_LIBRARIES})



