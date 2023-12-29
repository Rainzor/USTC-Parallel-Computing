cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(OPEMPM VERSION 1.0)
message(STATUS "[Project] ${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_executable(${PROJECT_NAME} "PSRS.cpp")