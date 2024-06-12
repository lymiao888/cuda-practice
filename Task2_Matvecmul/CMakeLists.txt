cmake_minimum_required(VERSION 3.12)
project(main LANGUAGES CXX CUDA)
set(CMAKE_CXX_COMPILER clang)

add_executable(main matvecmul.cu)
target_include_directories(main PUBLIC ./include)
