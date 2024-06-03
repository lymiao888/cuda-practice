cmake_minimum_required(VERSION 3.12)
project(gemm LANGUAGES CXX CUDA)

add_executable(gemm gemm.cu)
target_include_directories(gemm PUBLIC ./include)
