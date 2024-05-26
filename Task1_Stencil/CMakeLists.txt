cmake_minimum_required(VERSION 3.12)
project(Stencil LANGUAGES CXX CUDA)

add_executable(stencil stencil.cu)
target_include_directories(stencil PUBLIC ./include)