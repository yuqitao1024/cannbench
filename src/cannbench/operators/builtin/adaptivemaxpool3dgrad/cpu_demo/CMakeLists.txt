cmake_minimum_required(VERSION 3.16)

project(adaptivemaxpool3dgrad_cpu_demo LANGUAGES CXX)

add_executable(adaptivemaxpool3dgrad_cpu_demo main.cpp)
target_compile_features(adaptivemaxpool3dgrad_cpu_demo PRIVATE cxx_std_17)

