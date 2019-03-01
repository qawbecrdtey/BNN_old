cmake_minimum_required(VERSION 3.13)
project(BNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(BNN main.cpp LinearAlgebra/Matrix.hpp Net/NeuralNet.hpp)