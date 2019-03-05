cmake_minimum_required(VERSION 3.13)
project(BNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(BNN main.cpp LinearAlgebra/Matrix.hpp Net/NeuralNet.hpp Calculation/Operand/Scalar.hpp Calculation/Operand/Constant.hpp Calculation/Operand/Variable.hpp Calculation/SubClass/get_given_amount_of_parameters_for_constructor_class.hpp Calculation/Operand/OperandBase.hpp Calculation/Operator/OperatorBase.hpp)