cmake_minimum_required(VERSION 3.13)
project(BNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(BNN main.cpp LinearAlgebra/Matrix.hpp Net/NeuralNet.hpp Temporary/Operand/Scalar.hpp Temporary/Operand/Constant.hpp Temporary/Operand/Variable.hpp Temporary/SubClass/get_given_amount_of_parameters_for_constructor_class.hpp Temporary/Operand/OperandBase.hpp Temporary/Operator/OperatorBase.hpp Ops/Operators/Operator.hpp)