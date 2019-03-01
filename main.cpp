#include <iostream>
#include <cmath>
#include "LinearAlgebra/Matrix.hpp"
#include "Net/NeuralNet.hpp"
int main() {
	using Matrix = LinearAlgebra::Matrix<double>;
	using NN = Net::NeuralNet<double>;
	using nnint = std::size_t;

	constexpr nnint SIZE = 5;
	constexpr nnint testcase_num = 1<<SIZE;
	constexpr nnint layer_count = 3;
    constexpr nnint output_size = SIZE + (SIZE > 1 ? 2 : 3);

	auto *input = new Matrix[testcase_num];
    auto *output = new Matrix[1<<output_size];

    for(nnint i = 0; i < testcase_num; i++) {
        input[i] = Matrix(SIZE, 1);
        output[i] = Matrix(output_size, 1);
        nnint tmp = i;
        for(nnint r = 0; r < SIZE; r++) {
            input[i](r, 0) = tmp & 1;
            tmp >>= 1;
        }
        if(i % 2) {
            tmp = i * 3 + 1;
        }
        else {
            tmp = i / 2;
        }
        for(nnint r = 0; r < SIZE; r++) {
            output[i](r, 0) = tmp & 1;
            tmp >>= 1;
        }
    }

    nnint layer_size[layer_count]{SIZE, SIZE + layer_count, output_size};

    NN nn(layer_count,
          layer_size,
          // inner_function
          [](Matrix matrix) -> Matrix {
              const nnint row = matrix.get_row();
              const nnint col = matrix.get_col();
              const double max = matrix.get_max();
              Matrix M(row, col);
              double sum = 0;
              for(nnint i = 0; i < row; i++) {
                  for(nnint j = 0; j < col; j++) {
                      sum += exp(matrix(i, j) - max);
                  }
              }
              for(nnint i = 0; i < row; i++) {
                  for(nnint j = 0; j < col; j++) {
                      M(i, j) = exp(matrix(i, j) - max) / sum;
                  }
              }
              return M;
          },
          // outer_function
          [](Matrix matrix) -> Matrix { return matrix; },
          // dinner_function
          [](Matrix matrix) -> Matrix {
                const nnint row = matrix.get_row();
                const nnint col = matrix.get_col();
                const double max = matrix.get_max();
                Matrix M(row, col);
                double sum = 0;
                for(nnint i = 0; i < row; i++) {
                    for(nnint j = 0; j < col; j++) {
                        sum += exp(matrix(i, j) - max);
                    }
                }
                for(nnint i = 0; i < row; i++) {
                    for(nnint j = 0; j < col; j++) {
                        M(i, j) = exp(matrix(i, j) - max) / sum;
                    }
                }
                return M;
          },
          // douter_function
          [](Matrix matrix) -> Matrix {
              const nnint row = matrix.get_row();
              const nnint col = matrix.get_col();
              Matrix M(row, col);
              for(nnint i = 0; i < row; i++) {
                  for(nnint j = 0; j < col; j++) {
                      M(i, j) = 1;
                  }
              }
              return M;
          },
          0.001
    );

    for(nnint i = 0; i < 30; i++) {
        nn.learn(testcase_num, input, output);
    }

    for(nnint i = 0; i < testcase_num; i++) {
        nn.print_case(std::cout, input[i], output[i]);
    }

    delete[] input;
    delete[] output;
}