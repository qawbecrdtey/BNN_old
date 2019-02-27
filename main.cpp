#include <iostream>
#include "LinearAlgebra/Matrix.hpp"

int main() {
	double a[6] = { 1,2,3,4,5,6 };
	Matrix<double> const matrix1(3, 2, a);
	Matrix<double> const matrix2(2, 3, a);
	std::cout << matrix1 * matrix2;
}