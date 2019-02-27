#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>

template<typename T>
class Matrix
{
	// pointer that stores matrix
	T *mat;
	// size of matrix
	std::size_t row, col;

public:
	// constructors
	Matrix(std::size_t row, std::size_t col)
		: row(row), col(col), mat(new T[row * col]()) {
		assert(row != 0 && col != 0);
	}
	Matrix(std::size_t row, std::size_t col, T *arr)
		: row(row), col(col), mat(new T[row * col]) {
		assert(row != 0 && col != 0);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			mat[i] = arr[i];
		}
	}

	// move constructor
	Matrix(Matrix const &matrix)
		: row(matrix.row), col(matrix.col), mat(new T[row * col]) {
		std::copy(matrix.mat, matrix.mat + row * col, mat);
	}
	// copy constructor
	Matrix(Matrix &&matrix) noexcept
		: row(matrix.row), col(matrix.col), mat(matrix.mat) {
		matrix.mat = nullptr;
	}
	// copy assignment
	Matrix &operator=(Matrix const &matrix) {
		if (&matrix != this) {
			row = matrix.row, col = matrix.col;
			delete[] mat;
			mat = new T[row * col];
			std::copy(matrix.mat, matrix.mat + row * col, mat);
		}
		return *this;
	}
	// move assignment
	Matrix &operator=(Matrix &&matrix) noexcept {
		row = matrix.row, col = matrix.col;
		std::swap(mat, matrix.mat);
		delete[] matrix.mat;
		return *this;
	}

protected:
	// elementary functions made virtual
	virtual Matrix add(Matrix const &matrix) const {
		assert(row == matrix.row && col == matrix.col);
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] + matrix.mat[i];
		}
		return M;
	}
	virtual Matrix add(T const &scalar) const {
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] + scalar;
		}
		return M;
	}
	virtual Matrix subtract(Matrix const &matrix) const {
		assert(row == matrix.row && col == matrix.col);
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] - matrix.mat[i];
		}
		return M;
	}
	virtual Matrix subtract(T const &scalar) const {
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] - scalar;
		}
		return M;
	}
	virtual Matrix multiply(Matrix const &matrix) const {
		assert(col == matrix.row);
		Matrix<T> M(row, matrix.col);
		for (std::size_t i = 0; i < row; i++) {
			for (std::size_t k = 0; k < matrix.col; k++) {
				for (std::size_t j = 0; j < col; j++) {
					M.mat[i * matrix.col + k] += mat[i * col + j] * matrix.mat[j * matrix.col + k];
				}
			}
		}
		return M;
	}
	virtual Matrix multiply(T const &scalar) const {
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] * scalar;
		}
		return M;
	}
	virtual Matrix elementwise_multiply(Matrix const &matrix) const {
		assert(row == matrix.row && col == matrix.col);
		Matrix<T> M(row, col);
		std::size_t size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] * matrix.mat[i];
		}
		return M;
	}
	virtual Matrix divide(T const &scalar) const {
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] / scalar;
		}
		return M;
	}
	virtual Matrix elementwise_divide(Matrix const &matrix) const {
		assert(row == matrix.row && col == matrix.col);
		Matrix<T> M(row, col);
		std::size_t size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = mat[i] / matrix.mat[i];
		}
		return M;
	}

public:
	// operators
	Matrix operator-() const {
		Matrix<T> M(row, col);
		std::size_t const size = row * col;
		for (std::size_t i = 0; i < size; i++) {
			M.mat[i] = -mat[i];
		}
		return M;
	}

	friend Matrix<T> operator+(Matrix<T> const &A, Matrix<T> const &B) {
		return A.add(B);
	}
	friend Matrix<T> operator+(Matrix<T> const &A, T const &b) {
		return A.add(b);
	}
	friend Matrix<T> operator+(T const &a, Matrix const &B) {
		return B.add(a);
	}
	friend Matrix<T> operator-(Matrix<T> const &A, Matrix<T> const &B) {
		return A.subtract(B);
	}
	friend Matrix<T> operator-(Matrix<T> const &A, T const &b) {
		return A.subtract(b);
	}
	friend Matrix<T> operator-(T const &a, Matrix<T> const &B) {
		return -(B.subtract(a));
	}
	friend Matrix<T> operator*(Matrix<T> const &A, Matrix<T> const &B) {
		return A.multiply(B);
	}
	friend Matrix<T> operator*(Matrix<T> const &A, T const &b) {
		return A.multiply(b);
	}
	friend Matrix<T> operator*(T const &a, Matrix<T> const &B) {
		return B.multiply(a);
	}
	friend Matrix<T> operator/(Matrix<T> const &A, T const &b) {
		return A.divide(b);
	}

	// out stream operator
	friend std::ostream &operator<<(std::ostream &os, Matrix<T> const &matrix) {
		for (std::size_t i = 0; i < matrix.row; i++) {
			for (std::size_t j = 0; j < matrix.col; j++) {
				os << matrix.mat[i * matrix.col + j] << ' ';
			}
			os << '\n';
		}
		return os;
	}

	// destructor
	virtual ~Matrix() {
		delete[] mat;
	}
};
