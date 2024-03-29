#ifndef BNN_LinearAlgebra_Matrix_hpp
#define BNN_LinearAlgebra_Matrix_hpp


#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include "../Ops/Operators/Operator.hpp"

namespace LinearAlgebra {

    template<typename T>
    class Matrix {
        // pointer that stores matrix
        T *mat;
        // size of matrix
        std::size_t row, col;

    public:
        // constructors
        Matrix() : row(0), col(0), mat(nullptr) {}

        Matrix(std::size_t row, std::size_t col)
                : row(row), col(col), mat(new T[row * col]) {}

        Matrix(std::size_t row, std::size_t col, std::random_device &rd, T mean = 0, T stdv = 1)
                : row(row), col(col), mat(new T[row * col]) {
            assert(row != 0 && col != 0);
            std::default_random_engine e1(rd());
            std::normal_distribution<T> normal_dist(mean, stdv);
            for(std::size_t i = 0; i < row * col; i++) {
                mat[i] = static_cast<T>(normal_dist(e1));
            }
        }

        explicit Matrix(std::size_t row, std::size_t col, T constant)
                : row(row), col(col), mat(new T[row * col]) {
            assert(row != 0 && col != 0);
            for(std::size_t i = 0; i < row * col; i++) {
                mat[i] = constant;
            }
        }

        explicit Matrix(std::size_t row, std::size_t col, T *arr)
                : row(row), col(col), mat(new T[row * col]) {
            assert(row != 0 && col != 0);
            std::size_t const size = row * col;
            for (std::size_t i = 0; i < size; i++) {
                mat[i] = arr[i];
            }
        }

        // TODO : PLEASE USE STATIC CONSTRUCTORS. Below is the example.
        static Matrix<T> Constant(std::size_t row, std::size_t col, T constant) {
            Matrix<T> M(row, col);
            std::fill(M.mat, M.mat + row * col, constant);
            return M;
        }
        static Matrix<T> Zeros(std::size_t row, std::size_t col) {
            return Constant(row, col, static_cast<T>(0));
        }
        static Matrix<T> Ones(std::size_t row, std::size_t col) {
            return Constant(row, col, static_cast<T>(1));
        }

        // copy constructor
        Matrix(Matrix const &matrix)
                : row(matrix.row), col(matrix.col), mat(new T[matrix.row * matrix.col]) {
            std::copy(matrix.mat, matrix.mat + matrix.row * matrix.col, mat);
        }

        // move constructor
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
            return *this;
        }

    public:
        // compute function
        template<typename FuncOut, std::size_t size, typename ...Ts>
        friend FuncOut compute(Operators::OperatorBase<FuncOut, size> const &op, Ts &&...Is) {
            return op(std::forward<Ts>(Is)...);
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
            //Matrix<T> M(row, matrix.col, static_cast<T>(0));
            auto M = Matrix<T>::Zeros(row, matrix.col);
            for (std::size_t i = 0; i < row; i++) {
                for (std::size_t j = 0; j < col; j++) {
                    for (std::size_t k = 0; k < matrix.col; k++) {
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

        Matrix &operator+=(Matrix<T> &&A) {
            return *this = *this + A;
        }

        friend Matrix<T> operator+(Matrix<T> const &A, T const &b) {
            return A.add(b);
        }

        friend Matrix<T> operator+(T const &a, Matrix const &B) {
            return B.add(a);
        }

        Matrix &operator+=(T &&a) {
            return *this = *this + a;
        }

        friend Matrix<T> operator-(Matrix<T> const &A, Matrix<T> const &B) {
            return A.subtract(B);
        }

        Matrix &operator-=(Matrix<T> &&A) {
            return *this = *this - A;
        }

        friend Matrix<T> operator-(Matrix<T> const &A, T const &b) {
            return A.subtract(b);
        }

        Matrix &operator-=(T &&a) {
            return *this = *this - a;
        }

        friend Matrix<T> operator-(T const &a, Matrix<T> const &B) {
            return -(B.subtract(a));
        }

        friend Matrix<T> operator*(Matrix<T> const &A, Matrix<T> const &B) {
            return A.multiply(B);
        }

        Matrix &operator*=(Matrix &&A) {
            return *this = *this * A;
        }

        friend Matrix<T> operator*(Matrix<T> const &A, T const &b) {
            return A.multiply(b);
        }

        friend Matrix<T> operator*(T const &a, Matrix<T> const &B) {
            return B.multiply(a);
        }

        Matrix &operator*=(T &&a) {
            return *this = *this * a;
        }

        friend Matrix<T> operator/(Matrix<T> const &A, T const &b) {
            return A.divide(b);
        }

        Matrix operator/=(T &&a) {
            return *this = *this / a;
        }

        // basic operations only for matrix
        friend Matrix<T> elementwise_multiplied(Matrix<T> const &A, Matrix<T> const &B) {
            return A.elementwise_multiply(B);
        }

        virtual Matrix &elementwise_multiply(Matrix<T> &&A) {
            return *this = elementwise_multiplied(*this, A);
        }

        virtual Matrix transposed() const {
            Matrix M(col, row);
            for(std::size_t j = 0; j < col; j++) {
                for(std::size_t i = 0; i < row; i++) {
                    M.mat[j * row + i] = mat[i * col + j];
                }
            }
            return M;
        }
        [[deprecated]] virtual Matrix &transpose() {
            return *this = this->transposed();
        }

        // get max and min element
        T const get_max() const {
            auto max = mat[0];
            for(std::size_t i = 1; i < row * col; i++) {
                if(max < mat[i]) {
                    max = mat[i];
                }
            }
            return max;
        }
        T const get_min() const {
            auto min = mat[0];
            for(std::size_t i = 1; i < row * col; i++) {
                if(min < mat[i]) {
                    min = mat;
                }
            }
            return min;
        }

        // index
        T constexpr &operator()(std::size_t r, std::size_t c) const {
            return mat[r * col + c];
        }
        T &operator()(std::size_t r, std::size_t c) {
            return mat[r * col + c];
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

        // get row
        constexpr std::size_t get_row() const {
            return row;
        }
        //get col
        constexpr std::size_t get_col() const {
            return col;
        }

        // destructor
        virtual ~Matrix() {
            delete[] mat;
        }
    };

}
#endif