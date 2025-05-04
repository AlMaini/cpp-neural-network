#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>   // For std::exp()
#include <cstdlib> // For rand()
#include <cassert> // For assert()

class Matrix {
private:
    // 2D vector to store matrix data
    std::vector<std::vector<double>> data;
    
    // Number of rows and columns
    size_t rows;
    size_t cols;

public:
    // Constructor: resizes the matrix to r rows and c columns, initializes all elements to initVal
    // If initVal is not provided, defaults to 0.0
    Matrix(size_t r, size_t c, double initVal = 0.0) : rows(r), cols(c) {
        data.resize(rows);
        for (size_t i = 0; i < rows; i++) {
            data[i].resize(cols, initVal);
        }
    }

    // Copy constructor: creates a new matrix as a copy of another matrix
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

    // Access element (with bounds checking)
    // Non-const version of element access; can be used to modify the element
    double& at(size_t i, size_t j) {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[i][j];
    }

    // Const version of element access
    // Used to access elements without modifying them
    const double& at(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[i][j];
    }

    // Operator overloading for element access
    // Allows using matrix(i, j) to access elements
    double& operator()(size_t i, size_t j) {
        return at(i, j);
    }

    const double& operator()(size_t i, size_t j) const {
        return at(i, j);
    }

    // Matrix addition
    Matrix operator+(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(i, j) = this->at(i, j) + rhs.at(i, j);
            }
        }
        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& rhs) const {
        if (rows != rhs.rows || cols != rhs.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(i, j) = this->at(i, j) - rhs.at(i, j);
            }
        }
        return result;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& rhs) const {
        if (cols != rhs.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(rows, rhs.cols, 0.0);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < rhs.cols; j++) {
                for (size_t k = 0; k < cols; k++) {
                    result(i, j) += this->at(i, k) * rhs.at(k, j);
                }
            }
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(i, j) = this->at(i, j) * scalar;
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(j, i) = this->at(i, j);
            }
        }
        return result;
    }

    // Sum of all elements
    double sum() const {
        double total = 0.0;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                total += data[i][j];
            }
        }
        return total;
    }

    // Randomize elements between min and max
    Matrix& randomize(double min = 0.0, double max = 1.0) {
        assert(max > min);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                data[i][j] = min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
            }
        }
        return *this;
    }

    // Apply sigmoid function to each element (inplace)
    Matrix& sigmoid() {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double x = data[i][j];
                data[i][j] = 1.0 / (1.0 + std::exp(-x));
            }
        }
        return *this;
    }

    // Square each element (inplace)
    Matrix& square() {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                data[i][j] *= data[i][j];
            }
        }
        return *this;
    }

    // Access to raw data
    std::vector<std::vector<double>>& getData() {
        return data;
    }

    const std::vector<std::vector<double>>& getData() const {
        return data;
    }


    // Get dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Print matrix
    void print() const {
        for (size_t i = 0; i < rows; i++) {
            std::cout << "[ ";
            for (size_t j = 0; j < cols; j++) {
                std::cout << data[i][j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }
    }

};

// Friend function for scalar * matrix
// this is because we want to allow the scalar to be on the left side of the multiplication i.e. scalar * matrix
// rather than matrix * scalar
Matrix operator*(double scalar, const Matrix& mat) {
    return mat * scalar;
}

#endif // MATRIX_HPP