/**
 * @file mann.h
 * @brief Header file for the Mann namespace and Matrix class for neural network operations.
 * @author Jayansh Devgan, Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 *
 * This file defines the Matrix class within the Mann namespace, providing
 * functionality for matrix operations used in neural network computations.
 */

#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <future>
#include <stdexcept>
#include <time.h>
#include <iostream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <chrono>

/**
 * @brief Macro for constant unsigned short type.
 */
#define MU_SHORTC const unsigned short

/**
 * @brief Macro for unsigned short type.
 */
#define MU_SHORT unsigned short

/**
 * @namespace Mann
 * @brief Namespace for matrix-related classes and utilities used in neural networks.
 */
namespace Mann
{
    /**
     * @class Matrix
     * @brief A class representing a matrix for neural network computations.
     *
     * This class provides functionality for creating and manipulating matrices,
     * including arithmetic operations, randomization, and output formatting.
     */
    class Matrix
    {
    private:
        int m_rows; ///< Number of rows in the matrix.
        int m_cols; ///< Number of columns in the matrix.
        std::vector<float> m_data; ///< 1D vector storing matrix data in row-major order.
    public:
        /**
         * @brief Constructs a matrix with specified dimensions.
         * @param rows The number of rows in the matrix.
         * @param cols The number of columns in the matrix.
         */
        Matrix(int rows, int cols);
        
        /**
         * @brief Gets the number of rows in the matrix.
         * @return The number of rows.
         */
        int rows() const;

        /**
         * @brief Gets the number of columns in the matrix.
         * @return The number of columns.
         */
        int cols() const;

        const std::vector<float> data() const;

        /**
         * @brief Accesses a row of the matrix for modification.
         * @param index The row index.
         * @return A reference to the row as a vector of floats.
         */
        float& operator[](int index);

        /**
         * @brief Accesses a row of the matrix for read-only access.
         * @param index The row index.
         * @return A const reference to the row as a vector of floats.
         */
        const float& operator[](int index) const;

        /**
         * @brief Adds two matrices element-wise.
         * @param other The matrix to add.
         * @return A new matrix containing the element-wise sum.
         */
        Matrix operator+(const Matrix& other) const;

        /**
         * @brief Subtracts a matrix from this matrix element-wise.
         * @param other The matrix to subtract.
         * @return A new matrix containing the element-wise difference.
         */
        Matrix operator-(const Matrix& other) const;

        /**
         * @brief Adds a scalar to each element of the matrix.
         * @param scaler The scalar value to add.
         * @return A new matrix with the scalar added to each element.
         */
        Matrix operator+(float scaler) const;

        /**
         * @brief Subtracts a scalar from each element of the matrix.
         * @param scaler The scalar value to subtract.
         * @return A new matrix with the scalar subtracted from each element.
         */
        Matrix operator-(float scaler) const;

        /**
         * @brief Performs matrix multiplication with another matrix.
         * @param other The matrix to multiply with.
         * @return A new matrix containing the result of matrix multiplication.
         */
        Matrix operator*(const Matrix& other) const;

        /**
         * @brief Multiplies each element of the matrix by a scalar.
         * @param scalar The scalar value to multiply by.
         * @return A new matrix with each element scaled.
         */
        Matrix operator*(double scalar) const;

        /**
         * @brief Performs element-wise multiplication (Hadamard product) with another matrix.
         * @param other The matrix to multiply element-wise.
         * @return A new matrix containing the element-wise product.
         */
        Matrix operator^(const Matrix& other) const;

        /**
         * @brief Divides each element of the matrix by a scalar.
         * @param scalar The scalar value to divide by.
         * @return A new matrix with each element divided by the scalar.
         */
        Matrix operator/(double scalar) const;

        /**
         * @brief Assigns values to the matrix using an initializer list.
         * @param init The initializer list of vectors containing matrix data.
         * @return A reference to the modified matrix.
         */
        Matrix& operator=(std::initializer_list<float> init);

        /**
         * @brief Outputs the matrix to an output stream.
         * @param os The output stream.
         * @param matrix The matrix to output.
         * @return The output stream.
         */
        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

        /**
         * @brief Outputs a row of the matrix to an output stream.
         * @param os The output stream.
         * @param row The row to output as a vector of floats.
         * @return The output stream.
         */
        friend std::ostream& operator<<(std::ostream& os, const std::vector<float>& row);

        /**
         * @brief Randomizes the matrix elements with values between -1.0 and 1.0.
         * @return A reference to the modified matrix.
         */
        Matrix randomize();

        /**
         * @brief Randomizes the matrix elements within a specified range.
         * @param min The minimum value for randomization.
         * @param max The maximum value for randomization.
         * @return A reference to the modified matrix.
         */
        Matrix randomize(float min, float max);

        /**
         * @brief Sets all matrix elements to zero.
         * @return A reference to the modified matrix.
         */
        Matrix nullMatrix();
    };
}
