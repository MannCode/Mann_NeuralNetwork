
/**
 * @file mann.cpp
 * @brief Implementation of the Matrix class in the Mann namespace for neural network operations.
 * @author Jayansh Devgan, Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 *
 * This file contains the implementation of the Matrix class, providing functionality
 * for matrix operations such as addition, subtraction, multiplication, randomization,
 * and output formatting used in neural network computations.
 */

#include "mann.h"
#include <iomanip>


/**
 * @namespace Mann
 * @brief Namespace for matrix-related classes and utilities used in neural networks.
 */
namespace Mann
{
    /**
     * @brief Constructs a matrix with specified dimensions, initialized to zero.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    Matrix::Matrix(int rows, int cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0f) {}


    /**
     * @brief Gets the number of rows in the matrix.
     * @return The number of rows.
     */
    int Matrix::rows() const
    {
        return m_rows;
    }

    /**
     * @brief Gets the number of columns in the matrix.
     * @return The number of columns.
     */
    int Matrix::cols() const
    {
        return m_cols;
    }

    const std::vector<float> Matrix::data() const
    {
        return m_data;
    }

    /**
     * @brief Accesses a row of the matrix for modification.
     * @param index The row index.
     * @return A reference to the row as a vector of floats.
     */
    float& Matrix::operator[](int index)
    {
        return m_data[index];
    }

    /**
     * @brief Accesses a row of the matrix for read-only access.
     * @param index The row index.
     * @return A const reference to the row as a vector of floats.
     */
    const float& Matrix::operator[](int index) const
    {
        return m_data[index];
    }

    /**
     * @brief Adds two matrices element-wise.
     * @param other The matrix to add.
     * @return A new matrix containing the element-wise sum.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix Matrix::operator+(const Matrix& other) const
    {
        if (this->rows() != other.rows() || this->cols() != other.cols())
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] + other.m_data[i * m_cols + j];
            }
        }

        return result;
    }

    /**
     * @brief Subtracts a matrix from this matrix element-wise.
     * @param other The matrix to subtract.
     * @return A new matrix containing the element-wise difference.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix Matrix::operator-(const Matrix& other) const
    {
        if (this->rows() != other.rows() || this->cols() != other.cols())
        {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }

        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] - other.m_data[i * m_cols + j];
            }
        }

        return result;
    }

    /**
     * @brief Adds a scalar to each element of the matrix.
     * @param scaler The scalar value to add.
     * @return A new matrix with the scalar added to each element.
     */
    Matrix Matrix::operator+(float scaler) const
    {

        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] + scaler;
            }
        }

        return result;
    }

    /**
     * @brief Subtracts a scalar from each element of the matrix.
     * @param scaler The scalar value to subtract.
     * @return A new matrix with the scalar subtracted from each element.
     */
    Matrix Matrix::operator-(float scaler) const
    {

        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] - scaler;
            }
        }

        return result;
    }

// For windows
// #if defined(_WIN32) || defined(_WIN64) || defined(__linux__)

// For Macos
// #elif defined(__APPLE__) && defined(__MACH__)
    /**
     * @brief Performs matrix multiplication with another matrix.
     * @param other The matrix to multiply with.
     * @return A new matrix containing the result of matrix multiplication.
     * @throws std::invalid_argument If the matrix dimensions do not allow multiplication.
     */
    Matrix Matrix::operator*(const Matrix& other) const
    {
        Matrix result(m_rows, other.m_cols);

        // std::cout << "Metal platform not available. Using CPU for matrix multiplication." << std::endl;
        // Fallback to CPU implementation if Metal platform is not available
        // static MU_SHORTC TS = 8;

        // if (this->cols() != other.rows())
        // {
        //     throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
        // }

        // std::function<void(int, int)> multiplyTile = [&](int rowStart, int colStart)
        // {
        //     for (int i = rowStart; i < rowStart + TS  && i < m_rows; i++)
        //     {
        //         for (int j = colStart; j < colStart + TS && j < other.m_cols; j++)
        //         {
        //             float sum = 0.0f;
        //             for (int k = 0; k < m_cols; k++)
        //             {
        //                 sum += m_data[i * m_cols + k] * other.m_data[k * other.m_cols + j];
        //             }
        //             result[i * result.m_cols + j] += sum;
        //         }
        //     }
        // };

        // std::vector<std::future<void>> tasks;

        // for (int i = 0; i < m_rows; i += TS)
        // {
        //     for (int j = 0; j < other.m_cols; j += TS)
        //     {
                
        //         tasks.emplace_back(std::async(std::launch::async, multiplyTile, i, j));
        //     }
        // }

        // for (auto& thread : tasks)
        // {
        //     thread.get();
        // }

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < other.m_cols; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < m_cols; k++)
                {
                    sum += m_data[i * m_cols + k] * other.m_data[k * other.m_cols + j];
                }
                result[i * result.m_cols + j] = sum;
            }
        }

        return result;
    }


    
// #endif

    /**
     * @brief Multiplies each element of the matrix by a scalar.
     * @param scalar The scalar value to multiply by.
     * @return A new matrix with each element scaled.
     */
    Matrix Matrix::operator*(double scalar) const
    {
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] * scalar;
            }
        }

        return result;
    }

    /**
     * @brief Performs element-wise multiplication (Hadamard product) with another matrix.
     * @param other The matrix to multiply element-wise.
     * @return A new matrix containing the element-wise product.
     * @throws std::invalid_argument If the matrix dimensions do not match.
     */
    Matrix Matrix::operator^(const Matrix& other) const
    {
        // Element-wise multiplication
        if (m_rows != other.m_rows || m_cols != other.m_cols)
        {
            throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
        }

        
        Matrix result(m_rows, m_cols);


        for (int i = 0; i < m_rows; ++i)
        {
            for (int j = 0; j < m_cols; ++j)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] * other.m_data[i * m_cols + j];
            }
        }

        return result;
    }

    /**
     * @brief Divides each element of the matrix by a scalar.
     * @param scalar The scalar value to divide by.
     * @return A new matrix with each element divided by the scalar.
     * @throws std::invalid_argument If the scalar is zero.
     */
    Matrix Matrix::operator/(double scalar) const
    {
        if (scalar == 0)
        {
            throw std::invalid_argument("Division by zero.");
        }

        
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i * m_cols + j] = m_data[i * m_cols + j] / scalar;
            }
        }

        return result;
    }

    /**
     * @brief Assigns values to the matrix using an initializer list.
     * @param init The initializer list of vectors containing matrix data.
     * @return A reference to the modified matrix.
     */
    Matrix& Matrix::operator=(std::initializer_list<float> init)
    {
        if (init.size() != m_rows * m_cols)
        {
            throw std::invalid_argument("Initializer list size does not match matrix size.");
        }
        std::copy(init.begin(), init.end(), m_data.begin());
        return *this;
    }

    /**
     * @brief Outputs the matrix to an output stream.
     * @param os The output stream.
     * @param matrix The matrix to output.
     * @return The output stream.
     */
    std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
    {
        for (int i = 0; i < matrix.m_rows; ++i)
        {
            for (int j = 0; j < matrix.m_cols; ++j)
            {
                os << std::fixed << std::setprecision(4) << matrix.m_data[i * matrix.m_cols + j] << " ";
            }
            os << std::endl;
        }
        return os;
    }

    /**
     * @brief Randomizes the matrix elements with values between -1.0 and 1.0.
     * @return A reference to the modified matrix.
     */
    Matrix Matrix::randomize()
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
        for (int i = 0; i < m_rows; ++i) {
            for (int j = 0; j < m_cols; ++j) {
                m_data[i * m_cols + j] = distr(eng);
            }
        }
        return *this;
    }

    /**
     * @brief Randomizes the matrix elements within a specified range.
     * @param min The minimum value for randomization.
     * @param max The maximum value for randomization.
     * @return A reference to the modified matrix.
     */
    Matrix Matrix::randomize(float min, float max)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(min, max);
        for (int i = 0; i < m_rows; ++i) {
            for (int j = 0; j < m_cols; ++j) {
                m_data[i * m_cols + j] = distr(eng);
            }
        }
        return *this;
    }

    /**
     * @brief Sets all matrix elements to zero.
     * @return A reference to the modified matrix.
     */
    Matrix Matrix::nullMatrix()
    {
        for(int i = 0; i < m_rows * m_cols; ++i) {
            m_data[i] = 0.0f;
        }
        return *this;
    }
}
