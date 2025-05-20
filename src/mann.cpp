#include "mann.h"
#include <iomanip>

namespace Mann
{
    Matrix::Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows, std::vector<float>(cols, 0.0f)) {}

    int Matrix::rows() const
    {
        return m_rows;
    }
    int Matrix::cols() const
    {
        return m_cols;
    }

    std::vector<float>& Matrix::operator[](int index)
    {
        return m_data[index];
    }

    const std::vector<float>& Matrix::operator[](int index) const
    {
        return m_data[index];
    }

    Matrix Matrix::operator+(const Matrix& other) const
    {
        if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
        {
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] + other.m_data[i][j];
            }
        }

        return result;
    }

    Matrix Matrix::operator-(const Matrix& other) const
    {
        if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
        {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }

        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] - other.m_data[i][j];
            }
        }

        return result;
    }

    Matrix Matrix::operator+(float scaler) const
    {
        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] + scaler;
            }
        }

        return result;
    }

    Matrix Matrix::operator-(float scaler) const
    {
        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] - scaler;
            }
        }

        return result;
    }

// For windows
// #if defined(_WIN32) || defined(_WIN64) || defined(__linux__)

// For Macos
// #elif defined(__APPLE__) && defined(__MACH__)
    Matrix Matrix::operator*(const Matrix& other) const
    {
        static MU_SHORTC TS = 8;

        if (m_data[0].size() != other.m_data.size())
        {
            throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
        }

        size_t rows = m_data.size();
        size_t cols = other.m_data[0].size();
        Matrix result(rows, cols);

        std::function<void(int, int)> multiplyTile = [&](int rowStart, int colStart)
        {
            for (int i = rowStart; i < rowStart + TS && i < rows; ++i)
            {
                for (int j = colStart; j < colStart + TS && j < cols; ++j)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < m_data[0].size(); ++k)
                    {
                        sum += m_data[i][k] * other.m_data[k][j];
                    }
                    result[i][j] += sum;
                }
            }
        };

        std::vector<std::future<void>> tasks;

        for (int i = 0; i < rows; i += TS)
        {
            for (int j = 0; j < cols; j += TS)
            {
                tasks.emplace_back(std::async(std::launch::async, multiplyTile, i, j));
            }
        }

        for (auto& thread : tasks)
        {
            thread.get();
        }

        return result;
    }
// #endif

    Matrix Matrix::operator*(double scalar) const
    {
        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] * scalar;
            }
        }

        return result;
    }

    Matrix Matrix::operator^(const Matrix& other) const
    {
        // Element-wise multiplication
        if (m_data.size() != other.m_data.size() || m_data[0].size() != other.m_data[0].size())
        {
            throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
        }

        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);


        for (size_t i = 0; i < m_rows; ++i)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                result[i][j] = m_data[i][j] * other.m_data[i][j];
            }
        }

        return result;
    }

    Matrix Matrix::operator/(double scalar) const
    {
        if (scalar == 0)
        {
            throw std::invalid_argument("Division by zero.");
        }

        size_t rows = m_data.size();
        size_t cols = m_data[0].size();
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i][j] = m_data[i][j] / scalar;
            }
        }

        return result;
    }

    Matrix& Matrix::operator=(std::initializer_list<std::vector<float>> init)
    {
        m_rows = init.size();

        if (m_rows > 0)
            m_cols = init.begin()->size();
        else
            m_cols = 0;

        m_data.resize(m_rows);

        size_t index = 0;
        for (const auto& row : init)
        {
            m_data[index++] = row;
        }
        return *this;
    }

    std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
    {
        for (const auto& row : matrix.m_data)
        {
            for (const auto& elem : row)
            {
                if(elem > 0) {
                    os << "+";
                }
                os << std::fixed << std::setprecision(6) << elem << " ";
            }
            os << std::endl;
        }
        return os;
    }

    Matrix Matrix::randomize()
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(-1.0f, 1.0f);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] = distr(eng);
            }
        }
        return *this;
    }

    Matrix Matrix::randomize(float min, float max)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(min, max);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                m_data[i][j] = distr(eng);
            }
        }
        return *this;
    }

    Matrix Matrix::nullMatrix()
    {
        for (size_t i = 0; i < m_rows; ++i) {
            std::fill(m_data[i].begin(), m_data[i].end(), 0.0f);
        }
        return *this;
    }
}
