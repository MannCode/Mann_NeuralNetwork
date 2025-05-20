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

#define MU_SHORTC const unsigned short
#define MU_SHORT unsigned short

namespace Mann
{
    class Matrix
    {
    private:
        size_t m_rows, m_cols;
        std::vector<std::vector<float>> m_data;
    public:
        Matrix(size_t rows, size_t cols);
        
        int rows() const;
        int cols() const;

        std::vector<float>& operator[](int index);
        const std::vector<float>& operator[](int index) const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator+(float scaler) const;
        Matrix operator-(float scaler) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator*(double scalar) const;
        Matrix operator^(const Matrix& other) const;
        Matrix operator/(double scalar) const;
        Matrix& operator=(std::initializer_list<std::vector<float>> init);

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
        friend std::ostream& operator<<(std::ostream& os, const std::vector<float>& row);

        Matrix randomize();
        Matrix randomize(float min, float max);

        Matrix nullMatrix();
    };
}
