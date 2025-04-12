#include <iostream>
#include <chrono>

#include "mann.h"

size_t m1rows = 10;
size_t m1cols = 8294400;

size_t m2rows = 8294400;
size_t m2cols = 1;

int main()
{
    Mann::Matrix aMatrix(m1rows, m1cols);
    aMatrix = aMatrix.randomize();

    Mann::Matrix bMatrix(m2rows, m2cols);
    bMatrix = bMatrix.randomize();

    // Perform matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();

    Mann::Matrix result = aMatrix * bMatrix;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // std::cout << "Result: \n" << result << std::endl;
    std::cout << "RESULT TILLING + AMP ENDED in " << duration.count() << std::endl;

    return 0;
}