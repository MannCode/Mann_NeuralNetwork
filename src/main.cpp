#include "mnist.h"
#include "windowManager.h"
#include "mann.h"

// #include "platforms/platform.h"


// Do not remove these definations. PLEASE!!!!!!!!!!!!!!!!!!!!!!!!!
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

// #include "platforms/platform.h"

int main()
{

    Mnist mnist;

    // Init GLFW
    WindowManager::initalizeGlfwWithHints();

    // Create window
    GLFWwindow* window = WindowManager::createWindow(1920, 1080);

    // Initialize MannUI
    MannUI ui(window, &mnist);

    WindowManager::mainLoop(window, &ui);

    // // Cleanup
    WindowManager::adolfHitler(window);


    // Mann::Matrix A(2500, 3000);
    // Mann::Matrix B(3000, 1500);
    // Mann::Matrix C(2500, 1500);

    // Mann::Matrix A(784, 50);
    // Mann::Matrix B(50, 1);
    // Mann::Matrix C(784, 1);

    // // perform some operations on A and B
    // // C = A + B;


    // A.randomize();
    // B.randomize();

    // // std::cout << A << std::endl;
    // // std::cout << B << std::endl;


    // auto startTime = std::chrono::high_resolution_clock::now();

    // C = A * B;

    // auto endTime = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> duration = endTime - startTime;
    // std::cout << "Time taken for multiplication on CPU: " << duration.count() << " s" << std::endl;

    // // std::cout << C << std::endl;

    
    // // C = Mann::Matrix(100, 1); // reset C

    // MetalPlatform platform;

    // auto startTimeGPU = std::chrono::high_resolution_clock::now();

    // platform.matrixMultiply(A, B, C);

    // auto endTimeGPU = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float> durationGPU = endTimeGPU - startTimeGPU;
    // std::cout << "Time taken for multiplication on GPU: " << durationGPU.count() << " s" << std::endl;

    // std::cout << C << std::endl;

    return 0;
}

