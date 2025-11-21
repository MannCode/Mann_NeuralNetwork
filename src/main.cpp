#include "mnist.h"
#include "windowManager.h"

int main()
{
    // MNNetwork network;
    Mnist mnist;

    // Init GLFW
    WindowManager::initalizeGlfwWithHints();

    // Create window
    GLFWwindow* window = WindowManager::createWindow(1280, 720);

    // Initialize MannUI
    MannUI ui(window, &mnist);

    WindowManager::mainLoop(window, &ui);

    // Cleanup
    WindowManager::adolfHitler(window);

    return 0;
}
