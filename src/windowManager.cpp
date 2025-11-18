/**
 * @file windowManager.cpp
 * @brief Implementation of the WindowManager class for handling GLFW window creation, initialization, and the main rendering loop.
 * @author Jayansh Devgan
 */

#include "windowManager.h"

/**
 * @brief Initializes GLFW with specific hints for OpenGL context.
 *
 * This function initializes the GLFW library and sets window hints
 * tailored for macOS, including OpenGL version and profile settings.
 *
 * @return 0 on success, -1 if GLFW initialization fails.
 */
int WindowManager::initalizeGlfwWithHints()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // macOS-specific GLFW hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    return 0;
}

/**
 * @brief Creates a GLFW window with specified dimensions and title.
 *
 * This function creates a new GLFW window with a resolution of 1280x720
 * and the title "MannUI". It makes the context current and enables VSync.
 * If window creation fails, it terminates GLFW and returns nullptr.
 *
 * @return A pointer to the created GLFWwindow, or nullptr on failure.
 */
GLFWwindow* WindowManager::createWindow(int width, int height)
{
    GLFWwindow* window = glfwCreateWindow(width, height, "MannUI", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsyn

    return window;
}

/**
 * @brief Destroys the specified GLFW window and terminates GLFW.
 *
 * This function cleans up by destroying the given window and terminating
 * the GLFW library.
 *
 * @param window A pointer to the GLFWwindow to be destroyed.
 */
void WindowManager::adolfHitler(GLFWwindow* window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

/**
 * @brief Runs the main application loop for rendering and event handling.
 *
 * This function enters a loop that continues until the window should close.
 * It polls for events, clears the screen with a dark gray color, renders
 * the UI with the provided output text, and swaps the buffers.
 *
 * @param window A pointer to the GLFWwindow for the application.
 * @param ui A pointer to the MannUI object responsible for rendering.
 */
void WindowManager::mainLoop(GLFWwindow* window, MannUI* ui)
{
    while (!glfwWindowShouldClose(window))
     {
         glfwPollEvents();
         glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
         glClear(GL_COLOR_BUFFER_BIT);

         ui->Render(ui->outputText);

         glfwSwapBuffers(window);
     }
}
