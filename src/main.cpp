#include "mann.h"
#include "MNNetwork.h"
#include "mnist.h"
#include "mannui.hpp"

using namespace std;

int main()
{
    // MNNetwork network;
    Mnist mnist;

    
    // Load MNIST data
    std::vector<std::vector<double>> mnist_images_data, mnist_labels_data;
    mnist.ReadMNISTimages(10000, 784, mnist_images_data);
    mnist.ReadMNISTlabels(10000, mnist_labels_data);
    
    
    // std::string MNN_network_file = "MNN_Network_784_50_10_10.mms";
    float learning_rate = 0.01;
    size_t iterations = 10000;
    size_t batch_size = 20;

    // network.trainNetwork(iterations, batch_size, mnist_images_data, mnist_labels_data, MNN_network_file, learning_rate);

    // network.testNetworkByUser(mnist_images_data, mnist_labels_data, MNN_network_file);

    // network.testNetwork(mnist_images_data, mnist_labels_data, MNN_network_file);

    // Init GLFW
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

    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "MannUI", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize MannUI
    MannUI ui(window, learning_rate, iterations, batch_size);

   while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ui.Render();

        glfwSwapBuffers(window);
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    /*
    1. MODE: training, testing
    
    training: graphs: accuracy
    
    */
    
    return 0;
}