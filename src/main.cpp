#include "mnist.h"
#include "windowManager.h"

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
    WindowManager::initalizeGlfwWithHints();

    // Create window
    GLFWwindow* window = WindowManager::createWindow(1280, 720);

    // Initialize MannUI
    mnistData* mnist_data = new mnistData {mnist_images_data, mnist_labels_data};
    MannUI ui(window, mnist_data);

    WindowManager::mainLoop(window, &ui);

    // Cleanup
    WindowManager::adolfHitler(window);

    /*
    1. MODE: training, testing

    training: graphs: accuracy

    */

    return 0;
}
