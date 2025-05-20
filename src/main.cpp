#include "mann.h"
#include "MNNetwork.h"
#include "mnist.h"

using namespace std;

int main()
{
    MNNetwork network;
    Mnist mnist;

    // Load MNIST data
    std::vector<std::vector<double>> mnist_images_data, mnist_labels_data;
    mnist.ReadMNISTimages(10000, 784, mnist_images_data);
    mnist.ReadMNISTlabels(10000, mnist_labels_data);
    
    
    std::string MNN_network_file = "MNN_Network_784_16_16_10.txt";
    float learning_rate = 0.01;
    size_t iterations = 10000;
    size_t batch_size = 20;

    network.trainNetwork(iterations, batch_size, mnist_images_data, mnist_labels_data, MNN_network_file, learning_rate);

    // network.testNetworkByUser(mnist_images_data, mnist_labels_data, MNN_network_file);

    // network.testNetwork(mnist_images_data, mnist_labels_data, MNN_network_file);

    return 0;
}