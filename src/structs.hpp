#include <vector>

#include "mann.h"

struct mnistData
{
    std::vector<std::vector<double>> mnist_images_data;
    std::vector<std::vector<double>> mnist_labels_data;
};

struct NetworkConfiguration
{
  std::vector<size_t> &hidden_layers;
  float &learning_rate;
  size_t &batch_size;
};

struct NetworkInitialization {
    std::vector<size_t> &layers_size;
    std::vector<Mann::Matrix> &nodes;
    std::vector<Mann::Matrix> &weights;
    std::vector<Mann::Matrix> &biases;
};

struct NetworkArchitecture {
    NetworkInitialization* network_initialization;
    std::vector<size_t> &hidden_layers_size;
};
