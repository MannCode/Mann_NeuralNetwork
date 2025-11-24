#include <vector>

#include "mann.h"

struct NetworkConfiguration
{
  std::string model_name;
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
