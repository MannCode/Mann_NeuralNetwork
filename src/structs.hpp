#include <vector>

#include "mann.h"

struct NetworkConfiguration
{
  std::string model_name;
  std::vector<int> &hidden_layers;
  float &learning_rate;
  int &batch_size;
};

struct NetworkInitialization {
    std::vector<int> &layers_size;
    std::vector<Mann::Matrix> &nodes;
    std::vector<Mann::Matrix> &weights;
    std::vector<Mann::Matrix> &biases;
};

struct NetworkArchitecture {
    NetworkInitialization* network_initialization;
    std::vector<int> &hidden_layers_size;
};
