#include <iostream>
#include <chrono>

#include "mann.h"
#include "MNNetwork.h"

size_t m1rows = 10;
size_t m1cols = 8294400;

size_t m2rows = 8294400;
size_t m2cols = 1;

int main()
{
    MNNetwork network;

    std::vector<size_t> MNN_Layers_size = {1, 1, 1, 1};

    float first_layer[] = {0.5};

    float y[] = {0.1};
    Mann::Matrix MNN_y(1, 1);
    for (int i = 0; i < MNN_y.rows(); i++) { MNN_y[i][0] = y[i]; }

    std::vector<Mann::Matrix> MNN_Nodes, MNN_Weights, MNN_Bias;
    network.initializeNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias; 

    for (int i =0; i < MNN_Nodes[0].rows(); i++) {
        MNN_Nodes[0][i][0] = first_layer[i];
    }

    for(int i = 0; i < 10000; i++) {

        //feed forward
        network.feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

        Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
        MNN_cost = MNN_cost ^ MNN_cost;
        std::cout << "Cost: " << MNN_cost << std::endl;

        std::vector<std::vector<Mann::Matrix>> MNN_d_weights_biases;
        MNN_d_weights_biases = network.backPropagation(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias, MNN_y);

        for (int i = 0; i < MNN_Weights.size(); i++) {
            MNN_Weights[i] = MNN_Weights[i] - (MNN_d_weights_biases[0][i]);
            MNN_Bias[i] = MNN_Bias[i] - (MNN_d_weights_biases[1][i]);
        }
        
    }

    // std::cout << MNN_Nodes[0] << std::endl;

    return 0;
}