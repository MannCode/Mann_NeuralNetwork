#include <iostream>
#include <chrono>

#include "mann.h"
#include "MNNetwork.h"

int main()
{
    MNNetwork network;

    std::vector<size_t> MNN_Layers_size = {2, 3, 2};

    float first_layer[] = {1, 0};

    float y[] = {0, 1};
    Mann::Matrix MNN_y(2, 1);
    for (int i = 0; i < MNN_y.rows(); i++) { MNN_y[i][0] = y[i]; }

    std::vector<Mann::Matrix> MNN_Nodes, MNN_Weights, MNN_Bias;
    network.initializeNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    for (int i =0; i < MNN_Nodes[0].rows(); i++) {
        MNN_Nodes[0][i][0] = first_layer[i];
    }


    for(int i = 0; i < 100; i++) {

        //feed forward
        network.feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

        Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
        MNN_cost = MNN_cost ^ MNN_cost;
        float avg_cost = 0;
        for (int j = 0; j < MNN_cost.rows(); j++) {
            avg_cost += MNN_cost[j][0];
        }
        std::cout << "Cost: " << avg_cost << std::endl;

        std::cout << "Output: " << MNN_Nodes[MNN_Nodes.size() - 1] << std::endl;

        std::vector<std::vector<Mann::Matrix>> MNN_d_weights_biases;
        MNN_d_weights_biases = network.backPropagation(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias, MNN_y);
        
        for (int i = 0; i < MNN_Weights.size(); i++) {
            MNN_Weights[i] = MNN_Weights[i] - (MNN_d_weights_biases[0][i]);
            MNN_Bias[i] = MNN_Bias[i] - (MNN_d_weights_biases[1][i]);
        }
    }

    return 0;
}