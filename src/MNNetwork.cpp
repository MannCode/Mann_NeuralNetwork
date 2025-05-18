#include "MNNetwork.h"
#include "mann.h"

MNNetwork::MNNetwork() {};
MNNetwork::~MNNetwork() {};

void MNNetwork::initializeNetwork(std::vector<size_t> layers_size, std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases)
{
    for(int i=0; i < layers_size.size(); i++)
    {
        nodes.emplace_back(Mann::Matrix(layers_size[i], 1));
    }

    for(int i=0; i < layers_size.size() - 1; i++)
    {
        weights.emplace_back(Mann::Matrix(layers_size[i + 1], layers_size[i]));
        biases.emplace_back(Mann::Matrix(layers_size[i + 1], 1));

        weights[i].randomize();
        biases[i].randomize();
    }
}

void MNNetwork::feedForward(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases)
{
    for (size_t i = 0; i < nodes.size() - 1; ++i)
    {
        weighted_sum[i] = weights[i] * nodes[i] + biases[i];
        activationFunction(nodes[i + 1], weighted_sum[i]);
    }
}

void MNNetwork::activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum)
{
    for (size_t i = 0; i < matrix.rows(); ++i)
    {
        for (size_t j = 0; j < matrix.cols(); ++j)
        {
            matrix[i][j] = 1.0 / (1.0 + exp(-weighted_sum[i][j]));
        }
    }
}

void MNNetwork::der_activationFunction(Mann::Matrix &matrix, const Mann::Matrix &nodes)
{
    matrix = nodes ^ ((nodes * -1) + 1);
}

std::vector<std::vector<Mann::Matrix>> MNNetwork::backPropagation(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases, const Mann::Matrix &target)
{
    // Differentiation variables for backpropagation
    std::vector<Mann::Matrix> d_nodes;
    std::vector<Mann::Matrix> d_a_weighted_sum;
    std::vector<Mann::Matrix> d_weights;
    std::vector<Mann::Matrix> d_biases;

    std::vector<size_t> layers_size;
    for (size_t i = 0; i < nodes.size(); ++i) { layers_size.push_back(nodes[i].rows()); }
    initializeNetwork(layers_size, d_nodes, d_weights, d_biases);
    d_a_weighted_sum = d_biases;


    // Calculate gradients
    for(int i = d_a_weighted_sum.size()-1; i >= 0; i--)
    {
        der_activationFunction(d_a_weighted_sum[i], nodes[i+1]);
        if(i + 1 == nodes.size() - 1) {
            d_nodes[i+1] = (nodes[i+1] - target) * 2;
        }
        else {
            std::vector<Mann::Matrix> weights_front;
            // #pragma omp parallel for
            for (int j = 0; j < d_nodes[i+1].rows(); j++)
            {
                // d_nodes[i + 1][j] = 0;
                for (int k = 0; k < d_nodes[i+2].rows(); k++)
                {
                    d_nodes[i + 1][j][0] += weights[i+1][k][j] * d_a_weighted_sum[i+1][k][0] * d_nodes[i+2][k][0];
                }
            }
        }


        
        // #pragma omp parallel for
        for(int j = 0; j < nodes[i].rows(); j++)
        {
            for(int k = 0; k < nodes[i+1].rows(); k++)
            {
                d_weights[i][k][j] = nodes[i][j][0] * d_a_weighted_sum[i][k][0] * d_nodes[i+1][k][0];
            }
        }



        d_biases[i] = d_a_weighted_sum[i] ^ d_nodes[i + 1];
        
    }

    return {d_weights, d_biases};
}