#pragma once

#include "mann.h"

class MNNetwork
{
public:
    MNNetwork();
    ~MNNetwork();

    void initializeNetwork(std::vector<size_t> layers_size, std::vector<Mann::Matrix> &layers, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases);
    void feedForward(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases);
    std::vector<std::vector<Mann::Matrix>> backPropagation(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases, const Mann::Matrix &target);
    void activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);
    void der_activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);
};