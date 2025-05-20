#pragma once

#include "mann.h"

class MNNetwork
{
public:
    MNNetwork();
    ~MNNetwork();

    void trainNetwork(const size_t iterations, const size_t batch_size, std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename, float learning_rate);
    void testNetworkByUser(std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename);
    void testNetwork(std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename);
    void initializeNetwork(std::vector<size_t> layers_size, std::vector<Mann::Matrix> &layers, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases);
    void feedForward(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases);
    std::vector<std::vector<Mann::Matrix>> backPropagation(std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weighted_sum, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases, const Mann::Matrix &target);
    void activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);
    void der_activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);
    void saveNetwork(const std::vector<size_t>& layers_size, const std::vector<Mann::Matrix> &weights, const std::vector<Mann::Matrix> &biases, const std::string &filename);
    void loadNetwork(std::vector<size_t> &layers_size,  std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases, const std::string &filename);
    void saveImageDataToFile(const std::vector<double>& image_data, const std::vector<double>& lable_data, const std::string& filename);
};