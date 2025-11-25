#pragma once

#include <fstream>

#include "structs.hpp"
#include "mnist.h"
#include "GLFW/glfw3.h"
#include <queue>

/**
 * @file MNNetwork.h
 * @brief Header file for the MNNetwork class, which implements a neural network for training and testing.
 * @author Jayansh Devgan, Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 */

 /**
 * @class MNNetwork
 * @brief A class representing a multi-layer neural network for machine learning tasks.
 *
 * This class provides functionality for creating, training, testing, and saving/loading
 * a neural network. It supports feedforward and backpropagation operations, along with
 * activation functions and weight/bias management.
 */

class MNNetwork
{
public:
    /**
     * @brief Constructs a neural network by loading from a file.
     * @param model_id The identifier of the model containing the network configuration and weights.
     */
    MNNetwork(std::string model_id);

    /**
     * @brief Constructs a neural network with specified hidden layer sizes.
     * @param model_id The identifier of the model to save/load the network configuration.
     * @param hidden_layers_size A vector specifying the number of neurons in each hidden layer.
     * @param learning_rate The learning rate for weight updates during training.
     * @param batch_size The size of each training batch.
     */
    MNNetwork(std::string model_id, NetworkConfiguration* network_configuration);

    /**
     * @brief Destructor for the MNNetwork class.
     *
     * Cleans up resources used by the neural network.
     */
    ~MNNetwork();

    inline std::string getModels(std::string _id)
    {
        #ifdef _WIN32
            return "../../models/" + _id + ".mms";
        #else
            return "../models/" + _id + ".mms";
        #endif
    }

    inline std::string getLogModelFiles(std::string LogFile)
    {
        #ifdef _WIN32
            return "../../models/modelsLogData/" + LogFile;
        #else
            return "../models/modelsLogData/" + Logfile;
        #endif
    }

    /**
     * @brief Trains the neural network using the provided dataset.
     * @param iterations The number of training iterations.
     * @param images_data A vector of input image data for training.
     * @param labels_data A vector of corresponding label data for training.
     */
    void trainNetwork(const int iterations, Mnist::MnistData* image_data, bool *is_training);


    /**
     * @brief Tests the neural network interactively with user-provided data.
     * @param images_data A vector of input image data for testing.
     */
    Mann::Matrix predictSingleImage(std::vector<double> &image_data);

    /**
     * @brief Tests the neural network using the provided dataset.
     */
    float testNetwork(Mnist::MnistData* image_data);

    /**
     * @brief Initializes the neural network with the specified layer sizes.
     * @param layers_size A vector specifying the size of each layer.
     * @param layers A vector of matrices representing the nodes in each layer.
     * @param weights A vector of matrices representing the weights between layers.
     * @param biases A vector of matrices representing the biases for each layer.
     */
    void initializeNetwork(NetworkInitialization* network_initialization);

    /**
     * @brief Performs the feedforward operation on the neural network.
     * @param nodes A vector of matrices representing the node values in each layer.
     * @param weighted_sum A vector of matrices representing the weighted sums before activation.
     * @param weights A vector of matrices representing the weights between layers.
     * @param biases A vector of matrices representing the biases for each layer.
     */
    void feedForward(std::vector<Mann::Matrix> &nodes,
                     std::vector<Mann::Matrix> &weighted_sum,
                     std::vector<Mann::Matrix> &weights,
                     std::vector<Mann::Matrix> &biases);

    /**
     * @brief Performs backpropagation to compute gradients for weight updates.
     * @param nodes A vector of matrices representing the node values in each layer.
     * @param weighted_sum A vector of matrices representing the weighted sums before activation.
     * @param weights A vector of matrices representing the weights between layers.
     * @param biases A vector of matrices representing the biases for each layer.
     * @param target The target output matrix for the current input.
     * @return A vector of vectors of matrices containing gradients for weights and biases.
     */
    std::vector<std::vector<Mann::Matrix>> backPropagation(std::vector<Mann::Matrix> &nodes,
                                                          std::vector<Mann::Matrix> &weighted_sum,
                                                          std::vector<Mann::Matrix> &weights,
                                                          std::vector<Mann::Matrix> &biases,
                                                          const Mann::Matrix &target);

    /**
     * @brief Applies the activation function to the weighted sum.
     * @param matrix The output matrix after applying the activation function.
     * @param weighted_sum The input matrix containing the weighted sum.
     */
    void activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);

    /**
     * @brief Computes the derivative of the activation function.
     * @param matrix The output matrix containing the derivative values.
     * @param weighted_sum The input matrix containing the weighted sum.
     */
    void der_activationFunction(Mann::Matrix &matrix, const Mann::Matrix &weighted_sum);

    bool IsPredictionCorrect(const Mann::Matrix &output_layer, const Mann::Matrix &target);

    /**
     * @brief Saves the neural network configuration and weights to a file.
     */
    void saveNetwork();

    /**
     * @brief Loads the neural network configuration and weights from a file.
     * @param layers_size A vector to store the size of each layer.
     * @param nodes A vector of matrices to store the node values in each layer.
     * @param weights A vector of matrices to store the weights between layers.
     * @param biases A vector of matrices to store the biases for each layer.
     * @param model_id The identifier of the model containing the network configuration.
     */
    void loadNetwork(const std::string &model_id);

    /**
     * @brief Saves image and label data to a file.
     * @param image_data A vector containing the image data.
     * @param label_data A vector containing the corresponding label data.
     * @param model_id The identifier of the model to save the image and label data.
     */
    void saveImageDataToFile(Mnist::MnistData* image_data,
                             const std::string& model_id);

    /**
     * @brief Prints the labels from a matrix.
     * @param matrix The matrix containing label data.
     */
    void printLables(const Mann::Matrix &matrix);

    /**
     * @brief Creates a neural network with the specified configuration.
     */
    void CreateNetwork(NetworkArchitecture* network_arch);

    void saveHistoryData();

    void loadHistoryData();

public:
    /**
     * @struct Networks
     * @brief A structure to store multiple neural network models and their names.
     */
    
    std::string m_model_id;
    std::string m_model_name;
    float m_learning_rate;
    int m_batch_size;
    float m_accuracy;
    float m_accuracy_testdata;
    float m_average_cost;
    float m_total_training_time;
    int sample_image_label;
    int m_current_epoch;

    //training related
    int current_batch;
    std::vector<float> m_accuracy_history;
    std::vector<float> m_accuracy_testdata_history;
    std::vector<float> m_average_cost_history;
    std::vector<float> m_average_cost_testdata_history;
    float m_batch_accuracy;
    std::queue<float> m_batch_accuracy_history;
    std::mutex training_threads_mutex;
    float m_averageTimePerBatch;
    


// private:
    std::vector<int> MNN_Layers_size;      ///< Sizes of the layers in the neural network.
    std::vector<Mann::Matrix> MNN_Nodes;      ///< Nodes (activations) for each layer.
    std::vector<Mann::Matrix> MNN_Weights;    ///< Weights between layers.
    std::vector<Mann::Matrix> MNN_Bias;       ///< Biases for each layer.
};
