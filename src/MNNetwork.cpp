#include "MNNetwork.shit"
#include <cassert>

/**
 * @brief Constructs a neural network by loading from a file.
 * @param filename The name of the file containing the network configuration and weights.
 */
MNNetwork::MNNetwork(std::string filename)
        : m_filename(filename)
{
    m_accuracy = 0.0f;
    m_total_training_time = 0.0f;

    loadNetwork(m_filename);
}

/**
 * @brief Constructs a neural network with specified hidden layer sizes or loads from a file.
 * @param filename The name of the file to save/load the network configuration.
 * @param hidden_layers_size A vector specifying the number of neurons in each hidden layer.
 * @param learning_rate The learning rate for weight updates during training.
 * @param batch_size The size of each training batch.

 */
MNNetwork::MNNetwork(std::string filename, std::vector<size_t> hidden_layers_size, float learning_rate, size_t batch_size)
                    : m_learning_rate(learning_rate), m_batch_size(batch_size), m_filename(filename)
{
    m_accuracy = 0.0f;
    m_total_training_time = 0.0f;

    std::ifstream file("../models/" + m_filename);
    file.good() ?  loadNetwork(m_filename) 
    : CreateNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, hidden_layers_size);
};

/**
 * @brief Destructor for the MNNetwork class.
 *
 * Cleans up resources used by the neural network.
 */
MNNetwork::~MNNetwork() {};

/**
 * @brief Trains the neural network using the provided dataset.
 * @param iterations The number of training iterations.
 * @param batch_size The size of each training batch.
 * @param images_data A vector of input image data for training.
 * @param labels_data A vector of corresponding label data for training.
 * @param filename The file to save the trained network.
 * @param learning_rate The learning rate for weight updates during training.
 */
void MNNetwork::trainNetwork(const size_t iterations, std::vector<std::vector<double>> &images_data, 
                            std::vector<std::vector<double>> &labels_data, bool *is_training)
{
    
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;
    std::vector<Mann::Matrix> MNN_d_weights = MNN_Weights;
    std::vector<Mann::Matrix> MNN_d_biases = MNN_Bias;

    for(int n = 0; n < iterations; n++) {
        float avg_cost_bulk = 0;
        for(int batch = 0; batch < images_data.size()/m_batch_size; batch++) {
            float avg_cost_bulk_batch = 0;
            current_batch = batch;
            for (int j = 0; j < MNN_d_weights.size(); j++) {
                MNN_d_weights[j].nullMatrix();
                MNN_d_biases[j].nullMatrix();
            }

            for (int i = batch * m_batch_size; i < (batch + 1) * m_batch_size; i++) {
                // load image data in network
                for (int j =0; j < MNN_Nodes[0].rows(); j++) {
                    MNN_Nodes[0][j][0] = images_data[i][j];
                }
                for (int j = 0; j < MNN_y.rows(); j++) {
                    MNN_y[j][0] = labels_data[i][j];
                }

                feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

                Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
                MNN_cost = MNN_cost ^ MNN_cost;
                float avg_cost = 0;
                for (int j = 0; j < MNN_cost.rows(); j++) {
                    avg_cost += MNN_cost[j][0];
                }
                m_accuracy_crr_image = (10 - avg_cost) * 10;
                m_accuracy_crr_image_history.push_back(m_accuracy_crr_image);
                avg_cost_bulk = (avg_cost_bulk + avg_cost) / 2;
                avg_cost_bulk_batch = (avg_cost_bulk_batch + avg_cost) / 2;

                std::vector<std::vector<Mann::Matrix>> MNN_d_weights_biases = backPropagation(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias, MNN_y);
                for(int j = 0; j < MNN_d_weights.size(); j++) {
                    MNN_d_weights[j] = (MNN_d_weights[j] + MNN_d_weights_biases[0][j])/2;
                    MNN_d_biases[j] = (MNN_d_biases[j] + MNN_d_weights_biases[1][j])/2;
                }
            }

            m_accuracy_crr_batch = (10 - avg_cost_bulk_batch) * 10;
            m_accuracy_crr_batch_history.push_back(m_accuracy_crr_batch);
            
            // time to update the weights and biases
            for (int j = 0; j < MNN_Weights.size(); j++) {
                MNN_Weights[j] = MNN_Weights[j] - (MNN_d_weights[j] * m_learning_rate);
                MNN_Bias[j] = MNN_Bias[j] - (MNN_d_biases[j] * m_learning_rate);
            }
            
            if(is_training && !(*is_training)) {
                return;
            }
        }
        saveNetwork();
        m_accuracy = (10 - avg_cost_bulk) * 10;
        m_accuracy_history.push_back(m_accuracy);
    }
}

/**
 * @brief Tests the neural network interactively with user-provided image indices.
 * @param images_data A vector of input image data for testing.
 * @param labels_data A vector of corresponding label data for testing.
 * @param filename The file containing the network configuration.
 */
void MNNetwork::testNetworkByUser(std::vector<std::vector<double>> &images_data, 
                                 std::vector<std::vector<double>> &labels_data, 
                                 const std::string &filename)
{
    
    // loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    while (true) {
        // get index of the image to test by user
        int index = 0;
        std::cout << "Enter the index of the image to test (0 - 9999): ";
        std::cin >> index;
        if (index < 0 || index >= images_data.size()) {
            std::cout << "Invalid index. Exiting." << std::endl;
        }
        else {
            // load image data in network
            for (int j = 0; j < MNN_Nodes[0].rows(); j++) {
                MNN_Nodes[0][j][0] = images_data[index][j];
            }
            for (int j = 0; j < MNN_y.rows(); j++) { 
                MNN_y[j][0] = labels_data[index][j];
            }
            
            feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

            Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
            MNN_cost = MNN_cost ^ MNN_cost;
            float avg_cost = 0;
            for (int j = 0; j < MNN_cost.rows(); j++) {
                avg_cost += MNN_cost[j][0];
            }

            std::cout << "Actural number: ";
            for (int j = 0; j < MNN_y.rows(); j++) {
                if(MNN_y[j][0] == 1) {
                    std::cout << j << std::endl;
                    break;
                }
            }
            float highest = 0;
            int num = 0;
            std::cout << "Predicted number: ";
            for (int j = 0; j < MNN_Nodes[MNN_Nodes.size() - 1].rows(); j++) {
                if(MNN_Nodes[MNN_Nodes.size() - 1][j][0] > highest) {
                    highest = MNN_Nodes[MNN_Nodes.size() - 1][j][0];
                    num = j;
                }
            }
            std::cout << num << std::endl;
            std::cout << std::endl;
            printLables(MNN_Nodes[MNN_Nodes.size() - 1]);
            // std::cout << "Predicted Labels: ";
            // for (int j = 0; j < MNN_Nodes[MNN_Nodes.size() - 1].rows(); j++) {
            //     std::cout << MNN_Nodes[MNN_Nodes.size() - 1][j][0] << " ";
            // }
            std::cout << std::endl;
            std::cout << "Accuracy: " << (10 - avg_cost) * 10 << "%" << std::endl << std::endl << std::endl;

            // print the image
            saveImageDataToFile(images_data[index], labels_data[index], "test_image.mms");
        }
    }
}

/**
 * @brief Tests the neural network using the provided dataset.
 */
void MNNetwork::testNetwork(std::vector<std::vector<double>> &mnist_images_data, 
                          std::vector<std::vector<double>> &mnist_labels_data)
{
    
    // loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    float avg_cost_bulk = 0;

    // MNN_Nodes[0][783][0] = mnist_images_data[0][783];

    for (int i = 0; i < mnist_images_data.size(); i++) {

        // load image data in network
        // std::cout << MNN_Nodes[0].cols() << std::endl;
        for (int j = 0; j < MNN_Nodes[0].rows(); j++) {
            MNN_Nodes[0][j][0] = mnist_images_data[i][j];
            // std::cout << "hello";
        }
        for (int j = 0; j < MNN_y.rows(); j++) { 
            MNN_y[j][0] = mnist_labels_data[i][j];
        }
        
        feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

        Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
        MNN_cost = MNN_cost ^ MNN_cost;
        float avg_cost = 0;
        for (int j = 0; j < MNN_cost.rows(); j++) {
            avg_cost += MNN_cost[j][0];
        }

        avg_cost_bulk = (avg_cost_bulk + avg_cost) / 2;
    }
    // std::cout << std::endl;
    // std::cout << "Accuracy: " << (10 - avg_cost_bulk) * 10 << "%" << std::endl << std::endl << std::endl;
    // return (10 - avg_cost_bulk) * 10;
    m_accuracy = (10 - avg_cost_bulk) * 10;
}

/**
 * @brief Initializes the neural network with the specified layer sizes.
 * @param layers_size A vector specifying the size of each layer.
 * @param nodes A vector of matrices to store the node values in each layer.
 * @param weights A vector of matrices to store the weights between layers.
 * @param biases A vector of matrices to store the biases for each layer.
 */
void MNNetwork::initializeNetwork(std::vector<size_t> layers_size, 
                                 std::vector<Mann::Matrix> &nodes, 
                                 std::vector<Mann::Matrix> &weights, 
                                 std::vector<Mann::Matrix> &biases)
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

/**
 * @brief Performs the feedforward operation on the neural network.
 * @param nodes A vector of matrices representing the node values in each layer.
 * @param weighted_sum A vector of matrices representing the weighted sums before activation.
 * @param weights A vector of matrices representing the weights between layers.
 * @param biases A vector of matrices representing the biases for each layer.
 */
void MNNetwork::feedForward(std::vector<Mann::Matrix> &nodes, 
                           std::vector<Mann::Matrix> &weighted_sum, 
                           std::vector<Mann::Matrix> &weights, 
                           std::vector<Mann::Matrix> &biases)
{
    for (size_t i = 0; i < nodes.size() - 1; ++i)
    {
        weighted_sum[i] = weights[i] * nodes[i] + biases[i];
        activationFunction(nodes[i + 1], weighted_sum[i]);
    }
}

/**
 * @brief Applies the sigmoid activation function to the weighted sum.
 * @param matrix The output matrix after applying the activation function.
 * @param weighted_sum The input matrix containing the weighted sum.
 */
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

/**
 * @brief Computes the derivative of the sigmoid activation function.
 * @param matrix The output matrix containing the derivative values.
 * @param nodes The input matrix containing the node values after activation.
 */
void MNNetwork::der_activationFunction(Mann::Matrix &matrix, const Mann::Matrix &nodes)
{
    matrix = nodes ^ ((nodes * -1) + 1);
}

/**
 * @brief Performs backpropagation to compute gradients for weight updates.
 * @param nodes A vector of matrices representing the node values in each layer.
 * @param weighted_sum A vector of matrices representing the weighted sums before activation.
 * @param weights A vector of matrices representing the weights between layers.
 * @param biases A vector of matrices representing the biases for each layer.
 * @param target The target output matrix for the current input.
 * @return A vector of vectors of matrices containing gradients for weights and biases.
 */
std::vector<std::vector<Mann::Matrix>> MNNetwork::backPropagation(std::vector<Mann::Matrix> &nodes, 
                                                                std::vector<Mann::Matrix> &weighted_sum, 
                                                                std::vector<Mann::Matrix> &weights, 
                                                                std::vector<Mann::Matrix> &biases, 
                                                                const Mann::Matrix &target)
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

/**
 * @brief Saves the neural network configuration and weights to a file.
 */
void MNNetwork::saveNetwork()
{
    std::ofstream file("../models/" + m_filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving network: " << m_filename << std::endl;
        return;
    }
    // Save layers_size
    for (size_t i = 0; i < MNN_Layers_size.size(); ++i) {
        file << MNN_Layers_size[i] << (i + 1 < MNN_Layers_size.size() ? " " : "\n");
    }
    // Save Learning Rate
    file << m_learning_rate << "\n";
    // Save Batch Size
    file << m_batch_size << "\n";
    // Save Accuracy
    file << m_accuracy << "\n";
    // Save Total Training Time
    file << m_total_training_time << "\n";
    // Save weights
    for (Mann::Matrix weight : MNN_Weights) {
        file << weight;
    }
    // Save biases
    for (Mann::Matrix bias : MNN_Bias) {
        file << bias;
    }
}

/**
 * @brief Loads the neural network configuration and weights from a file.
 * @param filename The file containing the network configuration.
 */
void MNNetwork::loadNetwork(const std::string &filename)
{
    size_t layer_size;
    std::ifstream file(("../models/" + filename));

    if (!file.is_open()) {
        std::cerr << "Error opening file for loading network: " << filename << std::endl;
        return;
    }

    // get first line (layers size)
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    while (iss >> layer_size) {
        MNN_Layers_size.push_back(static_cast<size_t>(layer_size));
    }

    // Get second line (learning rate)
    file >> m_learning_rate;
    
    // Get third line (batch size)
    file >> m_batch_size;

    // Get fourth line (accuracy)
    file >> m_accuracy;

    // Get fifth line (total training time)
    file >> m_total_training_time;

    // Load weights
    for (size_t i = 0; i < MNN_Layers_size.size() - 1; ++i) {
        Mann::Matrix weight(MNN_Layers_size[i + 1], MNN_Layers_size[i]);
        for (size_t j = 0; j < MNN_Layers_size[i + 1]; ++j) {
            for (size_t k = 0; k < MNN_Layers_size[i]; ++k) {
                file >> weight[j][k];
            }
        }
        MNN_Weights.push_back(weight);
    }
    // Load biases
    for (size_t i = 0; i < MNN_Layers_size.size() - 1; ++i) {
        Mann::Matrix bias(MNN_Layers_size[i + 1], 1);
        for (size_t j = 0; j < MNN_Layers_size[i + 1]; ++j) {
            file >> bias[j][0];
        }
        MNN_Bias.push_back(bias);
    }

    // Initialize nodes
    for (size_t i = 0; i < MNN_Layers_size.size(); ++i) {
        MNN_Nodes.emplace_back(Mann::Matrix(MNN_Layers_size[i], 1));
    }
}

/**
 * @brief Creates a new neural network with the specified configuration and saves it.
 * @param layers_size A vector to store the size of each layer.
 * @param learning_rate The learning rate for weight updates during training.
 * @param batch_size The size of each training batch.
 * @param nodes A vector of matrices to store the node values in each layer.
 * @param weights A vector of matrices to store the weights between layers.
 * @param biases A vector of matrices to store the biases for each layer.
 * @param hidden_layers_size A vector specifying the number of neurons in each hidden layer.
 * @param modelName The name of the neural network model.
 */
void MNNetwork::CreateNetwork(std::vector<size_t> &layers_size,
                             std::vector<Mann::Matrix> &nodes, 
                             std::vector<Mann::Matrix> &weights, 
                             std::vector<Mann::Matrix> &biases, 
                             std::vector<size_t> &hidden_layers_size)
{
    std::string path = "../models/" + m_filename;
    std::ofstream outfile(path); // mandeep model storage
 
    if (!outfile)
    {
        std::cerr << "Error creating new file!" << std::endl;
        return;
    }

    // [783, hidden, 10]
    layers_size.push_back(784); // input layer
    for (size_t i = 0; i < hidden_layers_size.size(); i++) {
        layers_size.push_back(hidden_layers_size[i]); // hidden layers
    }
    layers_size.push_back(10);  // output layer

    initializeNetwork(layers_size, nodes, weights, biases);
    saveNetwork();

    outfile.close();
    std::cout << "File created successfully!" << std::endl; 
}

/**
 * @brief Saves image and label data to a file in a formatted manner.
 * @param image_data A vector containing the image data.
 * @param label_data A vector containing the corresponding label data.
 * @param filename The file to save the image and label data.
 */
void MNNetwork::saveImageDataToFile(const std::vector<double>& image_data, 
                                   const std::vector<double>& lable_data,
                                   const std::string& filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                // only 2 decimal points
                if (image_data[i * 28 + j] < 0.1)
                {
                    file << ".....";
                }
                else
                {
                    file << std::fixed << std::setprecision(2) << image_data[i * 28 + j] << " ";
                }
                
                // file << image_data[i * 28 + j] << " ";
            }
            file << std::endl << std::endl;
            
        }
        file << "Label: ";
        for (int i = 0; i < 10; i++)
        {
            file << lable_data[i] << " ";
        }
        file << std::endl;
        file.close();
    }
}

/**
 * @brief Prints the predicted label probabilities as a histogram.
 * @param matrix The matrix containing predicted label probabilities.
 */
void MNNetwork::printLables(const Mann::Matrix &matrix)
{
    std::cout << "Predicted Labels: " << std::endl;
    for (int j = 0; j < matrix.rows(); j++) {
        std::cout << j << ": " << " ";
        int _char = matrix[j][0] * 50;
        if (_char == 0) {
            _char = 1;
        }
        for (int i = 0; i < _char; i++) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }
}