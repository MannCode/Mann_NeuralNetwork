#include "MNNetwork.h"
#include <cassert>


/**
 * @brief Constructs a neural network by loading from a file.
 * @param filename The name of the file containing the network configuration and weights.
 */
MNNetwork::MNNetwork(std::string model_id)
        : m_model_id(model_id)
{
    m_accuracy = 0.0f;
    m_total_training_time = 0.0f;

    loadNetwork(m_model_id);
}

/**
 * @brief Constructs a neural network with specified hidden layer sizes or loads from a file.
 * @param filename The name of the file to save/load the network configuration.
 * @param hidden_layers_size A vector specifying the number of neurons in each hidden layer.
 * @param learning_rate The learning rate for weight updates during training.
 * @param batch_size The size of each training batch.

 */
MNNetwork::MNNetwork(std::string m_model_id, NetworkConfiguration* network_configuration)
                    : m_learning_rate(network_configuration->learning_rate), m_batch_size(network_configuration->batch_size), m_model_id(m_model_id), m_model_name(network_configuration->model_name)
{
    m_current_epoch = 1;
    m_total_training_time = 0.0f;
    NetworkInitialization* network_initalization = new NetworkInitialization{MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias};
    NetworkArchitecture* network_arch = new NetworkArchitecture{network_initalization, network_configuration->hidden_layers};

    std::ifstream file("../models/" + m_model_id + ".mms");
    file.good() ?  loadNetwork(m_model_id)
    : CreateNetwork(network_arch);
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
void MNNetwork::trainNetwork(const size_t iterations, Mnist::MnistData* image_data, bool *is_training)
{
    float start_time = static_cast<float>(glfwGetTime());

    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;
    std::vector<Mann::Matrix> MNN_d_weights = MNN_Weights;
    std::vector<Mann::Matrix> MNN_d_biases = MNN_Bias;

    for(int n = 0; n < iterations; n++) {
        // float avg_cost_bulk = 0.0f;
        for(int batch = 0; batch < image_data->mnist_images_data.size()/m_batch_size; batch++) {
            float batch_start_time = static_cast<float>(glfwGetTime());
            current_batch = batch;
            int correct_pred = 0;
            
            for (int j = 0; j < MNN_d_weights.size(); j++) {
                MNN_d_weights[j].nullMatrix();
                MNN_d_biases[j].nullMatrix();
            }

            int start = static_cast<int>(batch * m_batch_size);
            int end = static_cast<int>((batch + 1) * m_batch_size);

            for (int i = start; i < end; i++) {
                // if (i == 10)
                //     sample_image_label = image_data->mnist_labels_data[i][0]; // just a random number to initialize

                // load image data in network
                for (int j =0; j < MNN_Nodes[0].rows(); j++) {
                    MNN_Nodes[0][j][0] = image_data->mnist_images_data[i][j];
                }
                for (int j = 0; j < MNN_y.rows(); j++) {
                    MNN_y[j][0] = image_data->mnist_labels_data[i][j];
                }

                feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);

                Mann::Matrix MNN_cost = (MNN_Nodes[MNN_Nodes.size() - 1] - MNN_y);
                MNN_cost = MNN_cost ^ MNN_cost;
                float avg_cost = 0;
                for (int j = 0; j < MNN_cost.rows(); j++) {
                    avg_cost += MNN_cost[j][0];
                }

                if (IsPredictionCorrect(MNN_Nodes[MNN_Nodes.size() - 1], MNN_y)) correct_pred++;

                std::vector<std::vector<Mann::Matrix>> MNN_d_weights_biases = backPropagation(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias, MNN_y);
                for(int j = 0; j < MNN_d_weights.size(); j++) {
                    MNN_d_weights[j] = (MNN_d_weights[j] + MNN_d_weights_biases[0][j])/2;
                    MNN_d_biases[j] = (MNN_d_biases[j] + MNN_d_weights_biases[1][j])/2;
                }
            }


            // time to update the weights and biases
            training_threads_mutex.lock();
            for (int j = 0; j < MNN_Weights.size(); j++) {
                MNN_Weights[j] = MNN_Weights[j] - (MNN_d_weights[j] * m_learning_rate);
                MNN_Bias[j] = MNN_Bias[j] - (MNN_d_biases[j] * m_learning_rate);
            }
            training_threads_mutex.unlock();

            float end_time = static_cast<float>(glfwGetTime());
            m_total_training_time += (end_time - start_time);
            start_time = end_time;

            m_batch_accuracy = (static_cast<float>(correct_pred) / static_cast<float>(m_batch_size)) * 100.0f;
            m_batch_accuracy_history.push(m_batch_accuracy);
            if (m_batch_accuracy_history.size() - 1 > 100)
            {
                m_batch_accuracy_history.pop();
            }
            

            saveNetwork();

            float batch_end_time = static_cast<float>(glfwGetTime());
            float difference = m_averageTimePerBatch - (batch_end_time - batch_start_time);
            m_averageTimePerBatch -= difference * 0.05f; // moving average with weight 0.05

            if(is_training && !(*is_training)) {
                return;
            }
        }

        m_current_epoch += 1;
    }
}

/**
 * @brief Tests the neural network interactively with user-provided image indices.
 * @param images_data A vector of input image data for testing.
 * @param labels_data A vector of corresponding label data for testing.
 * @param filename The file containing the network configuration.
 */
Mann::Matrix MNNetwork::predictSingleImage(std::vector<double> &image_data)
{

    // loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    // load image data in network
    for (int j = 0; j < MNN_Nodes[0].rows(); j++) {
        MNN_Nodes[0][j][0] = image_data[j];
    }

    feedForward(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias);


    return MNN_Nodes[MNN_Nodes.size() - 1];
}

/**
 * @brief Tests the neural network using the provided dataset.
 */
float MNNetwork::testNetwork(Mnist::MnistData* image_data)
{

    // loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    training_threads_mutex.lock();
    std::vector<Mann::Matrix> MNN_NODES_COPY = MNN_Nodes;
    std::vector<Mann::Matrix> MNN_WEIGHTED_SUM_COPY = MNN_weighted_sum;
    std::vector<Mann::Matrix> MNN_WEIGHTS_COPY = MNN_Weights;
    std::vector<Mann::Matrix> MNN_BIAS_COPY = MNN_Bias;
    training_threads_mutex.unlock();

    float avg_cost_bulk = 0;
    int correct_pred = 0;

    for (int i = 0; i < image_data->mnist_images_data.size(); i++) {

        // load image data in network
        for (int j = 0; j < MNN_NODES_COPY[0].rows(); j++) {
            MNN_NODES_COPY[0][j][0] = image_data->mnist_images_data[i][j];
        }
        for (int j = 0; j < MNN_y.rows(); j++) {
            MNN_y[j][0] = image_data->mnist_labels_data[i][j];
        }

        feedForward(MNN_NODES_COPY, MNN_WEIGHTED_SUM_COPY, MNN_WEIGHTS_COPY, MNN_BIAS_COPY);

        Mann::Matrix MNN_cost = (MNN_NODES_COPY[MNN_NODES_COPY.size() - 1] - MNN_y);
        MNN_cost = MNN_cost ^ MNN_cost;
        float avg_cost = 0;
        for (int j = 0; j < MNN_cost.rows(); j++) {
            avg_cost += MNN_cost[j][0];
        }

        avg_cost_bulk += avg_cost;

        if (IsPredictionCorrect(MNN_NODES_COPY[MNN_NODES_COPY.size() - 1], MNN_y)) correct_pred++;
    }
    
    m_average_cost = avg_cost_bulk / static_cast<float>(image_data->mnist_images_data.size());
    return (static_cast<float>(correct_pred) / static_cast<float>(image_data->mnist_images_data.size())) * 100.0f;
}

/**
 * @brief Initializes the neural network with the specified layer sizes.
 * @param layers_size A vector specifying the size of each layer.
 * @param nodes A vector of matrices to store the node values in each layer.
 * @param weights A vector of matrices to store the weights between layers.
 * @param biases A vector of matrices to store the biases for each layer.
 */
void MNNetwork::initializeNetwork(NetworkInitialization* network_initialization)
{
    for(int i=0; i < network_initialization->layers_size.size(); i++)
    {
        network_initialization->nodes.emplace_back(Mann::Matrix(network_initialization->layers_size[i], 1));
    }

    for(int i=0; i < network_initialization->layers_size.size() - 1; i++)
    {
        network_initialization->weights.emplace_back(Mann::Matrix(network_initialization->layers_size[i + 1], network_initialization->layers_size[i]));
        network_initialization->biases.emplace_back(Mann::Matrix(network_initialization->layers_size[i + 1], 1));

        network_initialization->weights[i].randomize();
        network_initialization->biases[i].randomize();
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

bool MNNetwork::IsPredictionCorrect(const Mann::Matrix &output_layer, const Mann::Matrix &target)
{
    int predicted_label = 0;
    float max_value = output_layer[0][0];

    for (int i = 1; i < output_layer.rows(); i++) {
        if (output_layer[i][0] > max_value) {
            max_value = output_layer[i][0];
            predicted_label = i;
        }
    }

    int actual_label = 0;
    for (int i = 0; i < target.rows(); i++) {
        if (target[i][0] == 1.0f) {
            actual_label = i;
            break;
        }
    }

    return predicted_label == actual_label;
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
    NetworkInitialization* network_initialization = new NetworkInitialization{layers_size, d_nodes, d_weights, d_biases};
    initializeNetwork(network_initialization);
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
    std::ofstream file("../models/" + m_model_id + ".mms");
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving network: " << m_model_id << std::endl;
        return;
    }
    // Save Model Name
    file << m_model_name << "\n";
    // Save layers_size
    for (size_t i = 0; i < MNN_Layers_size.size(); ++i) {
        file << MNN_Layers_size[i] << (i + 1 < MNN_Layers_size.size() ? " " : "\n");
    }
    // Save Current Epoch
    file << m_current_epoch << "\n";
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
 * @param model_id The identifier of the model containing the network configuration.
 */
void MNNetwork::loadNetwork(const std::string &model_id)
{
    size_t layer_size;
    std::ifstream file(("../models/" + model_id + ".mms"));

    if (!file.is_open()) {
        std::cerr << "Error opening file for loading network: " << model_id << std::endl;
        return;
    }

    // get first line (Model name)
    std::getline(file, m_model_name);

    // get second line (layers size)
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    while (iss >> layer_size) {
        MNN_Layers_size.push_back(static_cast<size_t>(layer_size));
    }


    // Get third line (current epoch)
    file >> m_current_epoch;

    // Get fourth line (learning rate)
    file >> m_learning_rate;

    // Get fifth line (batch size)
    file >> m_batch_size;

    // Get sixth line (accuracy)
    file >> m_accuracy;

    // Get seventh line (total training time)
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

    file.close();
    loadHistoryData();
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
void MNNetwork::CreateNetwork(NetworkArchitecture* network_arch)
{

    std::string path = "../models/" + m_model_id + ".mms";
    std::ofstream outfile(path); // mandeep model storage

    if (!outfile)
    {
        std::cerr << "Error creating new file!" << std::endl;
        return;
    }

    // [783, hidden, 10]
    network_arch->network_initialization->layers_size.push_back(784); // input layer
    for (size_t i = 0; i < network_arch->hidden_layers_size.size(); i++) {
        network_arch->network_initialization->layers_size.push_back(network_arch->hidden_layers_size[i]); // hidden layers
    }
    network_arch->network_initialization->layers_size.push_back(10);  // output layer

    NetworkInitialization* network_initialization = new NetworkInitialization{network_arch->network_initialization->layers_size,
                    network_arch->network_initialization->nodes, network_arch->network_initialization->weights, network_arch->network_initialization->biases};
    initializeNetwork(network_initialization);
    saveNetwork();

    outfile.close();
    std::cout << "File created successfully!" << std::endl;

    saveHistoryData();
}

/**
 * @brief Saves image and label data to a file in a formatted manner.
 * @param image_data A vector containing the image data.
 * @param label_data A vector containing the corresponding label data.
 * @param model_id The identifier of the model to save the image and label data.
 */
void MNNetwork::saveImageDataToFile(Mnist::MnistData* image_data,
                                   const std::string& model_id)
{
    std::ofstream file("../models/" + model_id + ".mms");
    if (file.is_open())
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                // only 2 decimal points
                if (*image_data->mnist_images_data[i * 28 + j].data() < 0.1)
                {
                    file << ".....";
                }
                else
                {
                    file << std::fixed << std::setprecision(2) << image_data->mnist_images_data[i * 28 + j].data() << " ";
                }

                // file << image_data[i * 28 + j] << " ";
            }
            file << std::endl << std::endl;

        }
        file << "Label: ";
        for (int i = 0; i < 10; i++)
        {
            file << image_data->mnist_labels_data[i].data() << " ";
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

void MNNetwork::saveHistoryData()
{
    std::string Logfile = "Log_" + m_model_id + ".mml";
    std::ofstream file("../models/modelsLogData/" + Logfile);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving network: " << Logfile << std::endl;
        return;
    }
    // save m_accuracy_history
    for (size_t i = 0; i < m_accuracy_history.size(); ++i) {
        file << m_accuracy_history[i] << " ";
    }
    file << "\n";
    // save m_accuracy_testdata_history
    for (size_t i = 0; i < m_accuracy_testdata_history.size(); ++i) {
        file << m_accuracy_testdata_history[i] << " ";
    }
    file << "\n";
    // save m_average_cost_history
    for (size_t i = 0; i < m_average_cost_history.size(); ++i) {
        file << m_average_cost_history[i] << " ";
    }
    file << "\n";
    // save m_average_cost_testdata_history
    for (size_t i = 0; i < m_average_cost_testdata_history.size(); ++i) {
        file << m_average_cost_testdata_history[i] << " ";
    }
    file << "\n";
}

void MNNetwork::loadHistoryData()
{
    float accuracy_value;
    std::string Logfile = "Log_" + m_model_id + ".mml";
    std::ifstream file(("../models/modelsLogData/" + Logfile));

    if (!file.is_open()) {
        std::cerr << "Error opening file for loading network: " << Logfile << std::endl;
        saveHistoryData();
        std::cout << "Created new log file for network: " << Logfile << std::endl;
        return;
    }

    // get first line (m_accuracy_history)
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    while (iss >> accuracy_value) {
        m_accuracy_history.push_back(accuracy_value);
    }
    // get second line (m_accuracy_testdata_history)
    std::getline(file, line);
    std::istringstream iss2(line);
    while (iss2 >> accuracy_value) {
        m_accuracy_testdata_history.push_back(accuracy_value);
    }
    // get third line (m_average_cost_history)
    std::getline(file, line);
    std::istringstream iss3(line);
    while (iss3 >> accuracy_value) {
        m_average_cost_history.push_back(accuracy_value);
    }
    // get fourth line (m_average_cost_testdata_history)
    std::getline(file, line);
    std::istringstream iss4(line);
    while (iss4 >> accuracy_value) {
        m_average_cost_testdata_history.push_back(accuracy_value);
    }
}