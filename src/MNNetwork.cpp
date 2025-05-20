#include "MNNetwork.h"
#include "mann.h"

MNNetwork::MNNetwork() {};
MNNetwork::~MNNetwork() {};


void MNNetwork::trainNetwork(const size_t iterations, const size_t batch_size, std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename, float learning_rate)
{
    std::cout << "Training the network..." << std::endl;

    std::vector<size_t> MNN_Layers_size;
    std::vector<Mann::Matrix> MNN_Nodes, MNN_Weights, MNN_Bias;
    
    loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;
    
    std::vector<std::vector<Mann::Matrix>> MNN_d_weights_arr;
    std::vector<std::vector<Mann::Matrix>> MNN_d_biases_arr;
    for(int b=0; b < batch_size; b++) {
        MNN_d_weights_arr.push_back(MNN_Weights);
        MNN_d_biases_arr.push_back(MNN_Bias);
    }

    

    for(int n = 0; n < iterations; n++) {
        float avg_cost_bulk = 0;
        for(int batch = 0; batch < images_data.size()/batch_size; batch++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int b=0; b < batch_size; b++) {
                for (int j = 0; j < MNN_d_weights_arr[0].size(); j++) {
                    MNN_d_weights_arr[b][j].nullMatrix();
                    MNN_d_biases_arr[b][j].nullMatrix();
                }
            }
            std::function<void(int)> trainOneCycle = [&](int i) {
                std::cout << i << std::endl;
                
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
                avg_cost_bulk = (avg_cost_bulk + avg_cost) / 2;

                std::vector<std::vector<Mann::Matrix>> MNN_d_weights_biases = backPropagation(MNN_Nodes, MNN_weighted_sum, MNN_Weights, MNN_Bias, MNN_y);
                for(int j = 0; j < MNN_d_weights_arr[0].size(); j++) {
                    std::cout << "yoo" << std::endl;
                    MNN_d_weights_arr[i%batch_size][j] = MNN_d_weights_biases[0][j]/2;
                    MNN_d_biases_arr[i%batch_size][j] = MNN_d_weights_biases[1][j]/2;
                }
            };


            std::vector<std::future<void>> threads;
            for (int i = batch*batch_size; i < (batch+1)*batch_size+1; i++) {
                threads.emplace_back(std::async(std::launch::async, trainOneCycle, i));
            }
            for (auto& thread : threads) {
                thread.wait();
            }



            // calculte the average of MNN_d_weights_arr
            std::vector<Mann::Matrix> MNN_d_weights = MNN_d_weights_arr[100000];
            std::vector<Mann::Matrix> MNN_d_biases = MNN_d_biases_arr[0];
            for (std::vector<Mann::Matrix> &d_weights : MNN_d_weights_arr) {
                for (int j = 0; j < d_weights.size(); j++) {
                    MNN_d_weights[j] = (MNN_d_weights[j] + d_weights[j]) / 2;
                }
            }
            for (std::vector<Mann::Matrix> &d_biases : MNN_d_biases_arr) {
                for (int j = 0; j < d_biases.size(); j++) {
                    MNN_d_biases[j] = (MNN_d_biases[j] + d_biases[j]) / 2;
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "Elapsed time: " << std::fixed << std::setprecision(6) << duration.count()/1000000.0 << " seconds" << std::endl;
            


            // time to update the weights and biases
            for (int j = 0; j < MNN_Weights.size(); j++) {
                MNN_Weights[j] = MNN_Weights[j] - (MNN_d_weights[j] * learning_rate);
                MNN_Bias[j] = MNN_Bias[j] - (MNN_d_biases[j] * learning_rate);
            }
        }
        float accuracy = (10 - avg_cost_bulk) * 10;
        std::cout << "Iteration: " << n+1 << ", Average Cost: " << avg_cost_bulk << ", Accuracy: " << accuracy << std::endl;

        saveNetwork(MNN_Layers_size ,MNN_Weights, MNN_Bias, filename);
    }
}

void MNNetwork::testNetworkByUser(std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename)
{
    std::vector<size_t> MNN_Layers_size;
    std::vector<Mann::Matrix> MNN_Nodes, MNN_Weights, MNN_Bias;
    
    loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
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
            std::cout << std::endl;
            std::cout << "Predicted Labels: ";
            for (int j = 0; j < MNN_Nodes[MNN_Nodes.size() - 1].rows(); j++) {
                std::cout << MNN_Nodes[MNN_Nodes.size() - 1][j][0] << " ";
            }
            std::cout << std::endl;
            std::cout << "Accuracy: " << (10 - avg_cost) * 10 << "%" << std::endl << std::endl << std::endl;

            // print the image
            saveImageDataToFile(images_data[index], labels_data[index], "test_image.txt");
        }
    }
}

void MNNetwork::testNetwork(std::vector<std::vector<double>> &images_data, std::vector<std::vector<double>> &labels_data, const std::string &filename)
{
    std::cout << "Testing the network..." << std::endl;

    std::vector<size_t> MNN_Layers_size;
    std::vector<Mann::Matrix> MNN_Nodes, MNN_Weights, MNN_Bias;
    
    loadNetwork(MNN_Layers_size, MNN_Nodes, MNN_Weights, MNN_Bias, filename);
    Mann::Matrix MNN_y(MNN_Layers_size[MNN_Layers_size.size()-1], 1);
    std::vector<Mann::Matrix> MNN_weighted_sum = MNN_Bias;

    float avg_cost_bulk = 0;

    for (int i = 0; i < images_data.size(); i++) {
        // load image data in network
        for (int j = 0; j < MNN_Nodes[0].rows(); j++) {
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

        avg_cost_bulk = (avg_cost_bulk + avg_cost) / 2;
    }
    std::cout << std::endl;
    std::cout << "Accuracy: " << (10 - avg_cost_bulk) * 10 << "%" << std::endl << std::endl << std::endl;
}

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

void MNNetwork::saveNetwork(const std::vector<size_t> &layers_size, const std::vector<Mann::Matrix> &weights, const std::vector<Mann::Matrix> &biases, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving network: " << filename << std::endl;
        return;
    }
    // Save layers_size
    for (size_t i = 0; i < layers_size.size(); ++i) {
        file << layers_size[i] << (i + 1 < layers_size.size() ? " " : "\n");
    }
    // Save weights
    for (Mann::Matrix weight : weights) {
        file << weight;
    }
    // Save biases
    for (Mann::Matrix bias : biases) {
        file << bias;
    }
}

void MNNetwork::loadNetwork(std::vector<size_t>& layers_size,  std::vector<Mann::Matrix> &nodes, std::vector<Mann::Matrix> &weights, std::vector<Mann::Matrix> &biases, const std::string &filename)
{
    size_t layer_size;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "File Not Found. Creating a new neural Network" << std::endl;
        // extract layer size from filename :- MNN_Network_784_50_20_10.txt
        std::string layer_size_str = filename.substr(12, filename.find(".") - 12);
        for (char &c : layer_size_str) {
            if (c == '_') {
                c = ' ';
            }
        }
        std::istringstream iss(layer_size_str);
        while (iss >> layer_size) {
            layers_size.push_back(static_cast<size_t>(layer_size));
        }

        // Initialize network
        initializeNetwork(layers_size, nodes, weights, biases);
    }
    else {
        // get first line
        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);
        while (iss >> layer_size) {
            layers_size.push_back(static_cast<size_t>(layer_size));
        }
        // Load weights
        for (size_t i = 0; i < layers_size.size() - 1; ++i) {
            Mann::Matrix weight(layers_size[i + 1], layers_size[i]);
            for (size_t j = 0; j < layers_size[i + 1]; ++j) {
                for (size_t k = 0; k < layers_size[i]; ++k) {
                    file >> weight[j][k];
                }
            }
            weights.push_back(weight);
        }
        // Load biases
        for (size_t i = 0; i < layers_size.size() - 1; ++i) {
            Mann::Matrix bias(layers_size[i + 1], 1);
            for (size_t j = 0; j < layers_size[i + 1]; ++j) {
                file >> bias[j][0];
            }
            biases.push_back(bias);
        }

        // Initialize nodes
        for (size_t i = 0; i < layers_size.size(); ++i) {
            nodes.emplace_back(Mann::Matrix(layers_size[i], 1));
        }
    }
}

void MNNetwork::saveImageDataToFile(const std::vector<double>& image_data, const std::vector<double>& lable_data, const std::string& filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                // only 2 decimal points
                if (image_data[i * 28 + j] == 0)
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