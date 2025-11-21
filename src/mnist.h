/**
 * @file mnist.h
 * @brief Header file for the Mnist class, which handles reading MNIST dataset images and labels.
 * @author  Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 *
 * This file defines the Mnist class, responsible for reading and processing
 * MNIST dataset images and labels into a format suitable for neural network training.
 */

#pragma once

#include "mann.h"

/**
 * @class Mnist
 * @brief A class for reading and processing MNIST dataset images and labels.
 *
 * This class provides functionality to read MNIST image and label data from files
 * and convert them into vectors of doubles for use in neural network training.
 */

class Mnist
{
public:

    /** 
     * @brief Structure to hold MNIST training data.
     */
    struct Mnist_Data
    {
        std::vector<std::vector<double>> mnist_images_data;
        std::vector<std::vector<double>> mnist_labels_data;
    };

    /** 
     * @brief MNIST training data.
     */

    Mnist_Data mnist_trainingData;

    /** 
     * @brief MNIST test data.
     */
    Mnist_Data mnist_testData;

    /**
     * @brief Default constructor for the Mnist class.
     *
     * Initializes the Mnist object for reading MNIST dataset files.
     */
    Mnist();

    /**
     * @brief Destructor for the Mnist class.
     *
     * Cleans up resources used by the Mnist object.
     */
    ~Mnist();

    /**
     * @brief Reverses the byte order of an integer for handling endianness.
     * @param i The integer to reverse.
     * @return The integer with its byte order reversed.
     */
    int ReverseInt (int i);

    /**
     * @brief Reads MNIST image data from a file into a vector.
     * @param NumberOfImages The number of images to read.
     * @param DataOfAnImage The number of pixels per image (e.g., 784 for 28x28 images).
     * @param arr A vector of vectors to store the image data as doubles.
     * @param isTraining Boolean flag indicating whether to read training or test data.
     */
    void ReadMNISTimages(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<double>> &arr, bool isTraining);

    /**
     * @brief Reads MNIST label data from a file into a vector.
     * @param NumberOfImages The number of images (labels) to read.
     * @param arr A vector of vectors to store the label data as one-hot encoded doubles.
     * @param isTraining Boolean flag indicating whether to read training or test data.
     */
    void ReadMNISTlabels(int NumberOfImages, std::vector<std::vector<double>> &arr, bool isTraining);
};
