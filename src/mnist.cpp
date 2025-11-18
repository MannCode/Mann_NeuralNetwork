/**
 * @file Mnist.cpp
 * @brief Implementation of the Mnist class for reading MNIST dataset images and labels.
 * @author Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 *
 * This file contains the implementation of the Mnist class, which provides
 * functionality for reading MNIST image and label data from binary files
 * and converting them into vectors of doubles for neural network processing.
 */


#include "mnist.h"

/**
 * @brief Default constructor for the Mnist class.
 *
 * Initializes the Mnist object for reading MNIST dataset files.
 */
Mnist::Mnist() {};

/**
 * @brief Destructor for the Mnist class.
 *
 * Cleans up resources used by the Mnist object.
 */
Mnist::~Mnist() {};

/**
 * @brief Reverses the byte order of an integer to handle endianness.
 * @param i The integer to reverse.
 * @return The integer with its byte order reversed.
 */
int Mnist::ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

/**
 * @brief Reads MNIST image data from a binary file into a vector.
 * @param NumberOfImages The number of images to read.
 * @param DataOfAnImage The number of pixels per image (e.g., 784 for 28x28 images).
 * @param arr A vector of vectors to store the image data as normalized doubles.
 */
void Mnist::ReadMNISTimages(int NumberOfImages, int DataOfAnImage, std::vector<std::vector<double>> &arr)
{
    arr.resize(NumberOfImages, std::vector<double>(DataOfAnImage));
    std::ifstream file ("dependencies/includes/t10k-images-idx3-ubyte", std::ios::binary);
    if (file.is_open())
    {

        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);


        for(int i=0;i<number_of_images;i++)
        {
            for(int s=0;s<n_rows*n_cols;s++)
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                arr[i][s]= (double)temp/255.0;
            }
        }
    }
}

/**
 * @brief Reads MNIST label data from a binary file into a vector.
 * @param NumberOfImages The number of images (labels) to read.
 * @param arr A vector of vectors to store the label data as one-hot encoded doubles.
 */
void Mnist::ReadMNISTlabels(int NumberOfImages, std::vector<std::vector<double>> &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(10));
    std::ifstream file ("dependencies/includes/t10k-labels-idx1-ubyte", std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);

        for(int i=0;i<number_of_images;i++)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            arr[i][temp]=1;
        }
    }
}
