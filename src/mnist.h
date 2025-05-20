#pragma once

#include "mann.h"

using namespace std;

class Mnist
{
public:
    Mnist();
    ~Mnist();

    int ReverseInt (int i);
    void ReadMNISTimages(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr);
    void ReadMNISTlabels(int NumberOfImages, vector<vector<double>> &arr);
};