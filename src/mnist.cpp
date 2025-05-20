#include "mnist.h"
#include "mann.h"

using namespace std;

Mnist::Mnist() {};
Mnist::~Mnist() {};

int Mnist::ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void Mnist::ReadMNISTimages(int NumberOfImages, int DataOfAnImage, vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file ("dependencies/includes/t10k-images-idx3-ubyte",ios::binary);
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

void Mnist::ReadMNISTlabels(int NumberOfImages, vector<vector<double>> &arr)
{
    arr.resize(NumberOfImages,vector<double>(10));
    ifstream file ("dependencies/includes/t10k-labels-idx1-ubyte",ios::binary);
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