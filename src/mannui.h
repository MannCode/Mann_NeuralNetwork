#ifndef MANN_UI_H
#define MANN_UI_H

#include "utils.h"

#include <string>

// #define STB_IMAGE_IMPLEMENTATION
// #include "../dependencies/stb_image.h"

class MannUI
{
public:
    MannUI(GLFWwindow* window, float learning_rate, size_t iterations_rate, size_t batch_size);
    ~MannUI();

    void Render();

private:
    GLFWwindow* window;
    std::string outputText;
    float learning_rate;
    size_t iterations_rate;
    size_t batch_size;
};

#endif // MANN_UI_H