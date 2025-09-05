#ifndef MANN_UI_H
#define MANN_UI_H

#include "utils.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

inline bool endsWithTxt(const std::string& str) {
    const std::string ext = ".txt";
    if (str.length() >= ext.length()) {
        return (0 == str.compare(str.length() - ext.length(), ext.length(), ext));
    }
    return false;
}

inline std::string removeTxtExtension(const std::string& filename) {
    if (endsWithTxt(filename)) {
        return filename.substr(0, filename.length() - 4);
    }
    return filename;
}

inline std::vector<std::string> getTxtFileNamesWithoutExtension() {
    std::vector<std::string> filenames;

#ifdef _WIN32
    const char* cmd = "dir /b ..\\models\\*.txt";
#else
    const char* cmd = "ls ../models/*.txt 2> /dev/null";
#endif

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "Failed to run command\n";
        return filenames;
    }

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);

        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }

#ifdef _WIN32
        
#else
        size_t pos = line.find_last_of('/');
        if (pos != std::string::npos) {
            line = line.substr(pos + 1);
        }
#endif

        if (endsWithTxt(line)) {
            filenames.push_back(removeTxtExtension(line));
        }
    }

    pclose(pipe);
    return filenames;
}

class MannUI
{
public:
    MannUI(GLFWwindow* window, float learning_rate, size_t iterations_rate, size_t batch_size);
    virtual ~MannUI();

    void Render();

private:
    GLFWwindow* window;
    std::string outputText;
    float learning_rate;
    size_t iterations_rate;
    size_t batch_size;
};

#endif // MANN_UI_H