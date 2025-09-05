#ifndef MANN_UI_H
#define MANN_UI_H

#include "utils.h"
#include "MNNetwork.h"
#include "mann.h"

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <array>
#include <random>
#include <stdexcept>

inline bool endsWithTxt(const std::string& str) {
    const std::string ext = ".mms";
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
    const char* cmd = "dir /b ..\\models\\*.mms";
#else
    const char* cmd = "ls ../models/*.mms 2> /dev/null";
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


inline std::string trim(const std::string& s)
{
    std::string str;
    for (const char c : s)
    {
        if (c != ' ') str+=c;
    }
    return str;
}

inline std::string getRandomModelName()
{
    std::ifstream file("../src/modelnames.txt");

    if (!file.is_open())
    {
        std::cerr << "Error: Can't Open File modelnames.txt";
        throw std::runtime_error("Failed to open modelnames.txt");
    }

    std::vector<std::string> lines;
    std::string line;

    while (std::getline(file, line))
    {
        std::string trimmed_line = trim(line);
        if (!trimmed_line.empty())
            lines.push_back(trimmed_line);
    }

    if (lines.empty())
    {
        std::cerr << "Error: modelnames.txt is empty" << std::endl;
        throw std::runtime_error("modelnames.txt is empty");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, lines.size() - 1);
    return lines[dis(gen)];
}

class MannUI
{
public:
    MannUI(GLFWwindow* window, float learning_rate, size_t iterations_rate, size_t batch_size);
    virtual ~MannUI();

    void Render();

private:
    GLFWwindow* window;
    std::stringstream outputText;
    float learning_rate;
    size_t iterations_rate;
    size_t batch_size;
};

#endif // MANN_UI_H