/**
 * @file mann_ui.h
 * @brief Header file for the MannUI class and utility functions for managing neural network UI.
 * @author JayanshDevgan
 * @date 2025-09-06
 * @version 1.0
 *
 * This file defines the MannUI class, which provides a graphical user interface for
 * interacting with the neural network, along with utility functions for handling
 * model file names and random model name generation.
 */

#ifndef MANN_UI_H
#define MANN_UI_H

#include "utils.h"
#include "MNNetwork.h"
#include "mann.h"
#include "mannlogger.hpp"
#include "mnist.h"
#include "mannPopup.hpp"

#include <iostream>
#include <memory>
#include <future>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <array>
#include <random>
#include <stdexcept>

/**
 * @brief Platform-specific includes for file system operations.
 */
#ifdef _WIN32
    #include <cstdio>
    #define popen _popen
    #define pclose _pclose
#else
    #include <cstdio>
#endif

/**
 * @brief Checks if a string ends with the ".mms" extension.
 * @param str The string to check.
 * @return True if the string ends with ".mms", false otherwise.
 */
inline bool endsWithTxt(const std::string& str) {
    const std::string ext = ".mms";
    if (str.length() >= ext.length()) {
        return (0 == str.compare(str.length() - ext.length(), ext.length(), ext));
    }
    return false;
}

/**
 * @brief Removes the ".mms" extension from a filename.
 * @param filename The filename to process.
 * @return The filename without the ".mms" extension, or the original filename if no extension.
 */
inline std::string removeTxtExtension(const std::string& filename) {
    if (endsWithTxt(filename)) {
        return filename.substr(0, filename.length() - 4);
    }
    return filename;
}

/**
 * @brief Retrieves a list of model filenames without the ".mms" extension from the models directory.
 * @return A vector of model filenames without extensions.
 */
inline std::vector<std::string> getTxtFileNamesWithoutExtension() {
    std::vector<std::string> filenames;

#ifdef _WIN32
    const char* cmd = "dir /b ..\\..\\models\\*.mms";
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

/**
 * @brief Removes whitespace from a string.
 * @param s The input string to trim.
 * @return The string with all spaces removed.
 */
inline std::string trim(const std::string& s)
{
    std::string str;
    for (const char c : s)
    {
        if (c != ' ') str+=c;
    }
    return str;
}

/**
 * @brief Retrieves a random model name from a file.
 * @return A randomly selected model name from modelnames.txt.
 * @throws std::runtime_error If the file cannot be opened or is empty.
 */
inline std::string getRandomModelName()
{
#ifdef _WIN32
    std::ifstream file("../../src/modelnames.txt");
#else
    std::ifstream file("../src/modelnames.txt");
#endif

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

// Extern declarations for global variables
extern std::vector<std::string> filenames;
extern char csv_buffer[256];
extern Mnist* mnist;

struct NetworkEntry
{
    std::string modelName; ///< Name of the neural network model.
    MNNetwork* network;    ///< Pointer to neural network instance.

    NetworkEntry(const std::string& name, MNNetwork* net) : modelName(name), network(net) {}

    NetworkEntry(const NetworkEntry&) = delete;
    NetworkEntry(NetworkEntry&&) noexcept = default;
    NetworkEntry& operator=(const NetworkEntry&) = default;
    NetworkEntry& operator=(NetworkEntry&&) noexcept = default;

    ~NetworkEntry() = default;

    bool calculatingAccuracy = true;
};

extern std::vector<NetworkEntry> Networks; ///< Vector to store multiple neural network models.

/**
 * @class MannUI
 * @brief A class for managing the graphical user interface for the neural network.
 *
 * This class integrates with GLFW and ImGui to provide a user interface for
 * interacting with the MNNetwork class, allowing configuration of training
 * parameters and visualization of results.
 */
class MannUI
{
public:
    /**
     * @brief Constructs a MannUI object with specified parameters.
     * @param window The GLFW window for rendering the UI.
     * @param learning_rate The learning rate for neural network training.
     * @param iterations_rate The number of iterations for training.
     * @param batch_size The batch size for training.
     */
    MannUI(GLFWwindow* window, Mnist* mnist);

    /**
     * @brief Destructor for the MannUI class.
     *
     * Cleans up resources used by the UI.
     */
    virtual ~MannUI();

    /**
     * @brief Renders the UI using ImGui and GLFW.
     * @param outputText A stringstream to append output messages for display in the UI.
     */
    void Render(std::stringstream &outputText);
    void SetModernDarkTheme();

private:
    GLFWwindow* window;              ///< The GLFW window for rendering the UI.
    // bool show_models_window = true;    ///< Flag to show/hide the models window.
    // bool show_training_window = false;  ///< Flag to show/hide the training window.
    NetworkEntry* selected_model = nullptr;       ///< Pointer to the selected neural network model.

public:
    std::stringstream outputText;    ///< Stream for capturing UI output text.
    enum Shown_Windows
    {
        MODELS_WINDOW,
        TRAINING_WINDOW,
        TESTING_DATA_WINDOW,
        TESTING_CANVAS_WINDOW,
        NETWORK_VISUALIZER_WINDOW,
    } shown_windows_enum = NETWORK_VISUALIZER_WINDOW;

    //models_window specific variables

    //training window specific variables
    bool is_training = false;
    std::thread training_thread;
    std::thread testing_thread;

    struct OpenPopup
    {
        std::string name;
        bool to_open = false;
    };
    OpenPopup open_popup;
};

struct UIContext {
  std::stringstream &outputText;
  MannUI::Shown_Windows &shown_windows_enum;
  NetworkEntry* &selected_model;
  MannUI::OpenPopup &open_popup;
};


#endif // MANN_UI_H
