/**
 * @file mannui.cpp
 * @brief Implementation of the MannUI class for rendering a neural network GUI.
 * @author Jayansh Devgan, Mandeep Singh Warwal
 * @date 2025-09-06
 * @version 1.0
 *
 * This file contains the implementation of the MannUI class, which provides a
 * graphical user interface using ImGui and GLFW for interacting with the neural
 * network. It includes functionality for displaying available models, creating new
 * models, and rendering output.
 */

#include "mannui.hpp"

/**
 * @brief Global vector of model filenames retrieved from the models directory.
 */
std::vector<std::string> filenames = getTxtFileNamesWithoutExtension();
char csv_buffer[256] = ""; // Buffer for csv input.

/**
 * @brief Global network to store models data.
 */
std::vector<std::vector<double>> mnist_images_data, mnist_labels_data; ///< MNIST data for training and testing.
std::mutex resultsMutex;                                               ///< Mutex for synchronizing output in threaded operations.

std::vector<NetworkEntry> Networks; ///< Vector to store multiple neural network models.

/**
 * @brief Constructs a MannUI object and initializes ImGui with GLFW and OpenGL.
 * @param window The GLFW window for rendering the UI.
 * @param learning_rate The learning rate for neural network training.
 * @param iterations_rate The number of iterations for training.
 * @param batch_size The batch size for training.
 */
MannUI::MannUI(GLFWwindow *window, std::vector<std::vector<double>> mnist_images_data, std::vector<std::vector<double>> mnist_labels_data)
    : window(window)
{
    // Init ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ::mnist_images_data = mnist_images_data;
    ::mnist_labels_data = mnist_labels_data;
}

/**
 * @brief Destructor for the MannUI class.
 *
 * Cleans up ImGui and its backends.
 */
inline MannUI::~MannUI()
{

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// PROTOTYPES
void ShowAvalModels(std::stringstream &outputText, bool &show_training_window);

void NetworkConfigUI(std::vector<size_t> &hidden_layers, float &learning_rate, size_t &batch_size);

/**
 * @brief Renders the ImGui-based user interface.
 *
 * Sets up a new ImGui frame, renders the output window and control panel,
 * and handles docking and rendering of draw data.
 */
void MannUI::Render(std::stringstream &outputText)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGuiWindowFlags host_window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                         ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    // Begin a full-screen window to hold the dockspace
    ImGui::Begin("DockSpace Demo", nullptr, host_window_flags);

    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    static bool first_time = true;

    if (first_time)
    {
        first_time = false;
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);
        // Split dockspace into two nodes: left and right
        ImGuiID dock_main_id = dockspace_id;
        ImGuiID dock_id_left;
        ImGuiID dock_id_right;
        ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.7f, &dock_id_left, &dock_id_right);
        // Dock your windows into the nodes
        ImGui::DockBuilderDockWindow("Models", dock_id_left);
        ImGui::DockBuilderDockWindow("Train Model", dock_id_left);
        ImGui::DockBuilderDockWindow("Output", dock_id_right);
        // Finish dock builder
        ImGui::DockBuilderFinish(dockspace_id);

        // load the networks
        for (const std::string &name : filenames)
        {
            Networks.emplace_back(name, new MNNetwork(name + ".mms"));
        }

        MannLogger::info(outputText) << "Calculating model accuracy:" << std::endl;
        for (auto &entry : Networks)
        {
            std::thread test_thread([&entry, &outputText]()
            {
                entry.calculatingAccuracy = true;
                float accuracy = entry.network->testNetwork(::mnist_images_data, ::mnist_labels_data);
                entry.accuracy = accuracy;

                entry.accuracyAvailable = true;

                MannLogger::info(outputText) << entry.modelName << " --- Accuracy: " << accuracy << "%" << std::endl;
                // std::lock_guard<std::mutex> lock(resultsMutex);
            });
            test_thread.detach();
        }
    }

    ImGui::End();

    // Output Window
    ImGui::Begin("Output");
    std::stringstream tempStream;
    tempStream << outputText.str(); // Copy current content
    std::string line;
    while (std::getline(tempStream, line))
    {
        if (line.empty())
            continue;
        // Find the second occurrence of "[" to get the log level (e.g., [DEBUG], [INFO])
        size_t firstBracket = line.find("[");
        size_t secondBracket = line.find("[", firstBracket + 1);
        size_t thirdBracket = line.find("[", secondBracket + 1);
        size_t thirdBracketEnd = line.find("]", thirdBracket + 1);
        if (thirdBracket != std::string::npos && thirdBracketEnd != std::string::npos)
        {
            std::string levelStr = line.substr(thirdBracket + 1, thirdBracketEnd - thirdBracket - 1);
            ImVec4 color = IMGUI_COLOR_INFO; // Default to INFO color
            if (levelStr == "DEBUG")
                color = IMGUI_COLOR_DEBUG;
            else if (levelStr == "INFO")
                color = IMGUI_COLOR_INFO;
            else if (levelStr == "WARN")
                color = IMGUI_COLOR_WARN;
            else if (levelStr == "ERROR")
                color = IMGUI_COLOR_ERROR;

            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::TextWrapped("%s", line.c_str());
            ImGui::PopStyleColor();
        }
        else
            ImGui::TextWrapped("%s", line.c_str());
    }
    if (ImGui::Button("Clear Output"))
    {
        outputText.str("");
        outputText.clear();
    }
    ImGui::End();

    // Control Panel
    ImGui::Begin("Models");
    ShowAvalModels(outputText, show_training_window);
    ImGui::End();

    if(show_training_window)
    {
        ImGui::Begin("Train Model");
        ImGui::Text("Training in progress...");
        // Add training progress UI elements here
        if (ImGui::Button("Close"))
        {
            show_training_window = false;
        }
        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

/**
 * @brief Parses a CSV string into a std::vector<size_t> with exactly three values.
 * @param input The input CSV string.
 * @param output The vector to store parsed integers.
 * @return True if exactly three valid non-negative integers were parsed, false otherwise.
 */
bool ParseCSVToHiddenLayers(const std::string &input, std::vector<size_t> &output)
{
    output.clear();
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        try
        {
            size_t value = std::stoul(token);
            output.push_back((value > 699) ? 699 : value);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            return false;
        }
    }
    return !output.empty();
}

/**
 * @brief Displays available models and a button to create a new model.
 * @param outputText A stringstream to append output messages for display in the UI.
 */
void ShowAvalModels(std::stringstream &outputText, bool &show_training_window)
{
    // MannLogger::info(outputText) << "Available Models:" << std::endl;
    for (auto &entry : Networks)
    {
        ImGui::PushID(entry.modelName.c_str());
        
        if (ImGui::Button(entry.modelName.c_str()))
        {
            // ... show the details of the model in a popup
            ImGui::OpenPopup((entry.modelName + "Details").c_str());
        }
        
        ImGui::SameLine();

        if (entry.calculatingAccuracy)
        {
            if (entry.accuracyAvailable)
            {
                ImGui::Text("%.2f%%", entry.accuracy);
            }
            else
            {
                static const char *spinnerFrames[] = {".", "..", "..."};
                static int spinnerIndex = 0;
                static float lastTime = 0.0f;

                float currentTime = ImGui::GetTime();

                if (currentTime - lastTime > 0.4f)
                {
                    spinnerIndex = (spinnerIndex + 1) % 3;
                    lastTime = currentTime;
                }

                ImGui::Text("%s", spinnerFrames[spinnerIndex]);
            }
        }
        else if(entry.accuracyAvailable)
        {
            ImGui::Text("Accuracy: %.2f%%", entry.accuracy);
        }
        else{
            MannLogger::error(outputText) << "FUCKING FUCK ERROR!";
        }


        // popup for model details
        if (ImGui::BeginPopupModal((entry.modelName + "Details").c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // Show model details here
            ImGui::Text("Model Name: %s", entry.modelName.c_str());
            ImGui::NewLine();
            // show layers size
            ImGui::Text("Layers: ");
            ImGui::SameLine();
            for (const auto& layer : entry.network->MNN_Layers_size) {
                ImGui::Text("%zu,", layer);
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Text("Accuracy: %.2f%%", entry.accuracy);
            ImGui::Text("Learning Rate: %.6f", entry.network->m_learning_rate);
            ImGui::Text("Batch Size: %zu", entry.network->m_batch_size);
            ImGui::Text("Total Time Trained: %.2f", entry.network->m_total_training_time);

            ImGui::NewLine();
            //buttons (train, test network on data, test network by canvas)
            if (ImGui::Button("Train"))
            {
                // Train the network
                show_training_window = true;
                ImGui::CloseCurrentPopup();
                // open a new window where we can monitor training progress
            }
            ImGui::SameLine();
            if (ImGui::Button("Test on Data"))
            {
                // Test the network on data
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Test by Canvas"))
            {
                // Test the network by canvas
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Close"))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    if (ImGui::Button("Create New Model"))
    {
        ImGui::OpenPopup("CreateModelPopup");
    }

    if (ImGui::BeginPopupModal("CreateModelPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Enter new model details here.");

        static std::string modelName = "";
        static std::vector<size_t> hidden_layers{50, 10};
        static float learning_rate = 0.01f;
        static size_t batch_size = 32;

        // Buffer for model name input
        static char modelNameBuffer[256] = "";
        if (ImGui::InputText("Model Name", modelNameBuffer, sizeof(modelNameBuffer)))
        {
            modelName = modelNameBuffer;
        }

        NetworkConfigUI(hidden_layers, learning_rate, batch_size);

        if (ImGui::Button("Create"))
        {
            if (hidden_layers.empty())
            {
                // outputText << "Error: No layers added. Please add layers using the Add Layer button." << std::endl;
                MannLogger::error(outputText) << "Error: No layers added. Please add layers using the Add Layer button." << std::endl;
            }
            else
            {
                if (modelName.empty())
                    modelName = getRandomModelName();
                std::stringstream layersStr;
                for (size_t i = 0; i < hidden_layers.size(); ++i)
                {
                    layersStr << hidden_layers[i];
                }
                MannLogger::info(outputText) << "Creating Model: " << modelName << " with layers [" << layersStr.str() << "], learning rate: " << learning_rate << ", batch size: " << batch_size << std::endl;
                // Create the network
                MNNetwork network(modelName, hidden_layers, learning_rate, batch_size);
                Networks.push_back({modelName, new MNNetwork(modelName + ".mms")});

                filenames.push_back(modelName);

                // Reset inputs
                modelName.clear();
                hidden_layers.clear();
                learning_rate = 0.01f;
                batch_size = 32;
                modelNameBuffer[0] = '\0';
                csv_buffer[0] = '\0';

                ImGui::CloseCurrentPopup();
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel"))
        {
            // Reset inputs
            modelName.clear();
            hidden_layers.clear();
            learning_rate = 0.01f;
            batch_size = 32;
            modelNameBuffer[0] = '\0';
            csv_buffer[0] = '\0';
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

/**
 * @brief Renders UI for configuring neural network parameters.
 * @param hidden_layers Vector to store hidden layer sizes.
 * @param learning_rate Learning rate for the network.
 * @param batch_size Batch size for training.
 */
void NetworkConfigUI(std::vector<size_t> &hidden_layers, float &learning_rate, size_t &batch_size)
{
    ImGui::Text("Hidden Layers");
    ImGui::Separator();

    ImGui::InputText("Layers", csv_buffer, sizeof(csv_buffer), ImGuiInputTextFlags_CharsNoBlank);

    if (ImGui::Button("Add Layers"))
    {
        std::string input(csv_buffer);
        std::vector<size_t> new_layers;
        if (ParseCSVToHiddenLayers(input, new_layers))
        {
            hidden_layers.insert(hidden_layers.end(), new_layers.begin(), new_layers.end());
            csv_buffer[0] = '\0';
        }
    }

    // Display current hidden_layers with Remove buttons
    ImGui::Text("Current Layers:");
    for (size_t i = 0; i < hidden_layers.size(); ++i)
    {
        ImGui::PushID(i);
        ImGui::Text("Layer %zu: %zu", i + 1, hidden_layers[i]);
        ImGui::SameLine();
        if (ImGui::Button("Remove"))
        {
            hidden_layers.erase(hidden_layers.begin() + i);
            ImGui::PopID();
            continue;
        }
        ImGui::PopID();
    }

    ImGui::Text("Learning Rate");
    ImGui::Separator();
    if (ImGui::InputFloat("Learning Rate", &learning_rate, 0.001f, 0.1f, "%.4f", ImGuiInputTextFlags_CharsDecimal))
    {
        if (learning_rate < 0.0f)
            learning_rate = 0.0f;
        if (learning_rate > 1.0f)
            learning_rate = 1.0f;
    }

    ImGui::Text("Batch Size");
    ImGui::Separator();
    int batch_size_int = static_cast<int>(batch_size);
    if (ImGui::InputInt("Batch Size", &batch_size_int, 0, 0, ImGuiInputTextFlags_CharsNoBlank))
    {
        if (batch_size_int > 0)
            batch_size = static_cast<size_t>((batch_size_int > 200) ? 200 : batch_size_int);
    }
}

// GLuint LoadTexture(const std::string &filepath)
// {
//     int width, height, channels;
//     unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
//     if (!data)
//     {
//         outputText += "Failed to load image: " + filepath + "\n";
//         return 0;
//     }

//     GLuint textureID;
//     glGenTextures(1, &textureID);
//     glBindTexture(GL_TEXTURE_2D, textureID);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

//     stbi_image_free(data);
//     return textureID;
// }