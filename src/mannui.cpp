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

#include "mannui.h"
#include "MNNetwork.h"

#include <assert.h>

/**
 * @brief Global vector of model filenames retrieved from the models directory.
 */
std::vector<std::string> filenames = getTxtFileNamesWithoutExtension();
char csv_buffer[256] = ""; // Buffer for csv input.
Mnist* mnist;

/**
 * @brief Global network to store models data.
 */
std::mutex resultsMutex;                                               ///< Mutex for synchronizing output in threaded operations.

std::vector<NetworkEntry> Networks; ///< Vector to store multiple neural network models.

/**
 * @brief Constructs a MannUI object and initializes ImGui with GLFW and OpenGL.
 * @param window The GLFW window for rendering the UI.
 * @param learning_rate The learning rate for neural network training.
 * @param iterations_rate The number of iterations for training.
 * @param batch_size The batch size for training.
 */
MannUI::MannUI(GLFWwindow *window, Mnist *mnist)
    : window(window)
{
    // Init ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    ImGui::StyleColorsDark();
    // SetModernDarkTheme();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ::mnist = mnist;
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
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

// PROTOTYPES
void ShowAvalModels(UIContext* ui_context);
void TrainingWindow(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread);
void NetworkConfigUI(NetworkConfiguration* network_configuration);

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

                float accuracy = entry.network->testNetwork(&mnist->mnist_testData);
                entry.network->m_accuracy_testdata = accuracy;
                entry.network->m_accuracy_testdata_history.push_back(accuracy);
                entry.calculatingAccuracy = false;

                accuracy = entry.network->testNetwork(&mnist->mnist_trainingData);
                entry.network->m_accuracy = accuracy;
                entry.network->m_accuracy_history.push_back(accuracy);

                MannLogger::info(outputText) << entry.modelName << " --- Accuracy: " << entry.network->m_accuracy_testdata << "%" << std::endl;
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

    UIContext* ui_context = new UIContext{outputText, shown_windows, selected_model};
    
    // Control Panel
    if (shown_windows.models_window)
    {
        ImGui::Begin("Models");
        ShowAvalModels(ui_context);
        ImGui::End();
    }

    if (shown_windows.training_window)
    {
        ImGui::Begin("Train Model");
        TrainingWindow(ui_context, is_training, training_thread, testing_thread);
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
void ShowAvalModels(UIContext* ui_context)
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
        else
        {
            ImGui::Text("Accuracy: %.2f%%", entry.network->m_accuracy_testdata);
        }

        // popup for model details
        if (ImGui::BeginPopupModal((entry.modelName + "Details").c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // Non Graphical things
            ui_context->selected_model = &entry;

            // Show model details here
            ImGui::Text("Model Name: %s", entry.modelName.c_str());
            ImGui::NewLine();
            // show layers size
            ImGui::Text("Layers: ");
            ImGui::SameLine();
            for (const size_t &layer : entry.network->MNN_Layers_size)
            {
                ImGui::Text("%zu,", layer);
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Text("Accuracy: %.2f%%", entry.network->m_accuracy);
            ImGui::Text("Learning Rate: %.6f", entry.network->m_learning_rate);
            ImGui::Text("Batch Size: %zu", entry.network->m_batch_size);
            ImGui::Text("Total Time Trained: %.2f", entry.network->m_total_training_time);

            ImGui::NewLine();
            // buttons (train, test network on data, test network by canvas)
            if (ImGui::Button("Train"))
            {
                // Train the network
                ui_context->shown_windows.training_window = true;
                ui_context->shown_windows.models_window = false;
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

        NetworkConfiguration* network_configuration = new NetworkConfiguration{hidden_layers, learning_rate, batch_size};
        NetworkConfigUI(network_configuration);

        if (ImGui::Button("Create"))
        {
            if (hidden_layers.empty())
            {
                // outputText << "Error: No layers added. Please add layers using the Add Layer button." << std::endl;
                MannLogger::error(ui_context->outputText) << "Error: No layers added. Please add layers using the Add Layer button." << std::endl;
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
                MannLogger::info(ui_context->outputText) << "Creating Model: " << modelName << " with layers [" << layersStr.str() << "], learning rate: " << learning_rate << ", batch size: " << batch_size << std::endl;
                // Create the network
                // MNNetwork network(modelName + ".mms", hidden_layers, learning_rate, batch_size);

                NetworkConfiguration* network_configuration = new NetworkConfiguration{hidden_layers, learning_rate, batch_size};
                Networks.push_back({modelName, new MNNetwork(modelName + ".mms", network_configuration)});

                auto &newEntry = Networks.back();

                // test the network
                std::thread test_thread([&newEntry, &ui_context]()
                {
                    // dont fucking touch this code at all costs, pata nhi kese chala h ye, bus chal rha h.
                    float accuracy = newEntry.network->testNetwork(&mnist->mnist_testData);
                    newEntry.network->m_accuracy_testdata = accuracy;
                    newEntry.network->m_accuracy_testdata_history.push_back(accuracy);

                    newEntry.calculatingAccuracy = false;

                    accuracy = newEntry.network->testNetwork(&mnist->mnist_trainingData);
                    newEntry.network->m_accuracy = accuracy;
                    newEntry.network->m_accuracy_history.push_back(accuracy);
                    // MannLogger::info(ui_context->outputText) << newEntry.modelName << " --- Accuracy: " << newEntry.network->m_accuracy << "%" << std::endl;
                });
                test_thread.detach();
                
                filenames.push_back(modelName);                
                // Reset inputs
                modelName.clear();
                
                // NetworkConfiguration* network_configuration = new NetworkConfiguration{hidden_layers, learning_rate, batch_size};
                
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
void NetworkConfigUI(NetworkConfiguration* network_configuration)
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
            network_configuration->hidden_layers.insert(network_configuration->hidden_layers.end(), new_layers.begin(), new_layers.end());
            csv_buffer[0] = '\0';
        }
    }

    // Display current hidden_layers with Remove buttons
    ImGui::Text("Current Layers:");
    for (size_t i = 0; i < network_configuration->hidden_layers.size(); ++i)
    {
        ImGui::PushID(i);
        ImGui::Text("Layer %zu: %zu", i + 1, network_configuration->hidden_layers[i]);
        ImGui::SameLine();
        if (ImGui::Button("Remove"))
        {
            network_configuration->hidden_layers.erase(network_configuration->hidden_layers.begin() + i);
            ImGui::PopID();
            continue;
        }
        ImGui::PopID();
    }

    ImGui::Text("Learning Rate");
    ImGui::Separator();
    if (ImGui::InputFloat("Learning Rate", &network_configuration->learning_rate, 0.001f, 0.1f, "%.4f", ImGuiInputTextFlags_CharsDecimal))
    {
        if (network_configuration->learning_rate < 0.0f)
            network_configuration->learning_rate = 0.0f;
        if (network_configuration->learning_rate > 1.0f)
            network_configuration->learning_rate = 1.0f;
    }

    ImGui::Text("Batch Size");
    ImGui::Separator();
    int batch_size_int = static_cast<int>(network_configuration->batch_size);
    if (ImGui::InputInt("Batch Size", &batch_size_int, 0, 0, ImGuiInputTextFlags_CharsNoBlank))
    {
        if (batch_size_int > 0)
            network_configuration->batch_size = static_cast<size_t>((batch_size_int > 200) ? 200 : batch_size_int);
    }
}


void TrainingWindow(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread)
{
    ImGui::Text("Training Window - Under Construction");
    ImGui::Separator();
    if (ui_context->selected_model)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (ImGui::TreeNodeEx("Model Details", node_flags))
        {
            ImGui::Text("Model Name: %s", ui_context->selected_model->modelName.c_str());
            ImGui::Text("Layers: ");
            ImGui::SameLine();
            for (const size_t &layer : ui_context->selected_model->network->MNN_Layers_size)
            {
                ImGui::Text("%zu,", layer);
                ImGui::SameLine();
            }
            ImGui::NewLine();
            ImGui::Text("Learning Rate: %.6f", ui_context->selected_model->network->m_learning_rate);
            ImGui::Text("Batch Size: %zu", ui_context->selected_model->network->m_batch_size);

            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Training Details", node_flags))
        {
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            ImGui::Text("Epoch #: %zu", ui_context->selected_model->network->m_current_epoch);
            ImGui::Text("Batch #: %d/%zu", ui_context->selected_model->network->current_batch, (mnist->mnist_trainingData.mnist_images_data.size() / ui_context->selected_model->network->m_batch_size));
            // ImGui::Text("Time per Batch: %.4f seconds", ui_context->selected_model->network->m_time_per_batch);

            // show total time trained in hr:min:sec format
            float total_time = ui_context->selected_model->network->m_total_training_time;
            int hours = static_cast<int>(total_time) / 3600;
            int minutes = (static_cast<int>(total_time) % 3600) / 60;
            float seconds = total_time - (hours * 3600) - (minutes * 60);
            ImGui::Text("Total Time Trained: %02d:%02d:%05.2f (hh:mm:ss)", hours, minutes, seconds);
            // ImGui::Text("Total Time Trained: %.2f min", ui_context->selected_model->network->m_total_training_time);
            ImGui::TreePop();
        }

        ImGui::Separator();
        ImGui::NewLine();

        if (ImGui::TreeNodeEx("Training Graphs Per Epoch", node_flags))
        {
            if(ImPlot::BeginPlot("Graph Per Epoch"))
            {
                ImPlot::SetupAxes("Epochs", "Accuracy (%)");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, std::max(20.0, static_cast<double>(ui_context->selected_model->network->m_accuracy_history.size())), ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 120.0, ImGuiCond_Always);

                ImPlot::PlotLine("Accuracy On Training Data", ui_context->selected_model->network->m_accuracy_history.data(), static_cast<int>(ui_context->selected_model->network->m_accuracy_history.size()));
                ImPlot::PlotLine("Accuracy On Test Data", ui_context->selected_model->network->m_accuracy_testdata_history.data(), static_cast<int>(ui_context->selected_model->network->m_accuracy_testdata_history.size()));
                
                ImPlot::EndPlot();
            }

            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Training Graphs Per Batch", node_flags))
        {
            if(ImPlot::BeginPlot("Graph Per Batch"))
            {
                ImPlot::SetupAxes("Batches", "Accuracy (%)");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, std::max(100.0, static_cast<double>(ui_context->selected_model->network->m_batch_accuracy_history.size())), ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 120.0, ImGuiCond_Always);

                ImPlot::PlotLine("Accuracy Per Batch", ui_context->selected_model->network->m_batch_accuracy_history.data(), static_cast<int>(ui_context->selected_model->network->m_batch_accuracy_history.size()));
                
                ImPlot::EndPlot();
            }

            ImGui::TreePop();
        }

        ImGui::NewLine();
        ImGui::Separator();

        // Add training controls and progress here
        if (!is_training) {
            if (ImGui::Button("Start Training"))
            {
                if(training_thread.joinable())
                    training_thread.join();

                if(testing_thread.joinable())
                    testing_thread.join();

                // Start training logic
                MannLogger::info(ui_context->outputText) << "Starting training for model: " << ui_context->selected_model->modelName << std::endl;
                
                
                training_thread = std::thread([ui_context, &is_training]() {
                    // Access selected_model via ui_context
                    if (ui_context && ui_context->selected_model && ui_context->selected_model->network) {
                        ui_context->selected_model->network->trainNetwork(1000, &mnist->mnist_trainingData, &is_training);
                    } else {
                        // Handle error: e.g., log to outputText if available
                        if (ui_context && ui_context->outputText) {
                            ui_context->outputText << "Error: Invalid ui_context or selected_model for training." << std::endl;
                        }
                    }
                });

                is_training = true;

                testing_thread = std::thread([ui_context, &is_training]() {
                    // This thread is used to predict accuracy while training for better UI experience
                    while (is_training)
                    {
                        if (ui_context && ui_context->selected_model && ui_context->selected_model->network) {
                            float accuracy = ui_context->selected_model->network->testNetwork(&mnist->mnist_trainingData);
                            ui_context->selected_model->network->m_accuracy = accuracy;
                            // Update accuracy history for UI graph
                            ui_context->selected_model->network->m_accuracy_history.push_back(ui_context->selected_model->network->m_accuracy);
                            MannLogger::info(ui_context->outputText) << "Network Current Accuracy On Training Data: " << ui_context->selected_model->network->m_accuracy << "%" << std::endl;

                            accuracy = ui_context->selected_model->network->testNetwork(&mnist->mnist_testData);
                            ui_context->selected_model->network->m_accuracy_testdata = accuracy;
                            // Update accuracy history for UI graph
                            ui_context->selected_model->network->m_accuracy_testdata_history.push_back(ui_context->selected_model->network->m_accuracy_testdata);
                            MannLogger::info(ui_context->outputText) << "Network Current Accuracy On Test Data: " << ui_context->selected_model->network->m_accuracy_testdata << "%" << std::endl;
                        }
                    }
                    
                });
            }
        }
        else {
            ImGui::Text("Model is training...");

            if(ImGui::Button("Stop Training"))
            {
                // Stop training logic
                is_training = false;

                MannLogger::info(ui_context->outputText) << "Training stopped for model: " << ui_context->selected_model->modelName << std::endl;
            }

            // Here you can add a progress bar or other indicators
        }
    }
    else
        ImGui::Text("No Model Selected");

    if(!is_training) {
    if (ImGui::Button("Close"))
        {
            ui_context->shown_windows.training_window = false;
            ui_context->shown_windows.models_window = true;
        }
    }
}

void SetModernDarkTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.ChildRounding = 0;
    style.GrabRounding = 0;
    style.FrameRounding = 2;
    style.PopupRounding = 0;
    style.ScrollbarRounding = 0;
    style.TabRounding = 2;
    style.WindowRounding = 0;
    style.FramePadding = { 4, 4 };
    style.WindowTitleAlign = { 0.0, 0.5 };
    style.ColorButtonPosition = ImGuiDir_Left;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = { 1.0f, 1.0f, 1.0f, 1.00f };
    colors[ImGuiCol_TextDisabled] = { 0.25f, 0.25f, 0.25f, 1.00f };
    colors[ImGuiCol_WindowBg] = { 0.09f, 0.09f, 0.09f, 0.94f };
    colors[ImGuiCol_ChildBg] = { 0.11f, 0.11f, 0.11f, 1.00f };
    colors[ImGuiCol_PopupBg] = { 0.11f, 0.11f, 0.11f, 0.94f };
    colors[ImGuiCol_Border] = { 0.07f, 0.08f, 0.08f, 1.00f };
    colors[ImGuiCol_BorderShadow] = { 0.00f, 0.00f, 0.00f, 0.00f };
    colors[ImGuiCol_FrameBg] = { 0.35f, 0.35f, 0.35f, 0.54f };
    colors[ImGuiCol_FrameBgHovered] = { 0.31f, 0.29f, 0.27f, 1.00f };
    colors[ImGuiCol_FrameBgActive] = { 0.40f, 0.36f, 0.33f, 0.67f };
    colors[ImGuiCol_TitleBg] = { 0.1f, 0.1f, 0.1f, 1.00f };
    colors[ImGuiCol_TitleBgActive] = { 0.3f, 0.3f, 0.3f, 1.00f };
    colors[ImGuiCol_TitleBgCollapsed] = { 0.0f, 0.0f, 0.0f, 0.61f };
    colors[ImGuiCol_MenuBarBg] = { 0.18f, 0.18f, 0.18f, 0.94f };
    // ... continue for other ImGuiCol_ enums
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
