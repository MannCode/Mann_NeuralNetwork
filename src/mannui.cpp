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
// #include "SystemProfiler.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <numbers>
#include <assert.h>

#define IM_COL_RED(x) IM_COL32(234, 100, 0, x)
#define IM_COL_BLUE(x) IM_COL32(0, 182, 236, x)
#define IM_COL_GREEN(x) IM_COL32(0, 255, 0, x)
#define IM_COL_WHITE(x) IM_COL32(255, 255, 255, x)
 
inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

inline ImU32 GetFadingRed(float t)
{
    float fade = 0.08f * (1.0f + sinf(t * 3.14159265358979323846));

    float (*lerp_f)(float, float, float) = [](float a, float b, float t) {
        return a + (b - a) * t;
    };

    int alpha = static_cast<int>(lerp_f(0.f, 255.f, fade));
    return IM_COL32(255, 0, 0, alpha);
}

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
void LogWindow(std::stringstream &outputText);
void ProfilerWindow(std::stringstream &outputText);
void ShowAvalModels(UIContext* ui_context);
void TrainingWindow_1(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread);
void CreateTrainingThreads(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread);
void TrainingWindow_2(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread);
void TestingWindowData_1(UIContext* ui_context, Mann::Matrix &output_layer, int &selected_dataset, int &image_index);
void TestingWindowData_2(UIContext* ui_context, Mann::Matrix &output_layer, int selected_dataset, int image_index);
void TestingWindowCanvas_1(UIContext* ui_context, std::vector<std::vector<float>> &pixel_data, Mann::Matrix &output_layer);
void TestingWindowCanvas_2(UIContext* ui_context, Mann::Matrix &output_layer, int &response_index);
void NetworkVisualizationWindow_1(UIContext* ui_context);
void NetworkVisualizationWindow_2(UIContext* ui_context);
void NetworkConfigUI(NetworkConfiguration* network_configuration);
void PrintLayers(UIContext* ui_context);
void PrintTime(float total_time);
void PopupsContainer(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread);

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
        ImGuiID dock_id_top;
        ImGuiID dock_id_bottom;
        ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.8f, &dock_id_top, &dock_id_bottom);

        // Dock your windows into the nodes
        ImGui::DockBuilderDockWindow("Models", dock_id_top);

        ImGuiID dock_id_bottom_left;
        ImGuiID dock_id_bottom_right;

        ImGui::DockBuilderSplitNode(dock_id_bottom, ImGuiDir_Left, 0.35f, &dock_id_bottom_left, &dock_id_bottom_right);

        ImGui::DockBuilderDockWindow("Log Output", dock_id_bottom_left);
        ImGui::DockBuilderDockWindow("System Profiler", dock_id_bottom_right);

        ImGuiID dock_id_top_left;
        ImGuiID dock_id_top_right;
        ImGui::DockBuilderSplitNode(dock_id_top, ImGuiDir_Left, 0.2f, &dock_id_top_left, &dock_id_top_right);


        ImGui::DockBuilderDockWindow("Training Model Details", dock_id_top_left);
        ImGui::DockBuilderDockWindow("Training Progress", dock_id_top_right);

        ImGui::DockBuilderDockWindow("Data Testing Details", dock_id_top_left);
        ImGui::DockBuilderDockWindow("Data Testing Output", dock_id_top_right);

        ImGui::DockBuilderDockWindow("Canvas Testing Details", dock_id_top_left);
        ImGui::DockBuilderDockWindow("Canvas Testing Output", dock_id_top_right);

        ImGui::DockBuilderDockWindow("Network Visualizer Details", dock_id_top_left);
        ImGui::DockBuilderDockWindow("Network Visualizer", dock_id_top_right);

        // Finish dock builder
        ImGui::DockBuilderFinish(dockspace_id);

        // load the networks
        for (const std::string &filename : filenames)
        {
            Networks.emplace_back(filename, new MNNetwork(filename));
        }

        // Temporary code will delete later
        // selected_model = &Networks.front();

        MannLogger::info(outputText) << "Calculating model accuracy:" << std::endl;
        for (auto &entry : Networks)
        {
            std::thread test_thread([&entry, &outputText]()
            {

                float accuracy = entry.network->testNetwork(&mnist->mnist_testData);
                entry.network->m_accuracy_testdata = accuracy;
                entry.calculatingAccuracy = false;

                accuracy = entry.network->testNetwork(&mnist->mnist_trainingData);
                entry.network->m_accuracy = accuracy;

                MannLogger::info(outputText) << entry.network->m_model_name << " --- Accuracy: " << entry.network->m_accuracy_testdata << "%" << std::endl;
                // std::lock_guard<std::mutex> lock(resultsMutex);
            });
            test_thread.detach();
        }
    }

    ImGui::End();

    UIContext* ui_context = new UIContext{outputText, shown_windows_enum, selected_model, open_popup};

    // Output Window
    ImGui::Begin("Log Output");
    LogWindow(outputText);
    ImGui::End();

    ImGui::Begin("System Profiler");
    ProfilerWindow(outputText);
    ImGui::End();

    if( ui_context->open_popup.to_open ) {
        ImGui::OpenPopup( ui_context->open_popup.name.c_str() );
        ui_context->open_popup.to_open = false;
    }

    PopupsContainer(ui_context, is_training, training_thread, testing_thread);
    
    // Shared Global Static Variables
    static int selected_dataset = 0;
    static int image_index = 0;
    static Mann::Matrix output_layer = Mann::Matrix(10, 1);

    static std::vector<std::vector<float>> pixel_data = std::vector<std::vector<float>>(28, std::vector<float>(28, 0.0f)); // 28x28 canvas
    static Mann::Matrix output_layer_canvas = Mann::Matrix(10, 1);
    static int response_index = 0;

    // Top Windows
    switch (shown_windows_enum)
    {
    case MannUI::MODELS_WINDOW:
        ImGui::Begin("Models");
        ShowAvalModels(ui_context);
        ImGui::End();
        break;
    case MannUI::TRAINING_WINDOW:
        ImGui::Begin("Training Model Details");
        TrainingWindow_1(ui_context, is_training, training_thread, testing_thread);
        ImGui::End();
        ImGui::Begin("Training Progress");
        TrainingWindow_2(ui_context, is_training, training_thread, testing_thread);
        ImGui::End();

        ImGui::Begin("Network Visualizer");

        NetworkVisualizationWindow_2(ui_context);
        ImGui::End();
        break;
    case MannUI::TESTING_DATA_WINDOW:
        ImGui::Begin("Data Testing Details");
        TestingWindowData_1(ui_context, output_layer, selected_dataset, image_index);
        ImGui::End();
        ImGui::Begin("Data Testing Output");
        TestingWindowData_2(ui_context, output_layer, selected_dataset, image_index);
        ImGui::End();
        break;
    case MannUI::TESTING_CANVAS_WINDOW:
        ImGui::Begin("Canvas Testing Details");
        TestingWindowCanvas_1(ui_context, pixel_data, output_layer_canvas);
        ImGui::End();
        ImGui::Begin("Canvas Testing Output");
        TestingWindowCanvas_2(ui_context, output_layer_canvas, response_index);
        ImGui::End();
        break;
    case MannUI::NETWORK_VISUALIZER_WINDOW:
        ImGui::Begin("Network Visualizer Details");
        NetworkVisualizationWindow_1(ui_context);
        ImGui::End();

        ImGui::Begin("Network Visualizer");
        NetworkVisualizationWindow_2(ui_context);
        ImGui::End();
        break;
    default:
        break;
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

/**
 * @brief Parses a CSV string into a std::vector<int> with exactly three values.
 * @param input The input CSV string.
 * @param output The vector to store parsed integers.
 * @return True if exactly three valid non-negative integers were parsed, false otherwise.
 */
bool ParseCSVToHiddenLayers(const std::string &input, std::vector<int> &output)
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
            int value = std::stoul(token);
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

void LogWindow(std::stringstream &outputText)
{
    std::stringstream tempStream;
    tempStream << outputText.str(); // Copy current content
    std::string line;
    while (std::getline(tempStream, line))
    {
        if (line.empty())
            continue;
        // Find the second occurrence of "[" to get the log level (e.g., [DEBUG], [INFO])
        int firstBracket = line.find("[");
        int secondBracket = line.find("[", firstBracket + 1);
        int thirdBracket = line.find("[", secondBracket + 1);
        int thirdBracketEnd = line.find("]", thirdBracket + 1);
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
}

void ProfilerWindow(std::stringstream &outputText)
{
    // SystemProfiler* profiler = new SystemProfiler();
    // profiler->renderGraphs();
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
        ImGui::PushID(entry.model_id.c_str());

        if (ImGui::Button(entry.network->m_model_name.c_str()))
        {
            // ... show the details of the model in a popup
            ImGui::OpenPopup((entry.network->m_model_name + " Details").c_str());
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

        ImGui::NewLine();

        // popup for model details
        if (ImGui::BeginPopupModal((entry.network->m_model_name + " Details").c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            // Non Graphical things
            ui_context->selected_model = &entry;

            // Show model details here
            ImGui::Text("Model Name: %s", entry.network->m_model_name.c_str());
            ImGui::SameLine();
            // edit button on same line
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 100);
            if (ImGui::Button("Edit"))
            {
                // Open Edit Model Details popup
                ui_context->open_popup.name = "Edit Model Details";
                ui_context->open_popup.to_open = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
            ImGui::Separator();
            ImGui::NewLine();
            
            PrintLayers(ui_context);
            // ImGui::NewLine();
            ImGui::Text("Learning Rate: %.5f", ui_context->selected_model->network->m_learning_rate);
            ImGui::Text("Batch Size: %d", ui_context->selected_model->network->m_batch_size);
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            // show total time trained in hr:min:sec format
            float total_time = ui_context->selected_model->network->m_total_training_time;
            ImGui::Text("Total Time Trained:");
            ImGui::SameLine();
            PrintTime(total_time);

            ImGui::NewLine();
            // buttons (train, test network on data, test network by canvas)
            if (ImGui::Button("Train"))
            {
                // Train the network
                ui_context->shown_windows_enum = MannUI::TRAINING_WINDOW;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Test on Data"))
            {
                // Test the network on data
                ui_context->shown_windows_enum = MannUI::TESTING_DATA_WINDOW;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Test by Canvas"))
            {
                
                // Test the network by canvas
                ui_context->shown_windows_enum = MannUI::TESTING_CANVAS_WINDOW;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Network Visualizer"))
            { 
               // Open Network Visualizer confirmation popup
                ui_context->open_popup.name = "ConfirmVisualizerPopup";
                ui_context->open_popup.to_open = true;
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
        static std::vector<int> hidden_layers{50, 10};
        static float learning_rate = 0.01f;
        static int batch_size = 32;

        // Buffer for model name input
        static char modelNameBuffer[256] = "";
        if (ImGui::InputText("Model Name", modelNameBuffer, sizeof(modelNameBuffer)))
        {
            modelName = modelNameBuffer;
        }

        NetworkConfiguration* network_configuration = new NetworkConfiguration{modelName, hidden_layers, learning_rate, batch_size};
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
                for (int i = 0; i < hidden_layers.size(); ++i)
                {
                    layersStr << hidden_layers[i];
                }
                MannLogger::info(ui_context->outputText) << "Creating Model: " << modelName << " with layers [" << layersStr.str() << "], learning rate: " << learning_rate << ", batch size: " << batch_size << std::endl;
                // Create the network
                // MNNetwork network(modelName + ".mms", hidden_layers, learning_rate, batch_size);

                // Generate a unique model ID
                static const char alphanum[] =
                    "0123456789"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz";
                std::string model_id;
                model_id.reserve(16);
                for(int i = 0; i < 16; ++i) {
                    model_id += alphanum[rand() % (sizeof(alphanum) - 1)];
                }

                NetworkConfiguration* network_configuration = new NetworkConfiguration{modelName, hidden_layers, learning_rate, batch_size};
                Networks.push_back({model_id, new MNNetwork(model_id, network_configuration)});

                auto &newEntry = Networks.back();

                // test the network
                std::thread test_thread([&newEntry, &ui_context]()
                {
                    // dont fucking touch this code at all costs, pata nhi kese chala h ye, bus chal rha h.
                    float accuracy = newEntry.network->testNetwork(&mnist->mnist_testData);
                    newEntry.network->m_accuracy_testdata = accuracy;
                    newEntry.calculatingAccuracy = false;

                    accuracy = newEntry.network->testNetwork(&mnist->mnist_trainingData);
                    newEntry.network->m_accuracy = accuracy;
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
        std::vector<int> new_layers;
        if (ParseCSVToHiddenLayers(input, new_layers))
        {
            network_configuration->hidden_layers.insert(network_configuration->hidden_layers.end(), new_layers.begin(), new_layers.end());
            csv_buffer[0] = '\0';
        }
    }

    // Display current hidden_layers with Remove buttons
    ImGui::Text("Current Layers:");
    for (int i = 0; i < network_configuration->hidden_layers.size(); ++i)
    {
        ImGui::PushID(i);
        ImGui::Text("Layer %d: %d", i + 1, network_configuration->hidden_layers[i]);
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
            network_configuration->batch_size = static_cast<int>((batch_size_int > 200) ? 200 : batch_size_int);
    }
}


void TrainingWindow_1(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread)
{
    ImGui::Text("Details Panel");
    // if(!is_training) 
    // {
    if (is_training) ImGui::BeginDisabled();

    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 50);
    if (ImGui::Button("Close")) ui_context->shown_windows_enum = MannUI::MODELS_WINDOW;
    // }
    if (is_training) ImGui::EndDisabled();
    ImGui::Separator();
    ImGui::NewLine();
    if (ui_context->selected_model)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (ImGui::TreeNodeEx("Model Details", node_flags))
        {
            ImGui::SameLine();
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 50);
            if (ImGui::Button("Edit"))
            {
                // Open Edit Model Details popup
                ui_context->open_popup.name = "Edit Model Details";
                ui_context->open_popup.to_open = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::Text("Model Name: %s", ui_context->selected_model->network->m_model_name.c_str());
            PrintLayers(ui_context);
            ImGui::Text("Learning Rate: %.6f", ui_context->selected_model->network->m_learning_rate);
            ImGui::Text("Batch Size: %d", ui_context->selected_model->network->m_batch_size);

            ImGui::TreePop();
        }
        ImGui::NewLine();

        if (ImGui::TreeNodeEx("Training Details", node_flags))
        {
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            ImGui::Text("Epoch #: %d", ui_context->selected_model->network->m_current_epoch);
            ImGui::Text("Epoch Completion: %.2f%%", static_cast<float>(ui_context->selected_model->network->current_batch * ui_context->selected_model->network->m_batch_size) / static_cast<float>(mnist->mnist_trainingData.mnist_images_data.size()) * 100.0f);
            ImGui::Text("Batch #: %d/%lf", ui_context->selected_model->network->current_batch, (mnist->mnist_trainingData.mnist_images_data.size() / ui_context->selected_model->network->m_batch_size));
            // ImGui::Text("Time per Batch: %.4f seconds", ui_context->selected_model->network->m_time_per_batch);

            // show total time trained in hr:min:sec format
            ImGui::Text("Total Time Trained:");
            ImGui::SameLine();
            PrintTime(ui_context->selected_model->network->m_total_training_time);
            ImGui::Text("Average Time per Batch: %.4f seconds", ui_context->selected_model->network->m_averageTimePerBatch);
            ImGui::TreePop();
        }

        ImGui::NewLine();
        ImGui::Separator();
        ImGui::NewLine();

        // Add training controls and progress here
        if (!is_training) {
            if (ImGui::Button("Start Training"))
            {
                CreateTrainingThreads(ui_context, is_training, training_thread, testing_thread);
            }
        }
        else {
            ImGui::Text("Model is training...");

            if(ImGui::Button("Stop Training"))
            {
                // Stop training logic
                is_training = false;

                MannLogger::info(ui_context->outputText) << "Training stopped for model: " << ui_context->selected_model->network->m_model_name << std::endl;
            }
        }
    }
    else
        ImGui::Text("No Model Selected");
}

void CreateTrainingThreads(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread)
{
    if(training_thread.joinable())
        training_thread.join();

    if(testing_thread.joinable())
        testing_thread.join();

    // Start training logic
    MannLogger::info(ui_context->outputText) << "Starting training for model: " << ui_context->selected_model->network->m_model_name << std::endl;


    training_thread = std::thread([ui_context, &is_training]() {
        // Access selected_model via ui_context
        if (ui_context && ui_context->selected_model && ui_context->selected_model->network) {
            ui_context->selected_model->network->trainNetwork(1000, &mnist->mnist_trainingData, &is_training);
        } else {
            // Handle error: e.g., log to outputText if available
            if (ui_context && ui_context->outputText) {
                ui_context->outputText << "Error: Invalid ui_context or selected_model for training." << std::endl;
                is_training = false;
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

                ui_context->selected_model->network->m_average_cost_history.push_back(ui_context->selected_model->network->m_average_cost);


                accuracy = ui_context->selected_model->network->testNetwork(&mnist->mnist_testData);
                ui_context->selected_model->network->m_accuracy_testdata = accuracy;
                // Update accuracy history for UI graph
                ui_context->selected_model->network->m_accuracy_testdata_history.push_back(ui_context->selected_model->network->m_accuracy_testdata);

                ui_context->selected_model->network->m_average_cost_testdata_history.push_back(ui_context->selected_model->network->m_average_cost);

                ui_context->selected_model->network->saveHistoryData();

                MannLogger::info(ui_context->outputText) << "Graph Data Saved" << std::endl;
            }
        }
        
    });
}

void TrainingWindow_2(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread)
{
    ImGui::Text("Training Window");
    ImGui::Separator();
    ImGui::NewLine();

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
    if (ImGui::TreeNodeEx("Training Graphs Per Epoch", node_flags))
    {
        // 50% width for each plot
        ImVec2 plotSize(ImGui::GetContentRegionAvail().x * 0.5f - ImGui::GetStyle().ItemSpacing.x * 0.5f, 500);
        if(ImPlot::BeginPlot("Accuracy", plotSize))
        {
            ImPlot::SetupAxes("Epochs", "Accuracy (%)");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, std::max(50.0, static_cast<double>(ui_context->selected_model->network->m_accuracy_history.size())), ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 110.0, ImGuiCond_Always);

            ImPlot::PlotLine("Accuracy/TrainingData", ui_context->selected_model->network->m_accuracy_history.data(), static_cast<int>(ui_context->selected_model->network->m_accuracy_history.size()));
            ImPlot::PlotLine("Accuracy/TestData", ui_context->selected_model->network->m_accuracy_testdata_history.data(), static_cast<int>(ui_context->selected_model->network->m_accuracy_testdata_history.size()));
            
            ImPlot::EndPlot();
        }

        ImGui::TreePop();
        ImGui::SameLine();
        if(ImPlot::BeginPlot("Loss (Cost)", plotSize))
        {
            ImPlot::SetupAxes("Epochs", "Loss (Cost)");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, std::max(50.0, static_cast<double>(ui_context->selected_model->network->m_average_cost_history.size())), ImGuiCond_Always);
            // calculate max y limit
            double max_y = 0.0;
            for (const double &cost : ui_context->selected_model->network->m_average_cost_history)
            {
                if (cost > max_y)
                    max_y = cost;
            }
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(1.0, max_y * 1.1), ImGuiCond_Always);

            ImPlot::PlotLine("Cost/TrainingData", ui_context->selected_model->network->m_average_cost_history.data(), static_cast<int>(ui_context->selected_model->network->m_average_cost_history.size()));
            ImPlot::PlotLine("Cost/TestData", ui_context->selected_model->network->m_average_cost_testdata_history.data(), static_cast<int>(ui_context->selected_model->network->m_average_cost_testdata_history.size()));
            
            ImPlot::EndPlot();
        }
    }

    if (ImGui::TreeNodeEx("Training Graphs Per Batch", node_flags))
    {
        if(ImPlot::BeginPlot("Graph Per Batch"))
        {
            ImPlot::SetupAxes("Batches", "Accuracy (%)");
            // ImPlot::SetupAxisLimits(ImAxis_X1, std::max(0.0, static_cast<double>(ui_context->selected_model->network->m_batch_accuracy_history.size())-100), std::max(100.0, static_cast<double>(ui_context->selected_model->network->m_batch_accuracy_history.size())), ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, 100, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImGuiCond_Always);

            std::queue<float> batch_history_copy = ui_context->selected_model->network->m_batch_accuracy_history;
            std::vector<float> batch_history_vector;
            while(!batch_history_copy.empty())
            {
                batch_history_vector.push_back(batch_history_copy.front());
                batch_history_copy.pop();
            }
            ImPlot::PlotLine("Accuracy Per Batch", batch_history_vector.data(), static_cast<int>(batch_history_vector.size()));
            
            ImPlot::EndPlot();
        }

        ImGui::TreePop();
    }
}

void TestingWindowData_1(UIContext* ui_context, Mann::Matrix &output_layer, int &selected_dataset, int &image_index)
{
    ImGui::Text("Details Panel");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 50);
    if (ImGui::Button("Close")) ui_context->shown_windows_enum = MannUI::MODELS_WINDOW;
    ImGui::Separator();
    ImGui::NewLine();

    if (ui_context->selected_model)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (ImGui::TreeNodeEx("Model Details", node_flags))
        {
            ImGui::Text("Model Name: %s", ui_context->selected_model->network->m_model_name.c_str());
            PrintLayers(ui_context);
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            // show total time trained in hr:min:sec format
            ImGui::Text("Total Time Trained:");
            ImGui::SameLine();
            PrintTime(ui_context->selected_model->network->m_total_training_time);

            ImGui::TreePop();
        }

        ImGui::NewLine();

        if (ImGui::TreeNodeEx("Testing Controls", node_flags))
        {
            //select dataset for testing
            static const char* datasets[]{"MNIST Test Data", "MNIST Training Data"};
            int dataset_size = (selected_dataset == 0) ? mnist->mnist_testData.mnist_images_data.size() : mnist->mnist_trainingData.mnist_images_data.size();
            if(ImGui::Combo("Select Dataset", &selected_dataset, datasets, IM_ARRAYSIZE(datasets)))
            {
                image_index = dataset_size / 4;
            }
            ImGui::NewLine();

            //image index slider
            ImGui::Text("Dataset Size: %d", dataset_size);
            if(ImGui::Button("Random Image"))
            {
                image_index = rand() % dataset_size;
            }
            ImGui::SliderInt("Input Index", &image_index, 0, dataset_size - 1);
            ImGui::NewLine();


            Mnist::MnistData* data = (selected_dataset == 0) ? &mnist->mnist_testData : &mnist->mnist_trainingData;
            std::vector<double> image_data = data->mnist_images_data[image_index];

            if (ImGui::Button("Find Wrong Prediction"))
            {
                int i = 0;
                while(true) {
                    image_index = rand() % dataset_size;
                    image_data = data->mnist_images_data[image_index];
                    output_layer = ui_context->selected_model->network->predictSingleImage(image_data);
                    Mann::Matrix y(10, 1);
                    for (int i = 0; i < 10; ++i)
                    {
                        y[i][0] = data->mnist_labels_data[image_index][i];
                    }
                    if(!ui_context->selected_model->network->IsPredictionCorrect(output_layer, y)){
                        break;
                    }
                    i++;
                    // This will never happen (like its impossible to have 100% accuracy on MNIST) but just in case to avoid infinite loop
                    if(i > dataset_size) {
                        MannLogger::info(ui_context->outputText) << "All predictions are correct in the dataset!" << std::endl;
                        break;
                    }
                }
            }
            else {
                output_layer = ui_context->selected_model->network->predictSingleImage(image_data);
            }

            ImGui::TreePop();
        }
    }
}

void TestingWindowData_2(UIContext* ui_context, Mann::Matrix &output_layer, int selected_dataset, int image_index)
{
    ImGui::Text("Testing Output Panel");

    Mnist::MnistData* data = (selected_dataset == 0) ? &mnist->mnist_testData : &mnist->mnist_trainingData;

    ImGui::Separator();

    ImGui::Text("Actual Number: ");
    ImGui::SameLine();
    int actual_number = 0;
    for (int i = 0; i < 10; ++i)
    {
        if (data->mnist_labels_data[image_index][i] == 1.0f)
        {
            actual_number = i;
            break;
        }
    }
    ImGui::Text("%d", actual_number);

    Mann::Matrix y(10, 1);
    for (int i = 0; i < 10; ++i)
    {
        y[i][0] = data->mnist_labels_data[image_index][i];
    }

    Mann::Matrix Cost = (output_layer - y);
    Cost = Cost ^ Cost;
    float total_cost = 0.0f;
    for (int i = 0; i < Cost.rows(); ++i)
    {
        total_cost += Cost[i][0];
    }

    ImGui::Text("Predicted Output: ");
    ImGui::SameLine();
    int predicted_number = 0;
    float max_value = output_layer[0][0];
    for (int i = 1; i < 10; ++i)
    {
        if (output_layer[i][0] > max_value)
        {
            max_value = output_layer[i][0];
            predicted_number = i;
        }
    }
    ImGui::Text("%d", predicted_number);
    ImGui::Text("Cost: %.6f", total_cost);

    ImGui::Separator();


    // Display output probabilities are a bar graph
    if (ImPlot::BeginPlot("Output Probabilities", ImVec2(1000,400)))
    {
        ImPlot::SetupAxes("Numbers", "Probability", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImGuiCond_Always);
        float x[10];
        float y[10];
        for (int i = 0; i < 10; ++i)
        {
            x[i] = static_cast<float>(i);
            y[i] = output_layer[i][0];
        }
        ImPlot::PlotBars("Probabilities", x, y, 10, 0.5f);
        
        for (int i = 0; i < 10; ++i)
        {
            y[i] = data->mnist_labels_data[image_index][i];
        }
        ImPlot::PlotStems("Actual", x, y, 10);

        ImPlot::EndPlot();
    }

    ImGui::Separator();

    // Display the input image
    ImGui::Text("Input Image:");
    auto drawlist = ImGui::GetWindowDrawList();
    
    ImVec2 p = ImGui::GetCursorScreenPos();

    // draw 28x28 image scaled by 10
    float scale = 10.0f;
    for (int y = 0; y < 28; ++y )
    {
        for (int x = 0; x < 28; ++x)
        {
            float pixel_value = data->mnist_images_data[image_index][y * 28 + x];
            ImU32 col = IM_COL32(static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), 255);
            drawlist->AddRectFilled(ImVec2(p.x + x * scale, p.y + y * scale), ImVec2(p.x + (x + 1) * scale, p.y + (y + 1) * scale), col);
        }
    }
}

void TestingWindowCanvas_1(UIContext* ui_context, std::vector<std::vector<float>> &pixel_data, Mann::Matrix &output_layer)
{
    ImGui::Text("Details Panel");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 50);
    if (ImGui::Button("Close")) ui_context->shown_windows_enum = MannUI::MODELS_WINDOW;
    ImGui::Separator();
    ImGui::NewLine();

    if (ui_context->selected_model)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (ImGui::TreeNodeEx("Model Details", node_flags))
        {
            ImGui::Text("Model Name: %s", ui_context->selected_model->network->m_model_name.c_str());
            ImGui::Text("Layers: ");
            ImGui::SameLine();
            PrintLayers(ui_context);
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            // show total time trained in hr:min:sec format
            ImGui::Text("Total Time Trained:");
            ImGui::SameLine();
            PrintTime(ui_context->selected_model->network->m_total_training_time);

            ImGui::TreePop();
        }

        ImGui::NewLine();

        if (ImGui::TreeNodeEx("Testing Controls", node_flags))
        {
            ImGui::Text("CANVAS INPUT");
            
            // Make a canvas window
            ImGuiIO& io = ImGui::GetIO();
            ImVec2 mousePos = io.MousePos;
            float mouseX = mousePos.x;
            float mouseY = mousePos.y;

            if (ImGui::Button("Clear Canvas"))
            {
                for (int y = 0; y < 28; ++y)
                {
                    for (int x = 0; x < 28; ++x)
                    {
                        pixel_data[y][x] = 0; // set pixel to black
                    }
                }
            }

            // Canvas preview
            auto drawlist = ImGui::GetWindowDrawList();
    
            ImVec2 p = ImGui::GetCursorScreenPos();

            // draw 28x28 image scaled by 10
            float scale = 10.0f;
            for (int y = 0; y < 28; ++y )
            {
                for (int x = 0; x < 28; ++x)
                {
                    float pixel_value = pixel_data[y][x];
                    ImU32 col = IM_COL32(static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), 255);
                    drawlist->AddRectFilled(ImVec2(p.x + x * scale, p.y + y * scale), ImVec2(p.x + (x + 1) * scale, p.y + (y + 1) * scale), col);
                }
            } 

            // Canvas Input with faded effect
            ImGui::InvisibleButton("canvas", ImVec2(280, 280));
            ImVec2 canvasPos = ImGui::GetItemRectMin();
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                int x = static_cast<int>((mouseX - canvasPos.x) / 10.0f);
                int y = static_cast<int>((mouseY - canvasPos.y) / 10.0f);
                if (x >= 0 && x < 28 && y >= 0 && y < 28)
                {
                    pixel_data[y][x] = std::min(1.0f, pixel_data[y][x] + 0.3f); // increase pixel brightness
                    if (x > 0) pixel_data[y][x - 1] = std::max(pixel_data[y][x - 1], 0.2f); // left pixel
                    if (x < 27) pixel_data[y][x + 1] = std::max(pixel_data[y][x + 1], 0.2f); // right pixel
                    if (y > 0) pixel_data[y - 1][x] = std::max(pixel_data[y - 1][x], 0.2f); // top pixel
                    if (y < 27) pixel_data[y + 1][x] = std::max(pixel_data[y + 1][x], 0.2f); // bottom pixel
                    //corners
                    if (x > 0 && y > 0) pixel_data[y - 1][x - 1] = std::max(pixel_data[y - 1][x - 1], 0.1f); // top-left
                    if (x < 27 && y > 0) pixel_data[y - 1][x + 1] = std::max(pixel_data[y - 1][x + 1], 0.1f); // top-right
                    if (x > 0 && y < 27) pixel_data[y + 1][x - 1] = std::max(pixel_data[y + 1][ x - 1], 0.1f); // bottom-left
                    if (x < 27 && y < 27) pixel_data[y + 1][x + 1] = std::max(pixel_data[y + 1][ x + 1], 0.1f); // bottom-right
                }
            }

            ImGui::TreePop();
        }

        // Test the canvas input
        
        std::vector<double> image_data(28 * 28, 0.0);
        for (int y = 0; y < 28; ++y)
        {
            for (int x = 0; x < 28; ++x)
            {
                image_data[y * 28 + x] = static_cast<double>(pixel_data[y][x]);
            }
        }

        output_layer = ui_context->selected_model->network->predictSingleImage(image_data);
    }
}

void TestingWindowCanvas_2(UIContext* ui_context, Mann::Matrix &output_layer, int &response_index)
{
    ImGui::Text("Testing Output Panel");
    ImGui::Separator();

    // Randomly select an AI response
    std::vector<std::string> AI_responses = {
        "Is it a %d?",
        "Looks like a %d to me.",
        "I'm pretty sure it's a %d.",
        "This seems to be a %d.",
        "I'd say it's a %d."
    };
    
    if(rand() % 600 == 532) // change response every now and then
        response_index = rand() % AI_responses.size();

    // Calculate predicted number
    int predicted_number = 0;
    float max_value = output_layer[0][0];
    for (int i = 1; i < 10; ++i)
    {
        if (output_layer[i][0] > max_value)
        {
            max_value = output_layer[i][0];
            predicted_number = i;
        }
    }
    ImGui::Text(AI_responses[response_index].c_str(), predicted_number);

    ImGui::Separator();


    // Display output probabilities are a bar graph
    if (ImPlot::BeginPlot("Output Probabilities", ImVec2(1000,400)))
    {
        ImPlot::SetupAxes("Numbers", "Probability", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImGuiCond_Always);
        float x[10];
        float y[10];
        for (int i = 0; i < 10; ++i)
        {
            x[i] = static_cast<float>(i);
            y[i] = output_layer[i][0];
        }
        ImPlot::PlotBars("Probabilities", x, y, 10, 0.5f);

        ImPlot::EndPlot();
    }

    // Show predictions in decending order
    ImGui::Separator();
    ImGui::Text("Predictions in order:");
    std::vector<std::pair<int, float>> predictions;
    for (int i = 0; i < 10; ++i)
    {
        predictions.push_back({i, output_layer[i][0]});
    }
    std::sort(predictions.begin(), predictions.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });
    for (const auto &pred : predictions)
    {
        ImGui::Text("%d: %.2f%%", pred.first, pred.second*100.0f);
    }
}

void NetworkVisualizationWindow_1(UIContext* ui_context)
{
    ImGui::Text("Details Panel");
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 50);
    if (ImGui::Button("Close")) ui_context->shown_windows_enum = MannUI::MODELS_WINDOW;
    ImGui::Separator();
    ImGui::NewLine();

    if (ui_context->selected_model)
    {
        ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (ImGui::TreeNodeEx("Model Details", node_flags))
        {
            ImGui::Text("Model Name: %s", ui_context->selected_model->network->m_model_name.c_str());
            PrintLayers(ui_context);
            ImGui::Text("Training Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy);
            ImGui::Text("Test Data Accuracy: %.2f%%", ui_context->selected_model->network->m_accuracy_testdata);
            // show total time trained in hr:min:sec format
            ImGui::Text("Total Time Trained:");
            ImGui::SameLine();
            PrintTime(ui_context->selected_model->network->m_total_training_time);

            ImGui::TreePop();
        }

        ImGui::NewLine();

        if(ImGui::TreeNodeEx("Visualization Controls", node_flags))
        {
            static const char* visualizer_types[]{"Full Network (WARNING!!)", "Simplified Network"};
            static int selected_visualizer = 1;
            ImGui::Combo("Select Visualizer Type", &selected_visualizer, visualizer_types, IM_ARRAYSIZE(visualizer_types));
            

            ImGui::TreePop();
        }
        ImGui::NewLine();

        if (ImGui::TreeNodeEx("Testing Controls", node_flags))
        {
            //select dataset for testing
            static const char* datasets[]{"MNIST Test Data", "MNIST Training Data"};
            static int selected_dataset = 0;
            static int image_index = 0;
            int dataset_size = (selected_dataset == 0) ? mnist->mnist_testData.mnist_images_data.size() : mnist->mnist_trainingData.mnist_images_data.size();
            if(ImGui::Combo("Select Dataset", &selected_dataset, datasets, IM_ARRAYSIZE(datasets)))
            {
                image_index = dataset_size / 4;
            }
            ImGui::NewLine();

            //image index slider
            ImGui::Text("Dataset Size: %d", dataset_size);
            if(ImGui::Button("Random Image"))
            {
                image_index = rand() % dataset_size;
            }
            ImGui::SliderInt("Input Index", &image_index, 0, dataset_size - 1);
            ImGui::NewLine();


            Mnist::MnistData* data = (selected_dataset == 0) ? &mnist->mnist_testData : &mnist->mnist_trainingData;
            std::vector<double> image_data = data->mnist_images_data[image_index];

            if (ImGui::Button("Find Wrong Prediction"))
            {
                int i = 0;
                while(true) {
                    image_index = rand() % dataset_size;
                    image_data = data->mnist_images_data[image_index];
                    Mann::Matrix output_layer = ui_context->selected_model->network->predictSingleImage(image_data);
                    Mann::Matrix y(10, 1);
                    for (int i = 0; i < 10; ++i)
                    {
                        y[i][0] = data->mnist_labels_data[image_index][i];
                    }
                    if(!ui_context->selected_model->network->IsPredictionCorrect(output_layer, y)){
                        break;
                    }
                    i++;
                    // This will never happen (like its impossible to have 100% accuracy on MNIST) but just in case to avoid infinite loop
                    if(i > dataset_size) {
                        MannLogger::info(ui_context->outputText) << "All predictions are correct in the dataset!" << std::endl;
                        break;
                    }
                }
            }
            else {
                ui_context->selected_model->network->predictSingleImage(image_data);
            }
            ImGui::NewLine();

            // Display the input image
            ImGui::Text("Input Image:");
            auto drawlist = ImGui::GetWindowDrawList();
            
            ImVec2 p = ImGui::GetCursorScreenPos();

            // draw 28x28 image scaled by 10
            float scale = 10.0f;
            for (int y = 0; y < 28; ++y )
            {
                for (int x = 0; x < 28; ++x)
                {
                    float pixel_value = data->mnist_images_data[image_index][y * 28 + x];
                    ImU32 col = IM_COL32(static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), static_cast<ImU8>(pixel_value * 255), 255);
                    drawlist->AddRectFilled(ImVec2(p.x + x * scale, p.y + y * scale), ImVec2(p.x + (x + 1) * scale, p.y + (y + 1) * scale), col);
                }
            }

            ImGui::TreePop();
        }
    }
}

void NetworkVisualizationWindow_2(UIContext* ui_context)
{
    ImGui::Text("Network Visualizer");
    ImGui::Separator();
    ImGui::NewLine();

    if (ui_context->selected_model)
    {
        static std::vector<Mann::Matrix> MNN_NODES_live;
        static std::vector<Mann::Matrix> MNN_WEIGHTS_live;
        static std::vector<Mann::Matrix> MNN_BIAS_live;
        float count = 0.0f;

        // Update live node activations, weights, and biases
        if(count <= 0.0f)
        {
            ui_context->selected_model->network->training_threads_mutex.lock();
            MNN_NODES_live = ui_context->selected_model->network->MNN_Nodes;
            MNN_WEIGHTS_live = ui_context->selected_model->network->MNN_Weights;
            MNN_BIAS_live = ui_context->selected_model->network->MNN_Bias;
            ui_context->selected_model->network->training_threads_mutex.unlock();
            count = 1000.0f;
        }
        else
        {
            count -= 1.0f;
        }
        

        // Render the network visualization
        ImGui::Text("Model Name: %s", ui_context->selected_model->network->m_model_name.c_str());
        ImGui::NewLine();

        auto drawlist = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();

        // std::vector<ImVec2> canvas_boundary_points;
        float canvas_width = ImGui::GetContentRegionAvail().x - 30.0f;
        float canvas_height = ImGui::GetContentRegionAvail().y - 50.0f;

        // Draw Nodes
        float layer_spacing = canvas_width / static_cast<float>(MNN_NODES_live.size() + 1);

        for(int i=0; i < MNN_NODES_live.size(); ++i)
        {
            float nodes_spacing = canvas_height / static_cast<float>(MNN_NODES_live[i].rows() + 1);
            float radius = 10.0f - (8.0f / 783.0f) * (MNN_NODES_live[i].rows() - 1); // adjust radius based on number of nodes
            // float radius = 10.0f;

            for(int j = 0; j < MNN_NODES_live[i].rows(); ++j)
            {
                float x = p.x + (i + 1) * layer_spacing;
                float y = p.y + (j + 1) * nodes_spacing;
                // drawlist->AddCircleFilled(ImVec2(x, y), radius, IM_COL32(0, 255, 0, 255));
                ImU8 brightness_value = static_cast<ImU8>(MNN_NODES_live[i][j][0] * 255.0f);
                drawlist->AddCircleFilled(ImVec2(x, y), radius, IM_COL_WHITE(brightness_value));
                // circle border

                if(i > 0)
                {
                    ImU8 bias_brightness = static_cast<ImU8>(MNN_BIAS_live[i-1][j][0] * 255.0f);
                    drawlist->AddCircle(ImVec2(x, y), radius + 1.0f, IM_COL_GREEN(bias_brightness), 0, 3);
                }
                drawlist->AddCircle(ImVec2(x, y), radius, IM_COL32(250, 250, 255, 255));
            }
        }

        // Draw Connections

        for(int i=0; i < MNN_WEIGHTS_live.size(); ++i)
        {
            float nodes_spacing_current = canvas_height / static_cast<float>(MNN_NODES_live[i].rows() + 1);
            float nodes_spacing_next = canvas_height / static_cast<float>(MNN_NODES_live[i+1].rows() + 1);
            float radius_current = 10.0f - (8.0f / 783.0f) * (MNN_NODES_live[i].rows() - 1);
            float radius_next = 10.0f - (8.0f / 783.0f) * (MNN_NODES_live[i+1].rows() - 1);

            for(int j = 0; j < MNN_WEIGHTS_live[i].rows(); ++j)
            {
                for(int k = 0; k < MNN_WEIGHTS_live[i].cols(); ++k)
                {
                    float x1 = p.x + (i + 1) * layer_spacing;
                    float y1 = p.y + (k + 1) * nodes_spacing_current;
                    float x2 = p.x + (i + 2) * layer_spacing;
                    float y2 = p.y + (j + 1) * nodes_spacing_next;

                    // determine color based on weight value
                    float weight_value = MNN_WEIGHTS_live[i][j][k];
                    weight_value = std::max(-1.0f, std::min(1.0f, weight_value)); // clamp between -1 and 1
                    ImU32 color;
                    if(weight_value > 0)
                    {
                        weight_value = std::clamp((weight_value - 0.5f) / 0.5f, 0.0f, 1.0f); // remap to 0-1 range
                        color = IM_COL_RED(weight_value * 120.0f);
                    }
                    else
                    {
                        weight_value = std::clamp((-weight_value - 0.5f) / 0.5f, 0.0f, 1.0f); // remap to 0-1 range
                        color = IM_COL_BLUE(weight_value * 120.0f);
                    }

                    drawlist->AddLine(ImVec2(x1 + radius_current, y1), ImVec2(x2 - radius_next, y2), color);
                }
            }
        }
    }
    else
    {
        ImGui::Text("No Model Selected");
    }
}

void PrintLayers(UIContext* ui_context)
{
    ImGui::Text("Layers:");
    ImGui::SameLine();
    int i = 0;
    for (const int &layer : ui_context->selected_model->network->MNN_Layers_size)
    {
        ImGui::Text("%d", layer);
        if( i < ui_context->selected_model->network->MNN_Layers_size.size() - 1)
        {
            ImGui::SameLine();
            ImGui::Text("->");
            ImGui::SameLine();
        }
        i++;
    }
}

void PrintTime(float total_time) {
    int hours = static_cast<int>(total_time) / 3600;
    int minutes = (static_cast<int>(total_time) % 3600) / 60;
    float seconds = total_time - (hours * 3600) - (minutes * 60);
    ImGui::Text("%02d:%02d:%05.2f (hh:mm:ss)", hours, minutes, seconds);
}

void PopupsContainer(UIContext* ui_context, bool &is_training, std::thread &training_thread, std::thread &testing_thread)
{
    
    if(ImGui::BeginPopupModal("Edit Model Details", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        // variables to hold edited values
        static std::string _model_name = ui_context->selected_model->network->m_model_name;
        static float _learning_rate = ui_context->selected_model->network->m_learning_rate;
        static int _batch_size = ui_context->selected_model->network->m_batch_size;

        // Input fields
        char model_name_buffer[256];
        std::strncpy(model_name_buffer, _model_name.c_str(), sizeof(model_name_buffer));
        if (ImGui::InputText("Model Name", model_name_buffer, sizeof(model_name_buffer)))
        {
            _model_name = std::string(model_name_buffer);
        }
        ImGui::InputFloat("Learning Rate", &_learning_rate);
        ImGui::InputInt("Batch Size", &_batch_size);    
        ImGui::NewLine();
        if (ImGui::Button("Save Changes"))
        {
            bool was_training = is_training;
            if(is_training)
            {
                //stop training before applying changes
                is_training = false;
                if(training_thread.joinable())
                    training_thread.join();
                if(testing_thread.joinable())
                    testing_thread.join();
            }

            // Apply changes to the selected model
            ui_context->selected_model->network->m_model_name = _model_name;
            ui_context->selected_model->network->m_learning_rate = _learning_rate;
            ui_context->selected_model->network->m_batch_size = _batch_size;

            if(was_training)
            {
                //restart training if it was stopped
                is_training = true;
                CreateTrainingThreads(ui_context, is_training, training_thread, testing_thread);

                // click the start training button
            }

            ui_context->selected_model->network->saveNetwork();

            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();

        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }


    if(ImGui::BeginPopupModal("ConfirmVisualizerPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("This was your decision to open the Network Visualizer. If your system blast with smoke, I am not responsible.\nI am asking you one last time, are you sure you want to open the Network Visualizer?");
        
        if (ImGui::Button("I am using high-end system. I do not care"))
        {
            // Open Network Visualizer
            ui_context->shown_windows_enum = MannUI::NETWORK_VISUALIZER_WINDOW;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("No, take me back"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}