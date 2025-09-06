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

/**
 * @brief Constructs a MannUI object and initializes ImGui with GLFW and OpenGL.
 * @param window The GLFW window for rendering the UI.
 * @param learning_rate The learning rate for neural network training.
 * @param iterations_rate The number of iterations for training.
 * @param batch_size The batch size for training.
 */
MannUI::MannUI(GLFWwindow *window, float learning_rate, size_t iterations_rate, size_t batch_size)
    : window(window), learning_rate(learning_rate), iterations_rate(iterations_rate),
      batch_size(batch_size)
{
    // Init ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
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
void ShowAvalModels(std::stringstream &outputText);

/**
 * @brief Renders the ImGui-based user interface.
 *
 * Sets up a new ImGui frame, renders the output window and control panel,
 * and handles docking and rendering of draw data.
 */
void MannUI::Render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    ImGuiViewport* viewport = ImGui::GetMainViewport();
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
        ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.3f, &dock_id_left, &dock_id_right);
        // Dock your windows into the nodes
        ImGui::DockBuilderDockWindow("Models", dock_id_left);
        ImGui::DockBuilderDockWindow("Output", dock_id_right);
        // Finish dock builder
        ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::End();

    // Output Window
    ImGui::Begin("Output");
    ImGui::TextWrapped("%s", outputText.str().c_str());
    if (ImGui::Button("Clear Output"))
    {
        outputText.clear();
    }
    ImGui::End();

    // Control Panel
    ImGui::Begin("Models");
    ShowAvalModels(outputText);
    ImGui::End();
    
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

/**
 * @brief Displays available models and a button to create a new model.
 * @param outputText A stringstream to append output messages for display in the UI.
 */
void ShowAvalModels(std::stringstream &outputText)
{
    for (const std::string& name : filenames)
    {
        if (ImGui::Button(name.c_str()))
        {
            outputText << name << std::endl;
        }
    }

    if (ImGui::Button("Create New Model"))
    {
        ImGui::OpenPopup("CreateModelPopup");
    }

    if (ImGui::BeginPopupModal("CreateModelPopup", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Enter new model details here.");

        static std::string modelName = "";
        
        std::vector<char> buffer(modelName.begin(), modelName.end());
        buffer.resize(256);
        buffer.push_back('\0');
        
        if (ImGui::InputText("Model Name", buffer.data(), buffer.size()))
        {
            modelName = buffer.data();
        }

        if (ImGui::Button("Create"))
        {
            if (modelName.empty()) modelName = getRandomModelName();
            outputText << "Creating Model: " << modelName << std::endl;
            filenames.push_back(modelName);
            MNNetwork network(modelName, std::vector<size_t>{100, 20}); // Example hidden layers
            MNNetwork::Networks networks;
            networks.modelName.push_back(modelName);
            networks.network.push_back(network);

            // network.CreateNewModel(modelName);
            ImGui::CloseCurrentPopup();

            modelName[0] = '\0';
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel"))
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
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