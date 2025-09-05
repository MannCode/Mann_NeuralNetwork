#include "mannui.hpp"

std::vector<std::string> filenames = getTxtFileNamesWithoutExtension();

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

inline MannUI::~MannUI()
{

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// PROTOTYPES
void ShowAvalModels(std::stringstream &outputText);

void MannUI::Render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

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