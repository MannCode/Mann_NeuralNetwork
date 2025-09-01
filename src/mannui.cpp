#include "mannui.h"

#include <string>

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

void MannUI::Render()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    // Output Window
    ImGui::Begin("Output");
    ImGui::TextWrapped("%s", outputText.c_str());
    if (ImGui::Button("Clear Output"))
    {
        outputText.clear();
    }
    ImGui::End();

    // Control Panel
    ImGui::Begin("Control Panel");
    if (ImGui::Button("Run"))
    {
        outputText += "AHH AHH YEH MOMMY Ahh...\n";
    }
    ImGui::SameLine();
    if (ImGui::Button("Test"))
    {
        outputText += "Testing Model...\n";
    }
    ImGui::SameLine();
    if (ImGui::Button("Train"))
    {
        outputText += "Training model with LR=" + std::to_string(learning_rate) +
                      ", Iterations=" + std::to_string(iterations_rate) +
                      ", Batch Size=" + std::to_string(batch_size) + "\n";
    }

    ImGui::Text("Training Parameters");
    ImGui::InputFloat("Learning Rate", &learning_rate, 0.001f, 0.1f, "%.3f");
    ImGui::InputScalar("Iterations", ImGuiDataType_U64, &iterations_rate, nullptr, nullptr, "%zu");
    ImGui::InputScalar("Batch Size", ImGuiDataType_U64, &batch_size, nullptr, nullptr, "%zu");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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
