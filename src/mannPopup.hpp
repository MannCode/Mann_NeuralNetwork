#include "utils.h"
#include <unordered_map>
#include <string>

namespace MannPopups
{
    extern std::unordered_map<std::string, std::string> popupMessages;

    inline void showMessage(const std::string& id, const std::string& message)
    {
        popupMessages[id] = message;
        ImGui::OpenPopup(id.c_str());
    }

    inline void closeMessage(const std::string& id) { 
       popupMessages.erase(id);
   }

    inline void renderPopups()
    {
        for (auto it = popupMessages.begin(); it != popupMessages.end();)
        {
            const std::string& id = it->first;
            const std::string& message = it->second;
            
            if (ImGui::BeginPopup(id.c_str(), ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize))
            {
                ImGui::Text("%s", message.c_str());
                
                if (ImGui::Button(("Close##" + id).c_str()))
                {
                    ImGui::CloseCurrentPopup();
                    it = popupMessages.erase(it);
                    continue;
                }
                ImGui::EndPopup();
            }
            ++it;
        }
    }

};