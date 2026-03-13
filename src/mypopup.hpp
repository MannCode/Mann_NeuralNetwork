#include "utils.h"
  
#include <string>

class mypopup
{
private:
    
public:
    mypopup(std::string msg) : message(msg)
    {
        if(ImGui::BeginPopup("My Popo" , ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("%s", message.c_str());
            ImGui::EndPopup();
        }

        _id = _id_counter++;
    }

    void show()
    {
        ImGui::OpenPopup(std::to_string(_id).c_str());
    }

    void close()
    {
        ImGui::CloseCurrentPopup();
    }

    std::string message;


};

