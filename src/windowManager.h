#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include "mannui.h"

class WindowManager
{
public:
    static int initalizeGlfwWithHints();

    static GLFWwindow* createWindow(int width, int height);

    static void adolfHitler(GLFWwindow* window);

    static void mainLoop(GLFWwindow* window, MannUI* ui);

};

#endif // WINDOW_MANAGER_H
