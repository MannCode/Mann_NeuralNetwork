/**
 * @file utils.h
 * @brief Header file for setting up ImGui with GLFW and OpenGL3 backend.
 * @author Jayansh Devgan
 * @date 2025-09-06
 * @version 1.0
 *
 * This file includes necessary headers and macro definitions for integrating
 * ImGui with GLFW and OpenGL3 for creating graphical user interfaces.
 */

/**
 * @brief Include ImGui core library.
 *
 * Provides the core functionality for the ImGui immediate-mode GUI library.
 */
#include "../dependencies/includes/imgui/imgui.h"

/**
 * @brief Suppresses OpenGL deprecation warnings.
 *
 * This macro silences deprecation warnings for OpenGL functions on platforms
 * like macOS, where certain OpenGL functions are marked as deprecated.
 */
#define GL_SILENCE_DEPRECATION

/**
 * @brief Include GLFW library for window and input management.
 *
 * GLFW is used to create windows, handle input, and manage OpenGL contexts.
 */
#include "../dependencies/includes/GLFW/include/GLFW/glfw3.h"

/**
 * @brief Include ImGui GLFW backend implementation.
 *
 * This backend integrates ImGui with GLFW for handling window and input events.
 */
#include "../dependencies/includes/imgui/backends/imgui_impl_glfw.h"

/**
 * @brief Include ImGui OpenGL3 backend implementation.
 *
 * This backend integrates ImGui with OpenGL3 for rendering the GUI.
 */
#include "../dependencies/includes/imgui/backends/imgui_impl_opengl3.h"
