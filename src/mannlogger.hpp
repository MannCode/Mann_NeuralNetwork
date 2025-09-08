#ifndef MANNLOGGER_HPP
#define MANNLOGGER_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

// ANSI color codes for console output (used for console only)
#define COLOR_RESET "\033[0m"
#define COLOR_DEBUG "\033[34m"  // Dark Cyan (better contrast on light)
#define COLOR_INFO "\033[32m"   // Green (visible, but consider darker if needed)
#define COLOR_WARN "\033[33m"   // Yellow (visible, could use bold with \033[1m)
#define COLOR_ERROR "\033[31m"  // Red (visible, bold optional)

// ImGui color definitions (RGB values as floats 0.0f to 1.0f, adjusted for light themes)
#define IMGUI_COLOR_DEBUG ImVec4(0.0f, 0.5f, 0.7f, 1.0f)  // Darker Cyan
#define IMGUI_COLOR_INFO  ImVec4(0.0f, 0.6f, 0.0f, 1.0f)  // Darker Green
#define IMGUI_COLOR_WARN  ImVec4(0.8f, 0.5f, 0.0f, 1.0f)  // Darker Yellow
#define IMGUI_COLOR_ERROR ImVec4(0.8f, 0.0f, 0.0f, 1.0f)  // Darker Red

namespace MannLogger {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// Forward declaration of LogStream
class LogStream;

// Function to get current timestamp
inline std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// LogStream class to handle stream-like logging
class LogStream {
private:
    LogLevel level;
    std::stringstream ss;
    std::stringstream& guiOutput;

public:
    LogStream(LogLevel lvl, std::stringstream& gui) : level(lvl), guiOutput(gui) {}

    // Overload << operator for various types
    template <typename T>
    LogStream& operator<<(const T& value) {
        ss << value;
        return *this;
    }

    // Special handling for std::endl or flush
    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        manip(ss);  // Apply the manipulator (e.g., std::endl)
        flush();
        return *this;
    }

    // Flush the log to console and GUI
    void flush() {
        std::string color;
        std::string levelStr;
        ImVec4 imguiColor;
        switch (level) {
            case LogLevel::DEBUG:
                color = COLOR_DEBUG;
                levelStr = "DEBUG";
                imguiColor = IMGUI_COLOR_DEBUG;
                break;
            case LogLevel::INFO:
                color = COLOR_INFO;
                levelStr = "INFO";
                imguiColor = IMGUI_COLOR_INFO;
                break;
            case LogLevel::WARN:
                color = COLOR_WARN;
                levelStr = "WARN";
                imguiColor = IMGUI_COLOR_WARN;
                break;
            case LogLevel::ERROR:
                color = COLOR_ERROR;
                levelStr = "ERROR";
                imguiColor = IMGUI_COLOR_ERROR;
                break;
        }
        std::string logMsg = "[" + getTimestamp() + "] " + color + "[" + levelStr + "] " + COLOR_RESET + ss.str() + "\n";
        std::cout << logMsg;  // Console output with ANSI colors
        // For GUI, store the color information (we'll handle rendering in ImGui)
        guiOutput << "[" << getTimestamp() << "] [" << levelStr << "] " << ss.str() << "\n";  // Plain text for GUI
        // Note: ImGui will apply color in the rendering step using PushStyleColor
        ss.str("");  // Clear the stringstream for the next log
    }
};

// Static methods to start a log stream with a given level
inline LogStream debug(std::stringstream& guiOutput) { return LogStream(LogLevel::DEBUG, guiOutput); }
inline LogStream info(std::stringstream& guiOutput) { return LogStream(LogLevel::INFO, guiOutput); }
inline LogStream warn(std::stringstream& guiOutput) { return LogStream(LogLevel::WARN, guiOutput); }
inline LogStream error(std::stringstream& guiOutput) { return LogStream(LogLevel::ERROR, guiOutput); }

// Optional: Allow direct enum-based access
inline LogStream operator<<(std::stringstream& guiOutput, LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return debug(guiOutput);
        case LogLevel::INFO: return info(guiOutput);
        case LogLevel::WARN: return warn(guiOutput);
        case LogLevel::ERROR: return error(guiOutput);
    }
    return info(guiOutput);  // Default fallback
}

}  // namespace MannLogger

#endif  // MANNLOGGER_HPP