/**
 * @file MannLogger.hpp
 * @brief Header file for the MannLogger namespace, providing logging functionality.
 * @author Jayansh Devgan
 * @date 2025-09-09
 * @version 1.0
 *
 * This file defines the MannLogger namespace, which provides a stream-based logging
 * system with support for different log levels (DEBUG, INFO, WARN, ERROR) and
 * integration with both console and GUI output.
 */

#ifndef MANNLOGGER_HPP
#define MANNLOGGER_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include "utils.h"

/**
 * @brief ANSI color code to reset console text formatting.
 */
#define COLOR_RESET "\033[0m"

/**
 * @brief ANSI color code for DEBUG level messages (Dark Cyan).
 */
#define COLOR_DEBUG "\033[34m"

/**
 * @brief ANSI color code for INFO level messages (Green).
 */
#define COLOR_INFO "\033[32m"

/**
 * @brief ANSI color code for WARN level messages (Yellow).
 */
#define COLOR_WARN "\033[33m"

/**
 * @brief ANSI color code for ERROR level messages (Red).
 */
#define COLOR_ERROR "\033[31m"

/**
 * @brief ImGui color definition for DEBUG level messages (Darker Cyan).
 */
#define IMGUI_COLOR_DEBUG ImVec4(0.0f, 0.5f, 0.7f, 1.0f)

/**
 * @brief ImGui color definition for INFO level messages (Darker Green).
 */
#define IMGUI_COLOR_INFO ImVec4(0.0f, 0.6f, 0.0f, 1.0f)

/**
 * @brief ImGui color definition for WARN level messages (Darker Yellow).
 */
#define IMGUI_COLOR_WARN ImVec4(0.8f, 0.5f, 0.0f, 1.0f)

/**
 * @brief ImGui color definition for ERROR level messages (Darker Red).
 */
#define IMGUI_COLOR_ERROR ImVec4(0.8f, 0.0f, 0.0f, 1.0f)

/**
 * @namespace MannLogger
 * @brief Namespace for logging-related classes and utilities.
 */
namespace MannLogger {

    /**
     * @enum LogLevel
     * @brief Enumeration of supported log levels.
     */
    enum class LogLevel {
        DEBUG,  /**< Debug level for detailed diagnostic messages. */
        INFO,   /**< Informational level for general status messages. */
        WARN,   /**< Warning level for potential issues. */
        ERROR   /**< Error level for critical problems. */
    };

    // Forward declaration of LogStream
    class LogStream;

    /**
     * @brief Retrieves the current timestamp as a formatted string.
     * @return A string containing the current date and time (e.g., "2025-09-09 12:06:00").
     */
    inline std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_struct;
        std::stringstream ss;

        errno_t result = localtime_s(&tm_struct, &time);
        (result == 0) ? ss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S")
                      : ss << "Error: Failed to get local time";
        return ss.str();
    }

    /**
     * @class LogStream
     * @brief A class representing a stream-like interface for logging messages.
     *
     * This class supports streaming of various data types and flushes the log to
     * both console and GUI output with appropriate formatting and colors based on
     * the log level.
     */
    class LogStream {
    private:
        LogLevel level;         /**< The log level for this stream. */
        std::stringstream ss;   /**< Internal stringstream to accumulate log data. */
        std::stringstream& guiOutput; /**< Reference to the GUI output stream. */

    public:
        /**
         * @brief Constructs a LogStream with the specified log level and GUI output.
         * @param lvl The log level for this stream.
         * @param gui Reference to the GUI output stringstream.
         */
        LogStream(LogLevel lvl, std::stringstream& gui) : level(lvl), guiOutput(gui) {}

        /**
         * @brief Overloads the << operator to append data to the log stream.
         * @param value The value to append (any type with << operator support).
         * @return Reference to this LogStream for chaining.
         */
        template <typename T>
        LogStream& operator<<(const T& value) {
            ss << value;
            return *this;
        }

        /**
         * @brief Overloads the << operator to handle manipulators like std::endl.
         * @param manip A manipulator function (e.g., std::endl).
         * @return Reference to this LogStream for chaining.
         */
        LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
            manip(ss);  // Apply the manipulator
            flush();
            return *this;
        }

        /**
         * @brief Flushes the accumulated log data to console and GUI output.
         *
         * Applies appropriate colors and formatting based on the log level.
         */
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
            std::string logMsg = "[CONSOLE] [" + getTimestamp() + "] " + color + "[" + levelStr + "] " + COLOR_RESET + ss.str() + "\n";
            std::cout << logMsg;  // Console output with marker
            guiOutput << "[GUI] [" << getTimestamp() << "] [" << levelStr << "] " << ss.str() << "\n";  // GUI output with marker
            ss.str("");  // Clear the stringstream
        }
    };

    /**
     * @brief Starts a log stream for DEBUG level messages.
     * @param guiOutput Reference to the GUI output stringstream.
     * @return A LogStream object for DEBUG logging.
     */
    inline LogStream debug(std::stringstream& guiOutput) { return LogStream(LogLevel::DEBUG, guiOutput); }

    /**
     * @brief Starts a log stream for INFO level messages.
     * @param guiOutput Reference to the GUI output stringstream.
     * @return A LogStream object for INFO logging.
     */
    inline LogStream info(std::stringstream& guiOutput) { return LogStream(LogLevel::INFO, guiOutput); }

    /**
     * @brief Starts a log stream for WARN level messages.
     * @param guiOutput Reference to the GUI output stringstream.
     * @return A LogStream object for WARN logging.
     */
    inline LogStream warn(std::stringstream& guiOutput) { return LogStream(LogLevel::WARN, guiOutput); }

    /**
     * @brief Starts a log stream for ERROR level messages.
     * @param guiOutput Reference to the GUI output stringstream.
     * @return A LogStream object for ERROR logging.
     */
    inline LogStream error(std::stringstream& guiOutput) { return LogStream(LogLevel::ERROR, guiOutput); }

    /**
     * @brief Overloads the << operator to start a log stream based on a LogLevel.
     * @param guiOutput Reference to the GUI output stringstream.
     * @param level The desired log level.
     * @return A LogStream object for the specified log level.
     */
    inline LogStream operator<<(std::stringstream& guiOutput, LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return debug(guiOutput);
            case LogLevel::INFO:  return info(guiOutput);
            case LogLevel::WARN:  return warn(guiOutput);
            case LogLevel::ERROR: return error(guiOutput);
        }
        return info(guiOutput);  // Default fallback
    }

}  // namespace MannLogger

#endif  // MANNLOGGER_HPP
