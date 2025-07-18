#pragma once

#include <iostream>
#include <string>

namespace cpl {
namespace logging {

enum class LogLevel {
    NONE,
    ERROR,
    WARN,
    INFO,
    DEBUG
};

// Global log level for the library
inline LogLevel current_level = LogLevel::INFO;

inline void set_log_level(LogLevel level) {
    current_level = level;
}

template<typename... Args>
inline void log(LogLevel level, const Args&... args) {
    if (level <= current_level && current_level != LogLevel::NONE) {
        // Simple console logging
        ((std::cout << args), ...);
        std::cout << std::endl;
    }
}

template<typename... Args>
inline void log_error(const Args&... args) {
    log(LogLevel::ERROR, "ERROR: ", args...);
}

template<typename... Args>
inline void log_warn(const Args&... args) {
    log(LogLevel::WARN, "WARN: ", args...);
}

template<typename... Args>
inline void log_info(const Args&... args) {
    log(LogLevel::INFO, "INFO: ", args...);
}

template<typename... Args>
inline void log_debug(const Args&... args) {
    log(LogLevel::DEBUG, "DEBUG: ", args...);
}

} // namespace logging
} // namespace cpl 