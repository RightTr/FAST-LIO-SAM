#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

inline bool create_directory(const std::string& path) {
    if (fs::exists(path)) {
        return fs::is_directory(path);
    }
    return fs::create_directories(path);
}

inline std::string format_unix_time(double t) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9) << t;
    return oss.str();
}
