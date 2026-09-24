#pragma once

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace Runs
{
inline long long NowMilliseconds()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Creating the directory reserves an ID atomically, including concurrent runs.
// Existing IDs are never overwritten. A crash can leave an incomplete directory.
class Record
{
public:
    Record(const std::string& id, const std::string& outputDirectory)
    {
        if (id.empty() || id.size() > 128)
        {
            throw std::invalid_argument("Run ID must contain 1 to 128 characters");
        }
        for (unsigned char c : id)
        {
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_'))
            {
                throw std::invalid_argument("Run ID may contain only ASCII letters, digits, '-' and '_'");
            }
        }
        if (outputDirectory.empty())
        {
            throw std::invalid_argument("Results directory must not be empty");
        }
        const auto root = std::filesystem::absolute(outputDirectory);
        std::filesystem::create_directories(root);
        // Prefix also avoids Windows reserved filenames such as CON and NUL.
        const auto directory = root / ("run-" + id);
        if (!std::filesystem::create_directory(directory))
        {
            throw std::runtime_error("Run ID already exists: " + id);
        }
        path_ = directory / "result.json";
    }

    std::string Path() const
    {
        return path_.string();
    }

    void Save(const nlohmann::json& result) const
    {
        const auto temporary = path_.parent_path() / "result.json.tmp";
        std::ofstream output;
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.open(temporary, std::ios::binary | std::ios::trunc);
        output << result.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
        output.close();
        // Readers see the completed file only after serialization succeeds.
        std::filesystem::rename(temporary, path_);
    }
private:
    std::filesystem::path path_;
};
} // namespace Runs
