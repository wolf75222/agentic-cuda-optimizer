#pragma once

#include "cuda_common.h"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

namespace Harness
{
using Json = nlohmann::json;
namespace fs = std::filesystem;

inline std::string ReadFile(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    const auto size = input.tellg();
    if (size < 0)
    {
        throw std::runtime_error("Cannot determine file size: " + path.string());
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    input.seekg(0);
    if (!data.empty() && !input.read(data.data(), size))
    {
        throw std::runtime_error("Cannot read file: " + path.string());
    }
    return data;
}

inline void WriteFile(const fs::path& path, const char* data, std::size_t size)
{
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(path, std::ios::binary | std::ios::trunc);
    output.write(data, static_cast<std::streamsize>(size));
    output.close();
}

inline fs::path Resolve(const fs::path& base, const std::string& path)
{
    if (path.empty())
    {
        throw std::invalid_argument("File/directory path must not be empty");
    }
    return fs::absolute(base / fs::path(path)).lexically_normal();
}

template <typename T>
T Integer(const Json& value)
{
    static_assert(std::is_integral_v<T>);
    if (value.is_number_unsigned())
    {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>((std::numeric_limits<T>::max)()))
        {
            throw std::out_of_range("Integer exceeds target type");
        }
        return static_cast<T>(number);
    }
    if (value.is_number_integer())
    {
        const auto number = value.get<std::int64_t>();
        if constexpr (std::is_unsigned_v<T>)
        {
            if (number < 0 || static_cast<std::uint64_t>(number) > (std::numeric_limits<T>::max)())
            {
                throw std::out_of_range("Integer exceeds target type");
            }
        }
        else
        {
            if (number < (std::numeric_limits<T>::min)() || number > (std::numeric_limits<T>::max)())
            {
                throw std::out_of_range("Integer exceeds target type");
            }
        }
        return static_cast<T>(number);
    }
    throw std::invalid_argument("Expected an integer JSON value");
}

inline CUDA::Dimensions Dimensions(const Json& value)
{
    if (!value.is_array() || value.size() != 3)
    {
        throw std::invalid_argument("Grid and block must be arrays of three positive integers");
    }
    CUDA::Dimensions dimensions
    {
        Integer<unsigned>(value.at(0)), Integer<unsigned>(value.at(1)), Integer<unsigned>(value.at(2))
    };
    if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0)
    {
        throw std::invalid_argument("Grid and block dimensions must be positive");
    }
    return dimensions;
}

struct Request
{
    std::string id;
    fs::path results;
    fs::path source;
    std::string kernel;
    std::string mode;
    int device;
    CUDA::Dimensions grid;
    CUDA::Dimensions block;
    unsigned sharedBytes;
    unsigned warmup;
    unsigned iterations;
};

inline Request Parse(const Json& json, const fs::path& base)
{
    Request request
    {
        json.at("id").get<std::string>(),
        Resolve(base, json.at("results_dir").get<std::string>()),
        Resolve(base, json.at("source").get<std::string>()),
        json.at("kernel").get<std::string>(),
        json.at("mode").get<std::string>(),
        Integer<int>(json.at("device")),
        Dimensions(json.at("grid")),
        Dimensions(json.at("block")),
        Integer<unsigned>(json.at("shared_memory_bytes")),
        Integer<unsigned>(json.at("warmup")),
        Integer<unsigned>(json.at("iterations"))
    };
    if (request.device < 0 || request.kernel.empty() || request.iterations == 0)
    {
        throw std::invalid_argument("Require device >= 0, nonempty kernel, iterations > 0");
    }
    if (request.mode != "run" && request.mode != "benchmark")
    {
        throw std::invalid_argument("Mode must be run or benchmark");
    }
    if (!json.at("arguments").is_array())
    {
        throw std::invalid_argument("arguments must be an ordered array");
    }
    return request;
}

enum class BufferDirection
{
    Input,
    Output,
    InOut
};

inline BufferDirection ParseBufferDirection(const std::string& direction)
{
    if (direction == "input")
    {
        return BufferDirection::Input;
    }
    if (direction == "output")
    {
        return BufferDirection::Output;
    }
    if (direction == "inout")
    {
        return BufferDirection::InOut;
    }
    throw std::invalid_argument("Unknown buffer direction: " + direction);
}

// Storage stays alive and at stable addresses until all launches have finished.
// Buffers are raw bytes; Python owns their dtype, shape, and meaning.
class Argument
{
public:
    Argument(const Json& spec, const fs::path& base)
    {
        const auto kind = spec.at("kind").get<std::string>();
        if (kind == "buffer")
        {
            const auto bytes = Integer<std::size_t>(spec.at("bytes"));
            if (bytes == 0)
            {
                throw std::invalid_argument("Buffer bytes must be positive");
            }
            direction_ = ParseBufferDirection(spec.at("direction").get<std::string>());
            if (direction_ == BufferDirection::Output)
            {
                if (!spec.at("input_file").is_null())
                {
                    throw std::invalid_argument("Output buffers require input_file: null; use inout to load a file");
                }
                initial_.resize(bytes, '\0');
            }
            else
            {
                if (spec.at("input_file").is_null())
                {
                    throw std::invalid_argument("Input and inout buffers require an input_file");
                }
                initial_ = ReadFile(Resolve(base, spec.at("input_file").get<std::string>()));
                if (initial_.size() != bytes)
                {
                    throw std::invalid_argument("Input file size does not match buffer bytes");
                }
            }
            buffer_ = std::make_unique<CUDA::DeviceBuffer<char>>(bytes);
            Store(buffer_->Data());
        }
        else if (kind == "scalar")
        {
            const auto type = spec.at("type").get<std::string>();
            const auto& value = spec.at("value");
            if (type == "int32")
            {
                Store(Integer<std::int32_t>(value));
            }
            else if (type == "uint32")
            {
                Store(Integer<std::uint32_t>(value));
            }
            else if (type == "int64")
            {
                Store(Integer<std::int64_t>(value));
            }
            else if (type == "uint64")
            {
                Store(Integer<std::uint64_t>(value));
            }
            else if (type == "float32" || type == "float64")
            {
                if (!value.is_number())
                {
                    throw std::invalid_argument("Floating scalar requires a JSON number");
                }
                const double number = value.get<double>();
                if (!std::isfinite(number) || (type == "float32" &&
                    std::abs(number) > (std::numeric_limits<float>::max)()))
                {
                    throw std::out_of_range("Floating scalar exceeds target type");
                }
                if (type == "float32")
                {
                    Store(static_cast<float>(number));
                }
                else
                {
                    Store(number);
                }
            }
            else
            {
                throw std::invalid_argument("Unsupported scalar type: " + type);
            }
        }
        else
        {
            throw std::invalid_argument("Argument kind must be buffer or scalar");
        }
    }

    void* Parameter() noexcept
    {
        return storage_.data();
    }
    void Reset()
    {
        if (buffer_)
        {
            buffer_->CopyFromHost(initial_.data(), initial_.size());
        }
    }
    bool IsOutput() const noexcept
    {
        return buffer_ && direction_ != BufferDirection::Input;
    }
    std::size_t Save(const fs::path& path) const
    {
        std::string data(buffer_->Size(), '\0');
        buffer_->CopyToHost(data.data(), data.size());
        WriteFile(path, data.data(), data.size());
        return data.size();
    }
private:
    template <typename T>
    void Store(T value)
    {
        static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T>);
        std::memcpy(storage_.data(), &value, sizeof(value));
    }
    alignas(8) std::array<unsigned char, 8> storage_{};
    std::unique_ptr<CUDA::DeviceBuffer<char>> buffer_;
    std::string initial_;
    BufferDirection direction_ = BufferDirection::Input;
};
} // namespace Harness
