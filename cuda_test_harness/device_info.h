#pragma once

#include "cuda_common.h"
#include <nlohmann/json.hpp>
#include <charconv>
#include <iostream>

namespace Harness
{
inline nlohmann::json GetDeviceProperties(int ordinal)
{
    CUDA::Context context(ordinal);
    const auto device = context.Device();
    auto attribute = [device](CUdevice_attribute name)
    {
        int value = 0;
        CUDA_CHECK(cuDeviceGetAttribute(&value, name, device));
        return value;
    };

    char name[256]{};
    CUDA_CHECK(cuDeviceGetName(name, sizeof(name), device));
    std::size_t totalBytes = 0;
    std::size_t freeBytes = 0;
    CUDA_CHECK(cuMemGetInfo(&freeBytes, &totalBytes));

    return nlohmann::json
    {
        {"device", ordinal},
        {"name", name},
        {"architecture", context.Architecture()},
        {"compute_capability", {
            {"major", attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)},
            {"minor", attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)}}},
        {"multiprocessor_count", attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)},
        {"l2_cache_bytes", attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)},
        {"warp_size", attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE)},
        {"max_threads_per_block", attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)},
        {"max_threads_per_multiprocessor", attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)},
        {"max_block_dimensions", {
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
            attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)}},
        {"max_grid_dimensions", {
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
            attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)}},
        {"max_shared_memory_per_block_bytes", attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)},
        {"max_shared_memory_per_block_optin_bytes", attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)},
        {"max_shared_memory_per_multiprocessor_bytes", attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)},
        {"max_registers_per_block", attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)},
        {"max_registers_per_multiprocessor", attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)},
        {"total_memory_bytes", totalBytes},
        {"free_memory_bytes", freeBytes}
    };
}

inline int DeviceInfoCommand(int argc, char** argv)
{
    nlohmann::json result;
    try
    {
        if (argc != 4 || std::string(argv[2]) != "--device")
        {
            throw std::invalid_argument("Usage: cuda_test_harness --device-info --device <ordinal>");
        }
        const std::string text = argv[3];
        int ordinal = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), ordinal);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || ordinal < 0)
        {
            throw std::invalid_argument("Device ordinal must be a nonnegative integer");
        }
        result["properties"] = GetDeviceProperties(ordinal);
        result["ok"] = true;
    }
    catch (const std::exception& error)
    {
        result["ok"] = false;
        result["error"] = error.what();
    }
    std::cout << result.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
    return result["ok"].get<bool>() ? 0 : 1;
}
} // namespace Harness
