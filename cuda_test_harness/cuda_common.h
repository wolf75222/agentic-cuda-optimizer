#pragma once

#include <cuda.h>
#include <nvrtc.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace CUDA
{
inline void Check(CUresult status, const char* expression, const char* file, int line)
{
    if (status == CUDA_SUCCESS)
    {
        return;
    }
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + expression +
        ": " + (name ? name : "Unknown CUDA error") + " (" +
        (description ? description : "No description") + ")");
}

inline void CheckNVRTC(nvrtcResult status)
{
    if (status != NVRTC_SUCCESS)
    {
        throw std::runtime_error(nvrtcGetErrorString(status));
    }
}

inline void Cleanup(CUresult status, const char* operation) noexcept
{
    if (status != CUDA_SUCCESS)
    {
        std::fprintf(stderr, "%s cleanup failed: %d\n", operation, static_cast<int>(status));
    }
}
}

#define CUDA_CHECK(expression) ::CUDA::Check((expression), #expression, __FILE__, __LINE__)

namespace CUDA
{
inline std::size_t CeilDiv(std::size_t count, std::size_t divisor)
{
    if (divisor == 0)
    {
        throw std::invalid_argument("CeilDiv divisor must be nonzero");
    }
    return count / divisor + (count % divisor != 0);
}

// Single-thread scope: construct first and destroy last. All resources below
// must be used and destroyed with this context current on the calling thread.
class Context
{
public:
    explicit Context(int ordinal = 0)
    {
        CUDA_CHECK(cuInit(0));
        CUDA_CHECK(cuDeviceGet(&device_, ordinal));
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
        try
        {
            CUDA_CHECK(cuCtxPushCurrent(context_));
        }
        catch (...)
        {
            Cleanup(cuDevicePrimaryCtxRelease(device_), "cuDevicePrimaryCtxRelease");
            throw;
        }
    }
    ~Context() noexcept
    {
        CUcontext popped = nullptr;
        Cleanup(cuCtxPopCurrent(&popped), "cuCtxPopCurrent");
        Cleanup(cuDevicePrimaryCtxRelease(device_), "cuDevicePrimaryCtxRelease");
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    CUdevice Device() const noexcept
    {
        return device_;
    }
    std::string Architecture() const
    {
        int major = 0;
        int minor = 0;
        CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
        CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));
        return "sm_" + std::to_string(major) + std::to_string(minor);
    }
private:
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
};

struct Compilation
{
    std::string cubin;
    std::string log;
};

class Program
{
public:
    Program(const std::string& source, const std::string& name)
    {
        // Use NVIDIA's device-compatible integer definitions, not host CRT headers.
        const char* headerNames[] = {"stdint.h"};
        const char* headers[] = {"#pragma once\n#include <cuda/std/cstdint>\n"};
        CheckNVRTC(nvrtcCreateProgram(&program_, source.c_str(), name.c_str(), 1, headers, headerNames));
    }
    ~Program() noexcept
    {
        const auto status = nvrtcDestroyProgram(&program_);
        if (status != NVRTC_SUCCESS)
        {
            std::fprintf(stderr, "nvrtcDestroyProgram: %s\n", nvrtcGetErrorString(status));
        }
    }
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    nvrtcProgram Get() const noexcept
    {
        return program_;
    }
private:
    nvrtcProgram program_ = nullptr;
};

inline Compilation Compile(const std::string& source, const std::string& name,
    const std::string& architecture)
{
    Program program(source, name);
    const std::string archOption = "--gpu-architecture=" + architecture;
    const std::string includeOption = "--include-path=" CUDA_HARNESS_INCLUDE_DIR;
    const std::string ccclOption = includeOption + "/cccl";
    const char* options[] = {
        "--std=c++17", archOption.c_str(), "--generate-line-info",
        includeOption.c_str(), ccclOption.c_str()
    };
    const auto status = nvrtcCompileProgram(program.Get(), 5, options);
    Compilation result;
    std::size_t length = 0;
    CheckNVRTC(nvrtcGetProgramLogSize(program.Get(), &length));
    result.log.resize(length);
    CheckNVRTC(nvrtcGetProgramLog(program.Get(), result.log.data()));
    if (!result.log.empty() && result.log.back() == '\0')
    {
        result.log.pop_back();
    }
    if (status != NVRTC_SUCCESS)
    {
        throw std::runtime_error(std::string(nvrtcGetErrorString(status)) + "\n" + result.log);
    }
    CheckNVRTC(nvrtcGetCUBINSize(program.Get(), &length));
    result.cubin.resize(length);
    CheckNVRTC(nvrtcGetCUBIN(program.Get(), result.cubin.data()));
    return result;
}

class Module
{
public:
    explicit Module(const std::string& cubin)
    {
        CUDA_CHECK(cuModuleLoadData(&module_, cubin.data()));
    }
    ~Module() noexcept
    {
        Cleanup(cuModuleUnload(module_), "cuModuleUnload");
    }
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    CUfunction GetKernel(const std::string& name) const
    {
        CUfunction function = nullptr;
        CUDA_CHECK(cuModuleGetFunction(&function, module_, name.c_str()));
        return function;
    }
private:
    CUmodule module_ = nullptr;
};

class Stream
{
public:
    Stream()
    {
        CUDA_CHECK(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
    }
    ~Stream() noexcept
    {
        Cleanup(cuStreamDestroy(stream_), "cuStreamDestroy");
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    CUstream Get() const noexcept
    {
        return stream_;
    }
    void Synchronize() const
    {
        CUDA_CHECK(cuStreamSynchronize(stream_));
    }
private:
    CUstream stream_ = nullptr;
};

template <typename T>
class DeviceBuffer
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    explicit DeviceBuffer(std::size_t count = 0) : count_(count)
    {
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
        {
            throw std::length_error("DeviceBuffer size overflow");
        }
        if (count != 0)
        {
            CUDA_CHECK(cuMemAlloc(&data_, SizeBytes()));
        }
    }
    ~DeviceBuffer() noexcept
    {
        Release();
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, 0)), count_(std::exchange(other.count_, 0))
    {
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            Release();
            data_ = std::exchange(other.data_, 0);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    CUdeviceptr Data() const noexcept
    {
        return data_;
    }
    std::size_t Size() const noexcept
    {
        return count_;
    }
    std::size_t SizeBytes() const noexcept
    {
        return count_ * sizeof(T);
    }
    void CopyFromHost(const T* source, std::size_t count)
    {
        Validate(source, count);
        if (count != 0)
        {
            CUDA_CHECK(cuMemcpyHtoD(data_, source, count * sizeof(T)));
            // Pageable uploads may return after staging. Complete the default-
            // stream transfer before a nonblocking launch stream consumes it.
            CUDA_CHECK(cuStreamSynchronize(nullptr));
        }
    }
    // Synchronize the launching stream before downloading its results.
    void CopyToHost(T* destination, std::size_t count) const
    {
        Validate(destination, count);
        if (count != 0)
        {
            CUDA_CHECK(cuMemcpyDtoH(destination, data_, count * sizeof(T)));
        }
    }
    void Zero()
    {
        if (count_ != 0)
        {
            CUDA_CHECK(cuMemsetD8(data_, 0, SizeBytes()));
            CUDA_CHECK(cuStreamSynchronize(nullptr));
        }
    }
private:
    void Validate(const void* host, std::size_t count) const
    {
        if (count > count_)
        {
            throw std::out_of_range("Copy exceeds device allocation");
        }
        if (count != 0 && host == nullptr)
        {
            throw std::invalid_argument("Null host buffer");
        }
    }
    void Release() noexcept
    {
        if (data_ != 0)
        {
            Cleanup(cuMemFree(data_), "cuMemFree");
        }
    }
    CUdeviceptr data_ = 0;
    std::size_t count_ = 0;
};

struct Dimensions
{
    unsigned x = 1;
    unsigned y = 1;
    unsigned z = 1;
};

// Argument types and order must match the compiled kernel's ABI exactly.
// Device pointers are CUdeviceptr values, not host addresses.
template <typename... Args>
inline void Launch(CUfunction kernel, Dimensions grid, Dimensions block,
    CUstream stream, unsigned sharedBytes, Args... args)
{
    static_assert((std::is_trivially_copyable_v<Args> && ...));
    std::vector<void*> parameters{static_cast<void*>(&args)...};
    CUDA_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
        block.x, block.y, block.z, sharedBytes, stream,
        parameters.empty() ? nullptr : parameters.data(), nullptr));
}

class EventTimer
{
public:
    explicit EventTimer(CUstream stream) : stream_(stream)
    {
        CUDA_CHECK(cuEventCreate(&start_, CU_EVENT_DEFAULT));
        try
        {
            CUDA_CHECK(cuEventCreate(&stop_, CU_EVENT_DEFAULT));
        }
        catch (...)
        {
            Cleanup(cuEventDestroy(start_), "cuEventDestroy");
            throw;
        }
    }
    ~EventTimer() noexcept
    {
        Cleanup(cuEventDestroy(stop_), "cuEventDestroy");
        Cleanup(cuEventDestroy(start_), "cuEventDestroy");
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;
    void Start()
    {
        CUDA_CHECK(cuEventRecord(start_, stream_));
        started_ = true;
    }
    float StopMilliseconds()
    {
        if (!started_)
        {
            throw std::logic_error("Timer not started");
        }
        CUDA_CHECK(cuEventRecord(stop_, stream_));
        CUDA_CHECK(cuEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cuEventElapsedTime(&milliseconds, start_, stop_));
        started_ = false;
        return milliseconds;
    }
private:
    CUstream stream_ = nullptr;
    CUevent start_ = nullptr;
    CUevent stop_ = nullptr;
    bool started_ = false;
};
} // namespace CUDA
