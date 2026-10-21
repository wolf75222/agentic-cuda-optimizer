#include "kernel_request.h"
#include "run_record.h"
#include "device_info.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <numeric>

using namespace Harness;

struct WorkerState
{
    std::unique_ptr<CUDA::Context> context;
    int device = -1;
    Json properties;
    std::map<std::string, std::unique_ptr<CUDA::Module>> modules;
};

// Integrity check complements the evaluator's SHA-256 compilation identity.
static std::string BinaryChecksum(const std::string& data)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (unsigned char byte : data) value = (value ^ byte) * 1099511628211ULL;
    return std::to_string(value);
}

static Json RunRequest(const fs::path& requestPath, WorkerState& worker)
{

    std::string stage = "request";
    std::unique_ptr<Runs::Record> record;
    const auto start = std::chrono::steady_clock::now();
    Json result;
    result["schema_version"] = 3;
    result["started_at_unix_ms"] = Runs::NowMilliseconds();
    result["compile_ms"] = nullptr;
    result["latency_ms"] = nullptr;
    result["outputs"] = Json::array();
    try
    {
        const auto base = requestPath.parent_path();
        const auto requestText = ReadFile(requestPath);
        const auto json = Json::parse(requestText);
        const auto request = Parse(json, base);
        result["id"] = request.id;
        stage = "record";
        record = std::make_unique<Runs::Record>(request.id, request.results.string());
        const auto directory = fs::path(record->Path()).parent_path();
        result["result_path"] = record->Path();
        result["request_path"] = requestPath.string();
        WriteFile(directory / "run_info.json", requestText.data(), requestText.size());

        stage = "source";
        const auto source = ReadFile(request.source);
        WriteFile(directory / "kernel.cu", source.data(), source.size());
        stage = "context";
        if (!worker.context || worker.device != request.device)
        {
            worker.modules.clear();
            worker.context.reset();
            worker.context = std::make_unique<CUDA::Context>(request.device);
            worker.device = request.device;
            worker.properties = GetDeviceProperties(request.device);
        }
        auto& context = *worker.context;
        result["environment"] = worker.properties;
        result["worker_reusable"] = true;
        char deviceName[256]{};
        CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), context.Device()));
        result["device"] = deviceName;
        result["architecture"] = context.Architecture();

        stage = "compile";
        CUDA::Compilation compilation;
        const auto compileStart = std::chrono::steady_clock::now();
        Json cacheIdentity = {{"key", request.cacheKey}, {"source", source},
            {"options", request.compilerOptions}, {"architecture", context.Architecture()},
            {"driver_version", worker.properties.at("driver_version")},
            {"nvrtc_version", worker.properties.at("nvrtc_version")},
            {"toolkit_version", worker.properties.at("cuda_toolkit_version")},
            {"harness_build_id", CUDA_HARNESS_BUILD_ID}};
        fs::path cachePath;
        result["compile_cache_hit"] = false;
        if (!request.cacheDirectory.empty() && !request.cacheKey.empty() &&
            std::string(CUDA_HARNESS_BUILD_ID) != "unidentified-build-cache-disabled")
        {
            fs::create_directories(request.cacheDirectory);
            cachePath = request.cacheDirectory / (request.cacheKey + ".cubin");
            try
            {
                const auto receipt = Json::parse(ReadFile(cachePath.string() + ".json"));
                if (receipt.at("identity") == cacheIdentity)
                {
                    compilation.cubin = ReadFile(cachePath);
                    if (receipt.at("checksum").get<std::string>() != BinaryChecksum(compilation.cubin))
                        compilation.cubin.clear();
                    else result["compile_cache_hit"] = true;
                }
            }
            catch (const std::exception&) { compilation.cubin.clear(); }
        }
        try
        {
            if (compilation.cubin.empty())
            {
                auto options = request.compilerOptions;
                options.push_back("--include-path=" + request.source.parent_path().string());
                compilation = CUDA::Compile(source, request.source.string(), context.Architecture(), options);
                if (!cachePath.empty())
                {
                    const auto temporary = cachePath.string() + "." + request.id + ".tmp";
                    WriteFile(temporary, compilation.cubin.data(), compilation.cubin.size());
                    fs::rename(temporary, cachePath);
                    Json receipt = {{"identity", cacheIdentity}, {"checksum", BinaryChecksum(compilation.cubin)}};
                    const auto text = receipt.dump();
                    WriteFile(temporary, text.data(), text.size());
                    fs::rename(temporary, cachePath.string() + ".json");
                }
            }
        }
        catch (const std::exception& error)
        {
            result["compile_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - compileStart).count();
            result["compiler_log"] = error.what();
            throw;
        }
        result["compile_ms"] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - compileStart).count();
        result["compiler_log"] = compilation.log;
        result["compilation_identity"] = cacheIdentity;
        stage = "load";
        // Safe reuse requires the complete immutable compilation key.
        const auto moduleKey = request.cacheKey.empty() ? request.id : request.cacheKey;
        if (!worker.modules.count(moduleKey))
            worker.modules[moduleKey] = std::make_unique<CUDA::Module>(compilation.cubin);
        const auto kernel = worker.modules.at(moduleKey)->GetKernel(request.kernel);
        auto kernelAttribute = [kernel](CUfunction_attribute attribute)
        {
            int value = 0;
            CUDA_CHECK(cuFuncGetAttribute(&value, attribute, kernel));
            return value;
        };
        result["kernel_resources"] = {
            {"registers_per_thread", kernelAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)},
            {"static_shared_memory_bytes", kernelAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)},
            {"local_memory_bytes", kernelAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)},
            {"max_threads_per_block", kernelAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)}};
        CUDA::Stream stream;

        stage = "arguments";
        std::vector<std::unique_ptr<Argument>> arguments;
        std::vector<void*> parameters;
        for (const auto& spec : json.at("arguments"))
        {
            arguments.push_back(std::make_unique<Argument>(spec, base, request.guardOutputs, request.sentinelByte));
            parameters.push_back(arguments.back()->Parameter());
        }
        auto reset = [&]()
        {
            for (auto& argument : arguments)
            {
                argument->Reset();
            }
        };
        auto checkReadOnly = [&]()
        {
            for (auto& argument : arguments)
                if (!argument->CheckReadOnly())
                    throw std::runtime_error("Kernel modified a read-only input");
        };
        result["guard_readonly_ok"] = false;
        result["guard_outputs_poisoned"] = request.guardOutputs;
        auto launch = [&]()
        {
            CUDA_CHECK(cuLaunchKernel(kernel,
                request.grid.x, request.grid.y, request.grid.z,
                request.block.x, request.block.y, request.block.z,
                request.sharedBytes, stream.Get(),
                parameters.empty() ? nullptr : parameters.data(), nullptr));
        };

        stage = "warmup";
        for (unsigned i = 0; i < request.warmup; ++i)
        {
            reset();
            launch();
            stream.Synchronize();
            checkReadOnly();
        }
        stage = "execute";
        CUDA::EventTimer timer(stream.Get());
        Json rawSeries = Json::array();
        Json rawSamples = Json::array();
        std::vector<double> seriesMeans;
        for (unsigned series = 0; series < request.series; ++series)
        {
            Json samples = Json::array();
            double elapsedMs = 0;
            for (unsigned i = 0; i < request.iterations; ++i)
            {
                reset();
                if (request.mode == "benchmark") timer.Start();
                launch();
                if (request.mode == "benchmark")
                {
                    const auto elapsed = timer.StopMilliseconds();
                    elapsedMs += elapsed;
                    samples.push_back(elapsed);
                    rawSamples.push_back(elapsed);
                }
                else stream.Synchronize();
                // Check before the next reset can erase evidence of illicit writes.
                stage = "guards";
                checkReadOnly();
                stage = "execute";
            }
            if (request.mode == "benchmark")
            {
                seriesMeans.push_back(elapsedMs / request.iterations);
                rawSeries.push_back(samples);
            }
        }
        result["guard_readonly_ok"] = true;
        result["latency_samples_ms"] = rawSamples;
        result["latency_series_ms"] = rawSeries;
        result["measurement_protocol"] = {{"warmup", request.warmup}, {"iterations", request.iterations},
            {"series", request.series}, {"estimator", "median_of_series_means"},
            {"scope", "cuda_events_kernel_only"}, {"data_reset", "every_launch_outside_timing"}};
        if (request.mode == "benchmark")
        {
            std::sort(seriesMeans.begin(), seriesMeans.end());
            const auto mid = seriesMeans.size() / 2;
            result["latency_ms"] = seriesMeans.size() % 2 ? seriesMeans[mid] : (seriesMeans[mid - 1] + seriesMeans[mid]) / 2;
        }

        stage = "outputs";
        for (std::size_t i = 0; i < arguments.size(); ++i)
        {
            if (arguments[i]->IsOutput())
            {
                const auto path = directory / ("output-" + std::to_string(result["outputs"].size()) + ".bin");
                Json output;
                output["bytes"] = arguments[i]->Save(path);
                output["path"] = path.string();
                result["outputs"].push_back(std::move(output));
            }
        }
        result["ok"] = true;
        result["stage"] = "complete";
    }
    catch (const std::exception& error)
    {
        result["ok"] = false;
        result["stage"] = stage;
        result["error"] = error.what();
        // A CUDA failure may poison the context; no later measurement can reuse it.
        if (stage == "context" || stage == "arguments" || stage == "warmup" ||
            stage == "execute" || stage == "guards" || stage == "outputs" || stage == "load")
            result["worker_reusable"] = false;
    }
    result["finished_at_unix_ms"] = Runs::NowMilliseconds();
    result["duration_ms"] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (record)
    {
        try
        {
            record->Save(result);
        }
        catch (const std::exception& error)
        {
            result["ok"] = false;
            result["persistence_error"] = error.what();
        }
    }
    return result;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        std::cout << "cuda_test_harness --request run_info.json\n"
            "cuda_test_harness --device-info --device <ordinal>\n"
            "cuda_test_harness --worker (JSONL request_path messages on stdin)\n";
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--device-info") return DeviceInfoCommand(argc, argv);
    WorkerState worker;
    if (argc == 2 && std::string(argv[1]) == "--worker")
    {
        std::string line;
        while (std::getline(std::cin, line))
        {
            Json result;
            try
            {
                const auto envelope = Json::parse(line);
                result = RunRequest(fs::absolute(envelope.at("request_path").get<std::string>()), worker);
            }
            catch (const std::exception& error)
            {
                result = {{"ok", false}, {"stage", "request"}, {"error", error.what()}, {"worker_reusable", true}};
            }
            std::cout << result.dump(-1, ' ', false, Json::error_handler_t::replace) << std::endl;
            if (!result.value("worker_reusable", true)) return 2;
        }
        return 0;
    }
    if (argc != 3 || std::string(argv[1]) != "--request")
    {
        std::cout << Json({{"ok", false}, {"stage", "request"}, {"error", "Usage: --request path | --worker"}}).dump() << '\n';
        return 1;
    }
    const auto result = RunRequest(fs::absolute(argv[2]), worker);
    std::cout << result.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    return result["ok"].get<bool>() ? 0 : 1;
}
