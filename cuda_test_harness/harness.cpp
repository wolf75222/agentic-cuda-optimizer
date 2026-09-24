#include "kernel_request.h"
#include "run_record.h"
#include "device_info.h"
#include <iostream>

int main(int argc, char** argv)
{
    using namespace Harness;
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        std::cout << "cuda_test_harness --request run_info.json\n"
            "cuda_test_harness --device-info --device <ordinal>\n"
            "Compile and run one kernel. Output comparison belongs to the caller.\n";
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--device-info")
    {
        return DeviceInfoCommand(argc, argv);
    }

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
        if (argc != 3 || std::string(argv[1]) != "--request")
        {
            throw std::invalid_argument("Usage: cuda_test_harness --request run_info.json");
        }
        const auto requestPath = fs::absolute(argv[2]);
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
        CUDA::Context context(request.device);
        char deviceName[256]{};
        CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), context.Device()));
        result["device"] = deviceName;
        result["architecture"] = context.Architecture();

        stage = "compile";
        CUDA::Compilation compilation;
        const auto compileStart = std::chrono::steady_clock::now();
        try
        {
            compilation = CUDA::Compile(source, request.source.string(), context.Architecture());
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
        stage = "load";
        CUDA::Module module(compilation.cubin);
        const auto kernel = module.GetKernel(request.kernel);
        CUDA::Stream stream;

        stage = "arguments";
        std::vector<std::unique_ptr<Argument>> arguments;
        std::vector<void*> parameters;
        for (const auto& spec : json.at("arguments"))
        {
            arguments.push_back(std::make_unique<Argument>(spec, base));
            parameters.push_back(arguments.back()->Parameter());
        }
        auto reset = [&]()
        {
            for (auto& argument : arguments)
            {
                argument->Reset();
            }
        };
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
        }
        stage = "execute";
        CUDA::EventTimer timer(stream.Get());
        double elapsedMs = 0;
        for (unsigned i = 0; i < request.iterations; ++i)
        {
            reset();
            if (request.mode == "benchmark")
            {
                timer.Start();
            }
            launch();
            if (request.mode == "benchmark")
            {
                elapsedMs += timer.StopMilliseconds();
            }
            else
            {
                stream.Synchronize();
            }
        }
        if (request.mode == "benchmark")
        {
            result["latency_ms"] = elapsedMs / request.iterations;
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
    std::cout << result.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    return result["ok"].get<bool>() ? 0 : 1;
}
