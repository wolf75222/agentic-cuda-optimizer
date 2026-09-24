from langchain_core.tools import tool
from use_harness import HarnessRuner
from nsight_profiler import profile_saved_kernel, read_saved_metrics

def create_tools(runner: HarnessRuner, *, use_nsight: bool = False):
    @tool
    def get_device_properties(device:int) -> dict:
        """Read CUDA device properties and hardware limits."""
        return runner.get_device_properties(device)

    @tool
    def profile_kernel(iteration: int, case_index: int) -> dict:
        """Profile an already executed candidate in this session with Nsight Compute.

        Returns hardware counters and report paths to diagnose bottlenecks, not
        benchmark scores. Uses the saved kernel and launch configuration in a
        separate run. If Nsight or counter permission is unavailable, returns an
        error; do not repeatedly retry that environmental failure.
        """
        return profile_saved_kernel(runner, iteration, case_index)

    @tool
    def read_profile_metrics(iteration: int, case_index: int, metric_pattern: str,
                             offset: int = 0, names_only: bool = False) -> dict:
        """Read collected counters from the latest successful saved Nsight profile.

        No GPU execution or profiling is performed. Use an exact metric name or
        wildcard pattern such as '*bank_conflicts*', 'smsp__warp_issue_stalled*'.
        Discover available names with metric_pattern='*', names_only=True.
        Returns up to 50 matches with names, values and units; pass next_offset as
        offset to read another page. Missing matches mean not collected, not zero.
        Profiler timing is diagnostic; rank kernels using benchmark kernel_latency_ms.
        """
        return read_saved_metrics(runner, iteration, case_index, metric_pattern,
                                  offset=offset, names_only=names_only)

    return [get_device_properties, profile_kernel, read_profile_metrics] if use_nsight else [get_device_properties]
