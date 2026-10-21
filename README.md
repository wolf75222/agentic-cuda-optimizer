# CUDA Kernel Optimizer — Codex Luna Max / ROMEO

Fork of [bertaye/agentic-cuda-optimizer](https://github.com/bertaye/agentic-cuda-optimizer), based on `1e9464da54dfc651337a97c3643bfeefec712bc1`. The primary entry point runs **Codex locally with `gpt-6-luna` and reasoning effort `max`**, keeps LangGraph orchestration, and evaluates kernels on an exclusively reserved ROMEO GPU through SSH and Slurm.

“Local agent” means the Codex process, tools and orchestration run on your workstation. Luna inference uses the authenticated OpenAI service. The original API workflow remains available and is documented in [UPSTREAM_README.md](docs/UPSTREAM_README.md).

There is **no global token, cost, GPU-time or campaign-time ceiling**. Usage is recorded for comparison without stopping the agents. The user's explicit instruction supersedes the global-budget example in the supplied specification. Iteration/repair counts and technical per-call/worker timeouts remain; Slurm also requires a job walltime.
Research also has no default search-call/source-count quota or cumulative phase timeout.

## Install and run

Controller requirements: Python 3.12+, Git, SSH, an authenticated Codex CLI, and network access for mandatory research. No local CUDA GPU is needed for ROMEO execution.

```sh
./scripts/setup.sh
codex login
ssh -o BatchMode=yes romeo hostname
./scripts/run_optimizer.sh run --config configs/gemm-romeo.json
./scripts/run_optimizer.sh resume --config configs/gemm-romeo.json
./scripts/run_optimizer.sh compare --config configs/gemm-comparison.json
.venv/bin/python -m pytest -q
```

No API key is needed for the primary CLI path when ChatGPT authentication works. A cached model list does not establish account access. The terminal CLI `0.143.0` rejected Luna, while the application’s bundled `0.163.0-alpha.10.2` was verified to run it. The backend selects `CODEX_CLI_PATH`, then an existing ChatGPT application binary on macOS, then `codex` on `PATH`. Override explicitly if needed:

```sh
export CODEX_CLI_PATH='/path/to/working/codex'
"$CODEX_CLI_PATH" login status
"$CODEX_CLI_PATH" exec --ignore-user-config --ephemeral -s read-only \
  -m gpt-6-luna -c 'model_reasoning_effort="max"' 'Reply READY without tools.'
```

Calls use isolated ephemeral directories, disabled shell tools/project document injection, structured responses, real token receipts and a timeout. The backend never silently changes model. Subscription dollar cost is recorded as unavailable.

## Configure SSH and Slurm

Use an existing SSH key/agent, with no credentials in repository configuration:

```sshconfig
Host romeo
    HostName YOUR_ROMEO_LOGIN_HOST
    User YOUR_ROMEO_USER
    IdentityFile ~/.ssh/YOUR_EXISTING_KEY
    IdentitiesOnly yes
```

Edit [configs/romeo.json](configs/romeo.json): account, partition, constraint, resource, walltime, module, private scratch root. The tested installation uses account `r250127`, partition `instant`, constraint `armgpu`, `gpu:h100:1`, and module `cuda/12.6`. These are installation-specific values.

The tested login node is x86_64; CUDA tools target the AArch64 GPU nodes. **Compilation runs inside Slurm**, with the node’s C++ compiler and NVRTC. The remote worker supports ROMEO Python 3.9 and requires no controller packages or CMake on the login node.

Only the evaluator harness/worker, candidate sources and input buffers are staged. Credentials, Git metadata and the local virtual environment are excluded. Durable submission intent and exact job receipts recover uncertain acknowledgements without blindly repeating `sbatch`. Cancellation is restricted to the session’s recorded job.

The worker keeps its native CUDA process/context and compiled modules between normal requests, serializes benchmark/profile activity and checks for foreign GPU processes. A compromised or hung context quarantines the allocation; a compatible new worker is required after resource cleanup.

## Workloads and comparison

The built-in suite supports square float32 GEMM, reduction, inclusive scan, 1D three-point stencil, three-tap convolution and RMS normalization: `matmul`, `reduction`, `scan`, `stencil`, `convolution`, `normalization`. Each has small/tail cases and a separate lazily generated final suite. Scope remains one kernel, fixed ABI/layout/precision and deterministic launch arithmetic.

Use a new `run_id` after a workload, numerical policy, protocol, GPU or incompatible toolchain change. Completed runs are not overwritten. `compare` reserves one GPU for agent and vanilla, using the same CPU oracle, initial kernel, distribution and measurement rules. Vanilla has one ordinary optimization prompt with no mandatory research or tuning and is explicitly experimental. Actual resource consumption is reported. This pilot does not establish general optimizer superiority.

The contract explicitly fixes disjoint input/output allocations, read-only inputs,
fully overwritten output, float32 arithmetic and launch independence. Direct
generated-kernel comparisons must also qualify both exact artifacts with every
configured sanitizer on every measured shape before the comparison succeeds.

If an unfinished campaign's worker has stopped, `resume --config FILE --worker-session NEW_UNIQUE_ID` starts a new isolated worker while retaining the campaign's `run_id`. Target identity is checked before combining evidence; an incompatible GPU requires a new campaign. A completed `resume` reads its manifest without allocating a GPU.

The multi-campaign protocol and isolated ablations are explicit commands:

```sh
# Writes a 90-campaign plan; submits zero jobs.
./scripts/run_optimizer.sh plan-suite --config configs/gemm-comparison.json
# Explicitly executes the planned studies (substantial GPU/model use).
./scripts/run_optimizer.sh suite --config configs/gemm-comparison.json
./scripts/run_optimizer.sh report-suite --config configs/gemm-comparison.json
```

The plan repeats three seeds over six families and full/greedy-parent/no-tuning/no-research/no-PMPP modes. Each ablation retains the other evaluation rules. The greedy mode reproduces the original incumbent-parent policy; it is not a measurement of the entire unmodified upstream program. That reference still requires its own run and resource receipts. Missing and uncertified campaigns remain visible in the report.

## Architecture and evaluation

LangGraph owns initialization → selection → research guard → experiment subgraph → durable integration → diagnosis → finalization. The subgraph bounds repairs. A separate genuine TreeQuest AB-MCTS-A tree chooses parents/actions using `ask`/`tell`. TreeQuest is pinned to `3fac4f8b07e14ed92742471492a59764993b9ab0`; its maintained eligibility extension masks invalid selections **before trial creation**. Closures have explicit candidate, parent/action, subtree or strategy-family scopes, and upstream source hashes are checked.

Research opens real CCCL/CUB, CUTLASS/CuTe, applicable library code, CUDA documentation and available PMPP material. Dossiers retain queries, retrieval status, exact provenance/content hashes and applicable techniques. A new family requires new research. Unavailable search blocks that trial; documented negative results can support an explicitly original hypothesis. The original PMPP guide remains in proposal/diagnosis context; no full-book access or unconsulted page citations are claimed.

The independent evaluator owns full outputs, read-only input checks, sentinels, numerical rules, launch arithmetic and CPU oracles. Correctness precedes benchmarking. Integer comparison is exact; tolerances/nonfinite policy are frozen. Conservative source admission rejects known precision/ABI violations, but cannot prove arbitrary CUDA semantics.

CUDA Events measure the kernel; transfers, compilation and profiling are excluded from latency. The fixed estimator is the median of series means. All tree rewards use the same baseline: `gain/(1+gain)`. Promotion requires fresh alternating A/B series, a paired bootstrap lower bound above the useful-gain threshold and critical-case limits. Finalization checks the exact artifact on held-out data, required sanitizer passes and a fresh baseline comparison. Missing tools or incomplete finalization never imply certification.

SQLite separately persists LangGraph checkpoints, TreeQuest tree/RNG/pending trials, immutable evidence, integration journal and observed resource use. A result has one tree reward; cached observations retain their IDs and intentional remeasurements receive new ones. Final exposure is persisted before execution; interrupted finalization requires a new independent holdout.

## Profiling and fallback

Nsight Compute runs on ROMEO first, in the serialized allocation, with a diagnostic question and explicit metrics. Permission errors such as `ERR_NVGPUCTRPERM` and missing tools are retained with bounded attempts. A configured local CUDA runner/Nsight can provide fallback diagnostics with its own hardware fingerprint. The tested Mac has no CUDA GPU, so local fallback is unavailable.

Local profile times/counters never enter a ROMEO ranking. Replay/cache conditions and ordinary compilation/benchmark receipts remain available when counters cannot be read.

On a CUDA-capable Linux workstation, build the existing harness and install Nsight
Compute, then set `local_profile_harness` to its absolute executable path:

```sh
cmake -S cuda_test_harness -B cuda_test_harness/build -DCMAKE_BUILD_TYPE=Release
cmake --build cuda_test_harness/build --parallel
ncu --version
```

For older completed paired measurements lacking sanitizer qualification,
`scripts/qualify_comparison.py --config FILE --worker-session NEW_UNIQUE_ID`
checks the exact artifacts in a new compatible ROMEO allocation, without a model
call or a new benchmark. Measurement and qualification contracts/jobs remain distinct.

## Artifacts and qualification

Campaign artifacts include configuration/contract/environment, both SQLite stores, baseline/candidate sources, launch rules, proposals, research dossiers, raw evaluations/confirmations/profiles, `best.cu`, `best-launch.json`, `final.json`, `manifest.json` and events. Worker artifacts include Slurm ownership/terminal receipts, native requests/results, outputs and tool logs.

Termination reason is separate from certification: `certified`, `validated_no_gain`, `validated_gain_uncertified`, `final_validation_failed`, `finalization_incomplete`. The best known source stays available on failure.

`scripts/export_evidence.py --campaign results/RUN_ID --output /absolute/output`
exports a clean committed source tree, completed comparison, successful matching
Slurm accounting, raw receipts and an HTML report. It refuses incomplete results;
the archive manifest hashes the actual exported bytes. Downloaded reference bodies
stay in the local cache; their URLs, locators and hashes remain in the dossiers.

The [specification](docs/spec.md) is preserved as requirements. [Acceptance coverage](docs/ACCEPTANCE.md) distinguishes implementation, simulated tests, GPU evidence and remaining qualification. [Validation results](docs/VALIDATION.md) records actual executions. Local tests are distinct from GitHub CI; one GPU pilot does not replace the multi-campaign six-family ablation study in specification section16.
