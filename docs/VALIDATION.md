# Validation receipts — 21 October 2026

## Local policy and integration checks

`python -m pytest -q`: **148 passed** on the local Python 3.12 environment.
Shell entry points pass syntax checks; Python sources pass compilation checks.
The six-family/three-seed ablation planner produces 90 campaign entries and submits
zero jobs. Those planned studies have not been executed and establish no general
optimizer improvement.

Tests exercise genuine pinned TreeQuest, transactional result replay, eligibility,
the greedy-parent ablation, paired measurement policy, conservative ABI/precision
admission, exact final artifact checks, research provenance and transport errors.
Simulated runners/providers in those tests do not establish GPU performance.

The later user instruction removes global token/cost/GPU/campaign-time ceilings.
A regression records 3,000,001 model tokens and successfully performs another call.
Usage remains observable. Iteration/repair limits and technical timeouts remain.

## Real ROMEO GPU checks

Actual device: **NVIDIA GH200 120GB**, architecture `sm_90`, AArch64 node
`romeo-a057`, CUDA/NVRTC 12.6, driver API 13010. Slurm exposes this resource under
the `gpu:h100:1` name; that scheduler name does not change the observed device.
Compilation and execution took place inside allocations, never on the login node.

| Job | Work | Terminal evidence |
| --- | --- | --- |
| 769478 | Naive GEMM against independent CPU oracle, hidden shape96, all four Compute Sanitizer tools. | COMPLETED,0:0,4m37s. Baseline validated, no optimization gain claimed.|
| 769644 | Correct output; partial-output rejection before benchmark; compile-error recovery in the persistent process; native cache miss/hit/source/options invalidation; deliberate read-only input modification. | FAILED,3:0,1m02s, intentionally quarantined after the destructive final probe.|
| 769661 | A new clean worker on the same physical GPU, independently validated after retirement of the compromised worker. | COMPLETED,0:0,17s.|

The native cache sequence was `[miss, hit, miss, hit, miss]` for initial compile,
same request, changed options, same changed options and changed source. A partial
output kernel received no benchmark request. Illegal input mutation produced
`worker_reusable=false`; subsequent accepted measurements used a new allocation.
The existing comparison job 769515 was preserved during these independent probes.

The JSON acceptance summary is in [validation/gpu-acceptance.json](validation/gpu-acceptance.json).
Raw requests, outputs, CUDA receipts, tool logs and scheduler accounting are retained
in the local validation bundle. These checks establish the evaluator/worker path,
not a performance gain from an optimizer.

## Agent versus vanilla comparison

The current uncapped end-to-end comparison uses
`configs/gemm-comparison.json` and one shared allocation, job 769874. It evaluates
the naive GEMM baseline, the LangGraph/TreeQuest/research/autotuning agent and
a vanilla single-prompt Luna Max proposal. Generation runs locally with the
application's verified Codex CLI 0.163.0-alpha.10.2, `gpt-6-luna`, effort`max`.

Final candidate identities, correctness results, fresh paired series and intervals
will be recorded only after execution finishes. A source, model response or job
start is not reported as a completed optimization. The comparison is one workload
and one campaign per mode, separate from the full multi-campaign quality study.

## Profiling and CI boundaries

ROMEO-first profiling and explicit local fallback are implemented. Actual tool and
counter access outcomes belong to the campaign profile receipt; binary presence
alone does not establish counter permission. This Mac has no local CUDA GPU, so
local timing cannot substitute for ROMEO measurements.

The GitHub Actions workflow checks policy code on Python 3.12. A local passing suite
is distinct from the remote workflow on the published commit; neither is a
multi-workload scientific qualification.
