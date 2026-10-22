# Validation receipts — 21 October 2026

## Later scoped observations — 22 October 2026

The earlier sections below retain their historical test counts and execution
scope. The [public task observation at 06:18:25 UTC](../work/live/section16-functional-qualification-hold-public-task-observation-2.json)
records 56/144 generation tasks completed and 1/18 FINAL task completed, with
mixed source epochs. These are task states, not 56 successful methods or 56
scientifically qualified gains. Only owner job 775188 remained active in that
observation; future tasks were cooperatively held without interrupting it.
The [next public observation at 06:32:57 UTC](../work/live/section16-functional-qualification-hold-public-task-observation-4.json)
records 57/144 generation tasks completed, 87 pending, none running, and still
1/18 FINAL task completed. This is not a signed export-quiescence authority,
a completed study, or 57 successful methods or qualified gains.

The [native local MCP control 001](../results/candidate-history-native-functional-control-001/acceptance.json)
and its [independent review](../work/live/candidate-history-native-functional-control-001-peer-review.json)
establish one actual local Codex CLI invocation and one completed
`get_candidate_source` call over synthetic public records. The exact prompt
fingerprint excludes a marker returned from the served source and then from the
final structured response. Observed input/output usage is 33,994 + 465 = 34,459
tokens; 21,888 cached-input and 271 reasoning-output tokens are subsets, not
additional charges. USD cost is unknown. There was no compilation, GPU, SSH or
ROMEO execution. Requested `gpt-6-luna`/`max` does not attest provider weights or
exhaustive internal reading. Two earlier import failures are preserved as
primary-reported pre-output/pre-launch incidents, without independent public
logs or additional inferred model calls. The [source capsule](../work/live/candidate-history-native-control-source/manifest.json)
pins backend `57992232` and MCP server `6c2384be`; subsequent source changes need
their own evidence.

[Strict usage regressions](../work/live/model-usage-attribution-corrective-acceptance.json)
pass 124 CPU tests plus 8 subtests. Missing/invalid durations remain unknown,
known zero remains zero, known subtotals remain separate from complete totals,
and a missing model purpose cannot assign cost to research. No pre-gain model
usage is invented from later calls. These are synthetic CPU checks, separate
from the real MCP control and from future integrated-source CI.

The [declared global-write incident](../work/live/artifact-global-write-review-775158.json)
disqualifies artifact `146ffc0b…` / source `04e8d51f…` scientifically while
preserving its original SEARCH outputs, timings, rewards and terminal state.
This is a static source/launch finding, not a newly observed wrong output or
retroactive sanitizer failure. Compute Sanitizer `racecheck` checks shared-memory
hazards and does not certify absence of unsynchronized global writes.
[317 CPU report regressions](../work/live/report-method-comparison-qualified-summary-acceptance.json)
verify that HTML, Markdown and derived JSON honor the artifact-aware summary
instead of reviving a positive historical raw interval.

The [reconstruction guard receipt](../work/live/reconstructed-artifact-incidents-cpu-acceptance.json)
records 26 CPU tests; [one synthetic-ZIP integration test](../work/live/portable-artifact-refusal-zip-integration.xml)
checks the producer/consumer contract. A new qualification requires the observed,
authenticated refusal index before any numeric buffer read; historical
`qualified=true` cannot override it. Legacy extraction remains unexecuted and
uncertified. The [64-test integration capture](../work/live/functional-qualification-primary-after-64/manifest.json)
was taken after that run and does not cover the later malformed-usage delta,
which has its [own 24-test receipt](../work/live/functional-qualification-primary-deltas-v4-source/manifest.json).
The [original transport accounting correction](../work/live/upstream-usage-unknown-cpu-acceptance.json)
has 60 CPU tests and preserves its upstream decisions. The [initial static review](../work/live/reconstruction-usage-bounded-peer-review.json)
keeps its adapter follow-up unchanged; the [additive review](../work/live/reconstruction-usage-bounded-peer-review-v2.json)
closes it on corrected bytes. Unresolved intents/records make exact call totals
unknown, while a certain returned call stays counted when only tokens are missing.
Known call bounds require nonnegative plain integers. Orphan ledger counters
stay separate from observed consumption, and the MCP index remains in a single
JSON context frame. The [145-test CPU receipt](../work/live/functional-qualification-primary-corrections-v6-source/manifest.json)
predates the final three adapter record/known-call fixtures; subsequent complete
runs cover those bytes.

The [first complete integration run](../work/live/functional-qualification-full/acceptance.json)
remains failed: 9 failures, 2,147 passed tests and 13 subtests, 122.16 seconds;
186 declared files were stable and compileall succeeded. The [second run](../work/live/functional-qualification-full-v2/acceptance.json)
passes 2,164 tests plus 13 subtests, with no failure/skip, 135.25 seconds pytest,
186 declared stable files and successful compileall. It is preserved separately.
The final availability correction keeps a loaded-but-refused frozen artifact
available for cohort completion while leaving it unexecuted and scientifically
unqualified. Missing source remains unavailable; the [static peer](../work/live/artifact-availability-905ea769-independent-review.json)
confirms that bounded distinction.

The [final v3 run](../work/live/functional-qualification-full-v3/acceptance.json)
passes 2,165 tests plus 13 subtests, no failure/skip, 15 warnings, 136.27 seconds
pytest / 138.236 seconds driver, and successful compileall. The 186 declared
source/test/script/dependency files are hash-identical before/after; independently
edited documentation and whole-checkout Git stability are outside that scope.
This is CPU integration on the corrected final source, not published-SHA CI,
a real cold study ZIP, GPU reconstruction or new FINAL qualification. No global
resource ceiling was introduced.

## Local policy and integration checks

`python -m pytest -q`: **216 passed** on the local Python 3.12 environment.
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

The uncapped end-to-end comparison uses
`configs/gemm-comparison.json` (`gemm-e2e-20261021`) and one shared allocation. It evaluates
the naive GEMM baseline, the LangGraph/TreeQuest/research/autotuning agent and
a vanilla single-prompt Luna Max proposal. Generation runs locally with the
application's verified Codex CLI 0.163.0-alpha.10.2, `gpt-6-luna`, effort `max`.

Earlier pilots are retained as failed/incomplete evidence. In particular, the
research call in job 770104 completed with 1,290,314 tokens; generation correctly
blocked because the described contract omitted buffer-alias semantics. The
release now supplies the native harness's disjoint-allocation/read-only/full-output
rules explicitly and distinguishes adaptation prerequisites from rejected
unchanged reference implementations. This failure was not a token limit.

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
