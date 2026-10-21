# Acceptance evidence and qualification limits

The user's later instruction overrides specification section11.3's global resource
ceilings: normal execution records tokens/GPU use but does not impose a global
token, cost or time budget. Iteration/repair limits and technical timeouts remain.
The ledger's optional reservation APIs are unit-tested infrastructure, not an
active global cap on agents. Historical failed pilots preserve their original
receipts and are not presented as the current configuration.

This matrix maps the supplied specification's AC-01–AC-20 to implementation
and reproducible tests. An implemented guard, a simulated test, a local Codex
call, a Slurm submission and a completed CUDA measurement establish different
facts. This file does not certify an optimizer improvement or a successful
ROMEO campaign. Actual campaign results belong in the run manifest and report.

The repository retains the upstream LangGraph path. The new campaign uses local
Codex CLI with the explicitly configured model and reasoning effort, LangGraph
checkpoints, pinned TreeQuest, a durable experiment ledger and a separate CUDA
worker. The model is hosted by the configured Codex provider; “local” describes
the agent process and orchestration, not on-device model weights.

## Functional matrix

| Criterion | Implementation and automated evidence | Qualification still needed |
| --- | --- | --- |
| AC-01: research before generation | `campaign_workflow.py` routes research before experiments; `ResearchDossier.research_ready` verifies identity, read receipts, source hashes and substantive dossier fields. `test_research_blocks_generation_and_exploration_continues`, `test_boolean_or_search_snippet_cannot_satisfy_guard`. | A live run must retain actual web-search events, fetched sources, dossier and generated candidate. |
| AC-02: family and uncovered techniques | Dossiers bind contract, environment and family; proposals must stay within `covered_techniques`. `NeedsResearch` routes back to research with a bounded round count. `test_read_receipts_environment_and_new_technique`. | Observe a real uncovered-technique return and compatible reuse in a live campaign; the current unit test covers the deterministic guard. |
| AC-03: unavailable web versus negative findings | Missing observed web-tool queries block research; transport/fetch failures are persisted. A negative finding requires rejected examples and an explicit original hypothesis, with CUDA and PMPP evidence still present. `test_failed_live_search_stays_blocked_and_persists_reason`. | Exercise an actual provider/fetch outage and a complete negative-search dossier. Simulated sources do not establish live web availability. |
| AC-04: PMPP provenance | Original `PMPP_GUIDANCE` is retained. Ten installed topic cards report project authorship, referenced edition, source path and content hash; full book access is never claimed. Guidance enters generation and diagnostic context. `test_pmpp_original_guide_retained_and_cards_provenance`. | Any external companion material must have its own successful fetch receipt. The installed cards are original guidance, not book excerpts or a book index. |
| AC-05: exact selected parent | TreeQuest's selected parent source and parameters are archived and passed to generation; structured proposals validate both selected and effective source parent. `test_upstream_pin_and_exact_slower_parent`, `test_exact_parent_contract_and_launch_types`. | Inspect the selected-parent and generation receipts in a completed run, including a branch below the incumbent. |
| AC-06: closure and bounded repairs | Eligibility masks operate before trial creation; closures have explicit scope, and experiment repair counts are bounded. `test_scope_masks_before_trial_creation_and_batch`, `test_invalid_kernel_does_not_close_valid_parent`, `test_repairs_bounded_per_trial_and_checkpoint_replay`. | Exercise these transitions with real compilation/numerical failures and confirm unrelated branches continue. |
| AC-07: stationary reward | The reward is a fixed-baseline weighted geometric gain mapped monotonically to `[0,1]`. `test_stationary_score_and_small_gains`. | Check fixed case identities and measured latencies in real reward receipts. |
| AC-08: incorrect/partial output or modified input | Independent CPU oracle, short and complete validation, repeated correctness checks, output sentinel initialization and input integrity checks precede benchmarking. `test_invalid_candidates_never_benchmark`, `test_input_modification_is_rejected_before_gpu`. | CUDA execution and Compute Sanitizer must detect intentionally incorrect, partially written and illegally modifying kernels on the target. Mocked evaluator outputs alone do not establish device behavior. |
| AC-09: immutable numerical contract | Contract/test metadata and candidate source/parameters are frozen and hashed; provider output cannot supply tolerances or oracle code. `test_candidate_source_parameters_and_contract_are_pinned`. | A frozen contract does not by itself prove that every arbitrary CUDA implementation preserves an arithmetic-precision policy. Explicit precision-changing kernels and compiler-option changes need target rejection tests. |
| AC-10: noise and critical regressions | Fresh interleaved candidate/incumbent series, whole-series bootstrap intervals and critical-case checks gate promotion. `test_confirmation_alternates_pairs_and_uses_new_samples`, `test_bootstrap_resamples_complete_series_and_critical_regression`. | Complete target-side paired measurements under the fixed protocol; single best timings are insufficient. |
| AC-11: promotion and exact final artifact | The ledger promotes against an incumbent version atomically. Finalization validates an exact candidate on the withheld suite and records sanitizer availability. `test_finalization_protected_compatibility_and_atomic_promotion`, `test_final_holdout_is_one_use_and_artifact_is_exact`. | A completed final manifest must name the exact artifact, new final measurements and required sanitizer results. A local test or a submitted Slurm job is not final certification. |
| AC-12: bounded autotuning | Declarative integer domains/bindings, hardware/restriction preflight, common validation, observed usage and independent winner remeasurement are provided by `autotuning.py`. `test_tuning_filters_validates_and_remeasures_winner`, `test_tuning_explicit_source_and_launch_bindings`. | The campaign must invoke this service when tuning is enabled and retain each evaluated configuration and the fresh winner measurement. Target execution remains required. |
| AC-13: compilation cache | Source/dependency/options/target/toolchain manifests and binary content checks invalidate stale entries. `test_compilation_cache_invalidates_content_dependencies_options_target_and_toolchain`. | Demonstrate a target binary cache hit and misses after each relevant mutation. The Python manifest test does not alone establish the harness's binary-cache behavior. |
| AC-14: exclusive GPU access | A serial Slurm worker handles benchmark/profile requests; scheduler reservations exclude concurrent use. `test_worker_serializes_benchmark_profile_requests`, `test_gpu_reservations_exclude_other_threads_and_detect_interference`. | Confirm the actual exclusive allocation and absence of competing GPU processes in target receipts. A process-local lock cannot exclude unrelated allocations by itself. |
| AC-15: compromised context | A persistent native process retains the CUDA context and modules between requests. Timeout, illegal access or interrupted requests poison/quarantine the worker; subsequent requests are not silently accepted. `test_worker_interrupted_request_is_not_silently_reexecuted`, worker serialization test. | Real destructive probe job 769644 quarantined the allocation; job 769661 accepted independent validation in a new worker on the same physical GPU. See `VALIDATION.md`. |
| AC-16: interrupted ask/tell | Tree state, durable result integration and random state are persisted with transactional exactly-once tell handling. `test_crash_replay_is_atomic_once_and_restores_pending`, graph checkpoint replay test. | Test whole-process interruption of a real campaign around result/tell boundaries; local crash-injection tests establish ledger behavior only. |
| AC-17: measurement deduplication | Trial IDs and evidence IDs are distinct; duplicate evidence is not a second valid reward, while intentional remeasurement uses a new evaluation identity. `test_measurement_dedup_and_intentional_remeasurement`. | Inspect real attempt/evaluation relationships when deliberately replaying a run. |
| AC-18: tools and environment drift | Typed unavailable-tool outcomes, persisted candidate sources and resume identity checks preserve qualified state. Standalone ledger reservation tests cover optional accounting mechanics; active campaigns do not impose global token/GPU/time limits following the later user instruction. | Record actual consumption and reject an incompatible target at resume. Global resource-exhaustion enforcement from the original specification is superseded by the user's explicit request to leave agents unconstrained. Technical tool timeouts still qualify failures. |
| AC-19: unseen shapes and final failure | `LaunchRule` is frozen arithmetic metadata; final cases are constructed lazily after search and cannot trigger regeneration. Holdout exposure is persisted before final evaluation. `test_launch_rule_applies_tails_without_generator_or_optin_memory`, `test_lazy_holdout_created_after_search_preserves_contract_hash`. | Run unseen target sizes and an intentionally failing final candidate; verify an explicit failed/uncertified result with no new generator call. |
| AC-20: reconstruction | Candidate source/launch/parameters, dossier source cache and hashes, contract, environment, model calls, attempts, evaluations and final manifests have stable identifiers. Source/cache mutation is rejected. | Reconstruct an exported candidate using its recorded toolchain and revalidate it on the authenticated target. Paths alone are not a portable artifact bundle. |

## Current test boundaries

Run the repository test suite in the configured Python environment:

```sh
python -m pytest tests -q
```

The research/backend tests also run with the Python standard library:

```sh
python -m unittest tests.test_codex_research -v
```

The suite contains simulated runners and provider events deliberately. Those
tests establish control flow, numerical-decision policy, durable state handling
and failure behavior without spending GPU or model budgets. Actual CUDA
correctness, sanitizer behavior, binary-cache reuse, counter permissions and
performance remain claims that require target receipts.

The bundled desktop Codex CLI and a separately installed CLI can have different
model availability. The backend prefers `CODEX_CLI_PATH`, then an existing
ChatGPT bundled CLI on macOS, then `codex` on `PATH`. A provider rejection of the
requested model is recorded as an error; there is no automatic substitute.
During setup, the separately installed CLI `0.143.0` rejected `gpt-6-luna`
for the existing ChatGPT login. Selecting the already installed desktop bundle
`0.163.0-alpha.10.2` resolved that compatibility issue: a bounded real Luna Max
call completed successfully with usage tokens. This health check establishes
model access; it does not establish successful research or kernel generation.

Source fetches verify TLS certificates, keep bounded complete response bodies
and record SHA-256 identities. Source bodies are reference data. A fetched
copyright header or an algorithmic citation does not authorize copying unknown
licensed code; the generator receives a synthesis-only restriction until the
upstream license is independently established. Search queries must come from
observed Codex web-search events. A model's list of URLs is insufficient.

## Product evaluation beyond functional acceptance

The representative workload suite includes reduction, scan, stencil,
convolution, square float32 GEMM and normalization under a single-kernel ABI.
That implemented workload catalogue is separate from executing every workload.
Arbitrary multi-kernel pipelines, rectangular GEMM and alternative ABIs require
new explicit contracts.

One naive-kernel/agent/vanilla comparison is a demonstration, not evidence of a
general optimizer improvement. Specification section 16 additionally requires
multiple campaigns, equal enforced comparison budgets, per-case and aggregate
dispersion, correctness/configuration rates, time to useful improvement, model
tokens, GPU/compilation/research cost and visible regressions. Required ablations
include the upstream path versus TreeQuest, autotuning, research corpus and PMPP.
Those studies must be reported as pending until their individual runs finish.
The later user instruction removes imposed global model-token, GPU-consumption
and campaign-time caps. Consequently an uncapped comparison must publish actual
resource use and cannot claim equal enforced budgets. No automatic search-count
cap, source-count ceiling, shared research-phase deadline, or fixed search/query/URL
count is configured by default. Technical request
timeouts and a complete-response download-size safeguard remain separate from
campaign resource limits.

The research ablation uses a separate, explicitly labeled generator interface.
It retains the complete proposal, launch and tuning schema while requiring an
empty reference list; it never fabricates a successful research dossier. The
PMPP ablation preserves live research and actual code/CUDA read receipts, but
omits both permanent guidance and targeted PMPP cards/sources. Its dossier and
proposal carry `experimental_mode: ablation_no_pmpp`. Normal generation rejects
either ablation marker and still requires all normal evidence. These interfaces
make the comparisons executable; their existence is not a completed ablation.

Local profiling can supply an algorithmic hypothesis when a CUDA-capable local
worker exists. Its GPU, toolchain and counters must remain labeled with that
local environment. ROMEO timing and promotion decisions use ROMEO measurements;
local timings or published vendor timings never stand in for them. If neither
target nor local profiling works, retain the explicit unavailable outcome and
continue only where the evaluation contract permits it.
