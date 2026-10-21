# Actual upstream reference agent

`optimizer_agent.upstream_original.run_upstream_campaign(config, runner, directory,
resume=False, finalize=False)` runs the original LangGraph from commit
`1e9464da54dfc651337a97c3643bfeefec712bc1`. The caller supplies an already ready,
owned worker. This function submits no Slurm job and never materializes final data.
The section 16 orchestrator calls it as the `true_upstream` mode.

The source snapshot comes from the pinned Git objects, with SHA-256 checks for
every original Python source. The original bytes execute in a private Python
import namespace. The original prompts, PMPP text, graph edges, tool routing,
JSON validation/retries, repair of the latest failure, early stopping and best
valid geometric-mean selection run unchanged. Original optional NVIDIA research
and Nsight stay disabled, their upstream defaults. This reference is separate
from the fork's deterministic greedy-parent ablation.

The documented substitutions are provider and execution transports. The original
ChatOpenAI interface is bridged to local Codex `gpt-6-luna` with effort `max`.
The provider effort vocabulary is extended to `max`; all other original
configuration checks are retained.
The bridge serializes the entire original conversation and actual original tool
schemas; tool requests return to the original ToolNode. The original JSON
schemas still validate all launch plans and proposals. Codex has no shell access
through this bridge. The original harness request is translated to the common
ROMEO runner request, retaining its source, launch, case, warmup and repetitions.

The original public Workload interface receives the common suite signature,
search inputs, naive reference and initial source. A separate immutable oracle
admits the reference and checks its execution results, preventing a bad reference
launch from approving the same wrong candidate output. The original agent still
chooses and ranks its attempts. The selected source and its exact per-shape
launches are exported unchanged. `fixed_shape_table` rejects unseen shapes;
common final evaluation uses fresh data on the declared fixed shape domain.

`max_trials` maps to upstream `max_iterations`: iteration zero is the supplied
initial implementation, followed by at most `max_trials` proposals. Repairs
consume these original iterations. `max_repairs_per_trial` maps to original
`max_retries`; upstream's guard of `8 * (max_retries + 1)` model/tool calls for
each proposal remains part of that algorithm. No global resource ceiling is
added. The configured per-request Codex timeout is a transport timeout.

A SQLite checkpointer preserves the original graph state. Completed model and
harness requests have durable receipts reused when the same interrupted graph
task is replayed. Distinct original conversations retain distinct message IDs.
Source, configuration and target identities are checked on resume. Interruptions
without a completed receipt remain distinguishable from completed calls; exact
remote worker recovery belongs to the owning suite orchestrator.

Each original-model request records its prompt/schema/configuration intent before
launching Codex, with a separate CLI receipt directory for that attempt. If the
controller stops after Codex completes but before acknowledging the response,
resume authenticates that receipt and restores the response without another
invocation. A missing, partial or ambiguous receipt leaves the intent uncertain
and refuses further prompts; it is not silently retried.

`candidate.json` contains source and artifact hashes, the frozen launch table,
the original selected iteration and whether it is the supplied initial source
or an original generated proposal. `manifest.json` records the upstream and
adapter source identities, original summary, independent search validation,
actual model token receipts and raw original harness receipt references. Its
`generation_complete` does not certify GPU performance. The grouped evaluator
qualifies all frozen modes together on fresh data and a common new GPU.

An external method failure remains `method_status=failed`, with its recorded
error, even when a valid retained best can be exported. `candidate_fallback`
identifies that artifact's original iteration and whether it came from the
supplied initial source. The common final evaluator may measure this artifact;
the reports retain the method failure alongside that measurement. Finishing an
attempt, completing its Slurm job and qualifying its artifact are separate facts.
