# Original guidance inspired by PMPP's optimization topics; not book excerpts.
PMPP_GUIDANCE = """
Use the performance methodology of Programming Massively Parallel Processors
(PMPP, Hwu/Kirk/El Hajj): reason about the computation and data movement before
changing code. Choose techniques for the measured bottleneck, not for their names.

Consider these concrete experiments where applicable:
- Map neighboring lanes to neighboring addresses for coalesced global accesses.
- Cooperatively load reusable input/weight tiles into shared memory; estimate
  global loads saved, halo overhead, synchronization cost and bytes per block.
- Reuse values in registers across multiple outputs (register tiling / thread
  coarsening); weigh this against registers, spills and fewer independent threads.
- Expose enough useful blocks/warps to hide latency. Compare useful block count
  to SM count: launching extra blocks that do no work does not add parallelism.
- Reduce divergence and shared-memory bank conflicts; consider boundary handling
  separately from the interior without excluding valid edge cases.
- For convolutions, consider spatial and output-channel tiles with input/weight
  reuse, not just scalar index rewrites. For reductions/GEMV, consider cooperative
  reductions and reuse of vector data. Keep layouts, precision and ABI intact.

Use actual retrieved device values: warp size, SM count, shared-memory limits
and L2 capacity. Estimate resource use; do not claim exact occupancy or register
counts without compiler/profiler evidence. Shared-memory barriers must be reached
by all participating threads, including partial-tile threads. Do not introduce
host preprocessing, constant-memory uploads, extra launches or allocations that
the current single-kernel harness cannot provide.

Explain the experimental hypothesis, expected benefit, and main tradeoff briefly.
When history shows a technique failing, examine why and try a materially different
applicable technique. A repeat is allowed when motivated by a measurement question.
This is experimental guidance, not a requirement to use every technique or to
claim an improvement before measurement.
"""
