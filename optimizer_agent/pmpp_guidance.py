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


# These short cards are original project material, not quotations or a purported
# index of the copyrighted book. The publisher's fourth-edition companion can be
# consulted by ResearchService when it is actually accessible.
PMPP_COMPANION_URL = "https://shop.elsevier.com/books/book-companion/9780323912310"
PMPP_CARD_PROVENANCE = {
    "edition": "Project-authored PMPP-inspired guidance (fourth-edition companion referenced)",
    "provenance": "Original repository guidance; no full book text is installed or claimed",
    "source_file": __file__,
    "companion_url": PMPP_COMPANION_URL,
}
PMPP_TOPICS = {
    "coalescing": "Map neighboring lanes to adjacent addresses; check the actual stride, alignment and tails.",
    "tiling": "Load reusable tiles cooperatively; estimate saved traffic, synchronization and shared-memory cost.",
    "register_reuse": "Compute multiple outputs per thread while checking registers, spills and available parallelism.",
    "occupancy": "Compare useful blocks to SM count; registers and shared memory can reduce resident warps.",
    "divergence": "Inspect lane-dependent branches and tail predicates without removing required edge cases.",
    "synchronization": "Every participating thread must reach block barriers; initialization must cover padded tiles.",
    "reduction": "Combine values cooperatively and verify association, precision and synchronization against the oracle.",
    "scan": "Track dependencies, identity elements and ordering; device-wide algorithms may require multiple launches.",
    "stencil": "Include halo traffic and boundary conditions when comparing shared tiles to direct global access.",
    "convolution": "Explore spatial and channel reuse, preserving layout, precision and all valid output positions.",
}


def consult_pmpp(topics: list[str]) -> list[dict]:
    """Read only installed cards and report their exact provenance and hash."""
    import hashlib
    from pathlib import Path

    unknown = set(topics) - PMPP_TOPICS.keys()
    if unknown:
        raise ValueError(f"PMPP topics unavailable: {sorted(unknown)}")
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return [dict(PMPP_CARD_PROVENANCE, topic=topic, text=PMPP_TOPICS[topic],
                 content_sha256=digest, access="installed original guide") for topic in topics]
