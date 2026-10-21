# Portable study exports

Export a section 16 study without submitting work:

    .venv/bin/python scripts/export_study.py export \
      --study results/section16 \
      --output /absolute/output/study-export

The output contains study-evidence.zip and export-manifest.json. The manifest
lists candidate IDs and selections. Reconstruct one exact frozen candidate and
its archived real execution requests in a fresh directory:

    .venv/bin/python scripts/export_study.py reconstruct \
      --bundle /absolute/output/study-export/study-evidence.zip \
      --candidate CANDIDATE_ID \
      --output /absolute/output/reconstructed-candidate

Reconstruction verifies file checksums, source/launch/parameter identity,
compiler option hashes, declared and quoted dependencies, and input files.
It rewrites paths into the fresh directory. Relative quoted includes retain
their source-tree layout. Compiler include paths have an explicit relocation
record; original measured options and their hashes remain in the manifest.
Request files include an ID and results directory and can be passed to a
compatible harness with --request. Reconstruction does not execute the kernel
or qualify the new environment.

The ZIP retains the exact git archive of HEAD, observed dirty state, a binary
working-tree patch, supported untracked repository source files, and current
bridge source bytes. Current bridge hashes are checked against original-agent
execution receipts, including `gain_observer.py`. A current mismatch is retained
in the audit. The four recorded adapter files may instead be resolved together
from a historical `controller-sources/<commit>/source.tar.gz`: its complete file
map, observed source manifest and TAR contents must match that pinned local Git
tree. Missing, altered or ambiguous file identities keep the export partial.
All compatible archives are listed; the exporter does not choose an execution
revision from identical adapter bytes. This resolves the recorded source files,
and leaves an unknown task-to-controller revision explicitly unknown.
The original-agent source snapshot is separately authenticated against Git
commit 1e9464da54dfc651337a97c3643bfeefec712bc1 and archived byte for byte.

Experiments, LangGraph and suite SQLite journals use sqlite.Connection.backup.
Committed WAL pages are included through SQLite; raw WAL and SHM files are
excluded. Each backup passes integrity_check. The ZIP also contains readable
ledger rows, tree parent/child links, scores, pending/finished trials, RNG state,
eligibility decisions, reward claims and integration flags. Python tree pickles
are read only from the explicitly chosen trusted local study, never from a
received ZIP. Cross-database atomicity during active execution is not claimed.

An export is partial when execution is unfinished, a controller/worker lock is
active, a cached job has no terminal receipt, search trials are pending, inputs
or artifacts are missing, files change during export, or recorded identities
do not match. Complete denotes the exported study's recorded execution state.
Exporting creates no scientific certification; the original final validation,
sanitizer and paired confirmation receipts remain authoritative.

The exporter queries no scheduler and records job state from local journals and
receipts. Active jobs are untouched. Partial SQLite backups can contain sealed
final seeds: exports are protected with directory mode 0700 and archive mode
0600 and must remain private while generation continues.

Downloaded research corpus bodies under research/sources, supplied text under
research/passages and the text-bearing *-supplied-content.json synthesis inputs
are excluded. Exact dossier/receipt metadata bytes remain unchanged.
reference-rehydration.json records omitted file hashes and each source URL,
version, body/passage hash and locator without redistributing its text. Resuming
generation requires authenticating and rehydrating those sources and receipts
under a controlled relocation workflow before the research guard can pass.
Export and kernel reconstruction retrieve no external source, claim no rereading
and never approve research_ready. Kernel reconstruction needs no external corpus.
Kernel source,
baseline source, real input bytes, raw measurements, configuration, contracts,
environment and provenance evidence remain included.

## Offline search-tree visualization

Render a trusted local completed campaign's cold experiments database without
submitting or repeating any computation:

    .venv/bin/python scripts/render_search_tree.py \
      --database results/RUN_ID/experiments.sqlite \
      --output /absolute/new-directory/search-tree \
      --trust-local-pickle

The output contains a standalone `search-tree.svg` and a readable
`search-tree.json`: exact recorded scores, parent/child links, candidate/source
hashes, pending trials, eligibility and closure history. The greedy ablation is
shown as candidate lineage with its individual observations, without inventing
a single TreeQuest score. Search scores and the search incumbent are explicitly
distinct from final gain and certification. An optional `--manifest FILE` records
a same-run external status declaration without qualifying it.

The selected database must be trusted: its persisted search snapshot is a Python
pickle. The renderer rejects private suite journals, unknown schemas, nonempty
WAL/journal files and sources changing during capture. It reads the cold source
with SQLite's immutable read-only mode, backs it up into a temporary directory,
checks integrity and verifies source fingerprints before and after. It never
checkpoints or writes to the original. For an active campaign, use an existing
coherent exported experiments database; do not stop its process for a picture.
The output must be a fresh directory outside the generation/database directory.
Its JSON includes the readable ledger and search RNG and remains protected with
directory mode 0700 and file mode 0600 while campaigns continue.
