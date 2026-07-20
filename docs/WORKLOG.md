# Shared worklog

Append newest entries at the top, directly below this introduction. Keep entries concise
but sufficient for another task to continue without reading the originating chat.

## 2026-07-20 — Visualize all QC-flagged waveforms

**Task:** show in `csv_data_processing.ipynb` every signal marked by at least one
technical QC flag.

**Changes:** added reusable QC accounting and Plotly functions to
`psd_ml/pipeline.py`; added a notebook section defining “outlier” as the union of QC
flags; plotted each event once after per-event baseline subtraction and sign inversion,
but before time alignment or amplitude normalization. Hover shows sample/source row,
all active flags, amplitude, SNR, baseline RMS, and tail ratio. Standalone HTML files are
generated per non-empty source×channel group under `gamma_n_data/samples/`.

**Findings:** the fixed 10,000-event sample contains 192 unique flagged events: 2 from
the Co acquisition and 190 from the Cf acquisition. Seven source×channel groups contain
flagged events; the other three are reported explicitly as empty. Individual flag counts
overlap and therefore are not summed as event counts.

**Verification:** executed the complete notebook without errors or warnings; verified 7
new Plotly MIME outputs and 7 HTML files, exactly 192 traces, 192 unique `sample_row`
identifiers, and exact equality between the flag union and `~quality_ok`. The notebook
now contains 22 interactive figures in total.

**Files modified:** `psd_ml/__init__.py`, `psd_ml/pipeline.py`,
`csv_data_processing.ipynb`, `docs/PROJECT_CONTEXT.md`, and `docs/WORKLOG.md`.

**Next step:** inspect the QC plots before deciding whether any flag definition should be
revised; flags remain diagnostic annotations and do not authorize deleting events.

## 2026-07-17 — Audit and preprocess raw waveform sample

**Task:** determine necessary waveform preprocessing from measured data, apply it
reproducibly, and describe processed shapes without assigning particle identity.

**Changes:** added the project mini-library `psd_ml/pipeline.py`; refactored
`csv_data_processing.ipynb` so its code cells contain only parameters and stage-level
library calls; audited a fixed stratified sample of 10,000 pulses; generated aligned
amplitude, aligned peak-normalized, per-event feature/QC, and group summary CSV files;
and added five interactive channel-level processed-shape summaries. No event is silently
deleted.

**Findings:** no sampled pulse reached an ADC rail; median baseline RMS is 3.136 ADC;
the 95th percentile absolute baseline slope is 0.584 ADC/sample; baseline estimates from
8–16 initial samples differ from the 12-sample median by at most 2 ADC at the reported
95th percentiles; median SNR is 403.6; raw CFD-50 spans 16.62–25.10 samples at the 1–99%
quantiles; and relative tail residual has 95/99% quantiles 1.31%/3.10%. Per-group
baseline-versus-source-row correlations are reported separately to avoid confounding
time drift with channel offsets.

**Processing:** median baseline from samples 0–11, sign inversion, sub-sample CFD-50
alignment to sample 20, and separate amplitude-retaining/peak-normalized branches. No
smoothing or linear detrending. QC flags cover clipping, SNR, baseline noise, alignment,
tail recovery, and conservative multi-peak structure.

**Verification:** executed the full notebook without errors or warnings; verified 15
native Plotly outputs, 10 raw and 5 processed HTML files, 10,000 rows in each processed
waveform CSV, 10,001 rows including headers in provenance/features, and 11 rows including
the header in the group audit. The 99th percentile post-alignment CFD error is 0.2355
sample and normalized valid pulses peak at one.

**Files created/modified:** `psd_ml/__init__.py`, `psd_ml/pipeline.py`,
`csv_data_processing.ipynb`, `docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and generated
ignored artifacts under `gamma_n_data/samples/`.

**Limitations/next step:** the audit is balanced rather than representative and uses only
one run per source condition. Obtain independent repeated/background runs, confirm timing
units and acquisition-filter semantics, then test preprocessing stability by complete
run before interpreting separation features.

## 2026-07-17 — Initialize and publish Git repository

**Task:** Activate Git for `PSD_ML` and connect it to the user's GitHub account as a
public repository.

**Changes:** initialized local branch `main`; installed and authenticated GitHub CLI as
`iv-gonch`; added `.gitignore` excluding ROOT files, 2.1 GB derived CSV data, generated
samples/plots, `.venv`, and local caches; recorded the public repository and publication
boundary in shared memory.

**Verification:** `git check-ignore` confirmed representative ROOT, CSV, sample, and
environment files are excluded. The remote repository did not exist before creation.

**Files created/modified:** `.git/`, `.gitignore`, `docs/DECISIONS.md`,
`docs/PROJECT_CONTEXT.md`, and `docs/WORKLOG.md`.

**Remote:** `https://github.com/iv-gonch/PSD_ML`, branch `main`.

## 2026-07-17 — Record detector/channel identities

**Task:** Preserve the experimental detector compositions and use exact PMT +
scintillator labels from now on.

**Changes:** added the canonical channel map and crystal sizes to `docs/DECISIONS.md`;
added `DETECTOR_LABELS` plus a mapping table to `csv_data_processing.ipynb`; updated all
embedded Plotly and standalone HTML titles to include `CHn: <PMT> + <scintillator>`.

**Verification:** reran the complete notebook and checked all ten embedded Plotly titles
and all ten HTML files. Each channel has exactly two correctly labelled plots, one Co and
one Cf.

**Files modified:** `csv_data_processing.ipynb`, ten generated HTML plots,
`docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and `docs/WORKLOG.md`.

**Next step:** use this mapping as a grouping variable in every channel-specific audit
and model comparison.

## 2026-07-17 — Replace Plotly iframes with native notebook outputs

**Task:** The previous `/files/` iframe fix still produced black/empty frames in the
user's JupyterLab.

**Resolution:** removed iframe rendering entirely. `csv_data_processing.ipynb` now sets
Plotly's `plotly_mimetype` renderer and emits each figure directly as a native
`application/vnd.plotly.v1+json` notebook output. Standalone HTML files remain available
as a fallback and reproducible artifact.

**Verification:** reran the full notebook; it contains exactly ten Plotly MIME outputs,
each with one `scattergl` trace and the expected source/channel title, and contains zero
iframe outputs. The executed notebook is approximately 29.7 MiB because plot data are
now embedded directly.

**Files modified:** `csv_data_processing.ipynb`, `docs/PROJECT_CONTEXT.md`,
`docs/WORKLOG.md`.

## 2026-07-17 — Fix blank Plotly iframes inside JupyterLab

**Task:** Interactive HTML plots opened directly in a browser but appeared as empty
frames inside `csv_data_processing.ipynb`.

**Cause and fix:** iframe `src` values were workspace-relative paths, which JupyterLab
resolved below its `/lab/tree/` route. Updated the notebook to serve each generated HTML
through Jupyter's contents endpoint: `/files/gamma_n_data/samples/...`.

**Verification:** reran the complete notebook and checked all ten stored iframe outputs;
each now uses an absolute `/files/` URL. The external HTML files and sampling results are
unchanged.

**Files modified:** `csv_data_processing.ipynb`, `docs/WORKLOG.md`.

## 2026-07-17 — Stratify waveform sample by source and channel

**Task:** Draw separate plots for every source and channel, using 1,000 pulses per plot,
and retain HTML rather than PNG outputs.

**Changes:**

- Updated `csv_data_processing.ipynb` to sample exactly 1,000 events without replacement
  from each of the ten source CSVs (`60Co`/`252Cf` × channels 0/2/3/4/5).
- The generated waveform sample now contains 10,000 rows × 144 samples (8.223 MiB), with
  a matching provenance table.
- The notebook generates ten independent Plotly WebGL HTML files. PNG generation was
  removed. Saved HTMLs are embedded back into the notebook with lightweight iframes so
  the `.ipynb` does not duplicate millions of plot points.

**Verification:** ran all notebook cells using the local `psd-ml` kernel; verified 1,000
provenance rows for every source-channel group, 10,000 waveform rows total, ten HTML
files, and Plotly/`scrollZoom` configuration in every HTML.

**Files created/modified:** `csv_data_processing.ipynb`, `gamma_n_data/samples/*`,
`docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and `docs/WORKLOG.md`.

**Next step:** perform baseline and alignment diagnostics separately for each of these
ten groups before comparing pulse shape distributions.

## 2026-07-17 — Balance Co/Cf sample and add interactive plots

**Task:** Use equal numbers of Co and Cf pulses, plot the sources separately, and make
the plots zoomable where possible.

**Changes:**

- Updated `csv_data_processing.ipynb` to sample exactly 913 `60Co` and 913 `252Cf`
  events, uniformly without replacement within each run. The resulting waveform CSV is
  1.501 MiB.
- Replaced the combined visualization stage with separate Co and Cf static PNGs and
  offline Plotly WebGL HTML plots. Scroll zoom, box zoom, pan, reset, and local rendering
  without internet are supported.
- Installed Plotly 6.9.0 in `.venv` and documented the exact reinstall command.
- Found that the pre-existing user kernel named `.venv` pointed to another project.
  Added project-local kernel `psd-ml`, wired it into `start_jupyter.sh`, and updated the
  notebook metadata.

**Verification:** executed all notebook cells using the local `psd-ml` kernel; assertions
confirmed equal source counts and 1–2 MiB output size; verified both HTMLs contain Plotly
and `scrollZoom`; visually inspected both PNGs.

**Files created/modified:** `csv_data_processing.ipynb`, `.jupyter/kernels/psd-ml/kernel.json`,
`start_jupyter.sh`, `gamma_n_data/samples/*`, `gamma_n_data/JUPYTER.md`,
`docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and `docs/WORKLOG.md`.

**Next step:** add baseline subtraction and pulse alignment as explicit notebook stages,
keeping raw and transformed representations side by side.

## 2026-07-17 — Start reproducible CSV notebook pipeline

**Task:** Create `csv_data_processing.ipynb`; uniformly select a random 1–2 MiB test
sample from all waveform CSVs and draw all selected pulses on one graph.

**Changes:**

- Added the ordered notebook pipeline with centralized parameters, fixed seed 20260717,
  automatic source discovery/inventory, global random sampling without replacement,
  validation, artifact saving, and plotting.
- Generated `gamma_n_data/samples/test_waveforms_seed_20260717.csv`: 1,827 pulses × 144
  ADC samples, exactly 1.500 MiB, headerless.
- Generated an aligned provenance CSV containing run, channel, source filename, and
  source row for every sampled pulse, plus a combined raw-waveform PNG.
- Updated Jupyter usage notes and recorded the notebook-first reproducibility decision.

**Verification:** executed the notebook end-to-end with kernel `PSD_ML (.venv)`; all seven
code cells completed, source inventory totalled 2,486,390 pulses, artifact size assertion
passed, output CSV has 1,827 rows and 144 columns, and the PNG was visually inspected.

**Files created/modified:** `csv_data_processing.ipynb`, `gamma_n_data/samples/*`,
`gamma_n_data/JUPYTER.md`, `docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and
`docs/WORKLOG.md`.

**Next step:** append preprocessing/audit stages to this notebook, beginning with baseline
estimation, polarity confirmation, alignment diagnostics, and quality flags before any
class-separation model.

## 2026-07-17 — Export all ROOT waveforms to CSV

**Task:** Write pulse shapes from every `Data_CH*.root` file to CSV with one pulse per
line.

**Changes:**

- Added `gamma_n_data/export_waveforms_csv.py`, a streaming PyROOT exporter that reads
  only the `Samples` branch and writes atomically through `.part` files.
- Created ten headerless waveform files under `gamma_n_data/CSV/`, mirroring both runs
  and all active channels.
- Added `gamma_n_data/CSV/README.md` documenting positional columns and row mapping.
- Recorded the waveform CSV interface in `docs/DECISIONS.md` and current paths here.

**Format:** every CSV row is one ROOT event, every row has 144 ADC samples, columns map to
sample positions 0–143, and row order matches the source TTree entry order. Event metadata
are intentionally excluded.

**Verification:** checked all ten outputs: row counts match their ROOT trees (2,486,390
pulses total), every first row has 144 columns, no `.part` files remain, and the export
occupies approximately 2.1 GB.

**Files created/modified:** `gamma_n_data/export_waveforms_csv.py`,
`gamma_n_data/CSV/**`, `docs/DECISIONS.md`, `docs/PROJECT_CONTEXT.md`, and
`docs/WORKLOG.md`.

**Next step:** retain ROOT files as the authoritative source; export event metadata to a
separate aligned table only if the downstream analysis requires it.

## 2026-07-17 — Study current PSD_ML project state

**Task:** Inspect the whole project, shared plan, data layout, ROOT utilities, runtime,
and readiness for neutron/gamma separation work.

**Findings:**

- The project is an early research workspace, not yet a training/evaluation pipeline.
  It contains a sound staged analysis plan, ROOT inspection/plotting helpers, and a
  notebook for sampling waveforms.
- Data consist of one 30-minute `60Co` gamma-control run and one 60-minute `252Cf`
  neutron/gamma-mixture run on channels 0, 2, 3, 4, and 5. Full 144-sample waveforms and
  event metadata are available.
- CoMPASS classical PSD is `(Energy - EnergyShort) / Energy`; the configured gates are
  140 ns long, 40 ns short (60 ns for channels 4/5), and 10 ns pre-gate.
- Cf metadata reports nonzero PSD rejection and channel-specific cuts/thresholds that are
  not present in the Co run. The semantics of `UNFILTERED` and possible acquisition-time
  selection must be clarified before treating the runs as comparable distributions.
- Only one run per condition exists, with no background run or independent neutron truth,
  so run-level generalization and neutron efficiency cannot yet be measured honestly.
- `.venv` provides Python 3.14, ROOT 6.36, NumPy, Matplotlib, and JupyterLab; pandas and
  scikit-learn are not installed. The plotting utility successfully rendered a waveform.
- `root_inspect.py` can report duplicate `Data` trees when a ROOT file contains multiple
  key cycles with the same name; this is a utility issue, not duplicate physical runs.

**Files modified:** `docs/PROJECT_CONTEXT.md`, `docs/WORKLOG.md`.

**Verification:** compiled all three Python ROOT utilities; inspected a Cf event tree and
sample entries; rendered a waveform PNG; compared run metadata and channel overrides.

**Recommended next step:** resolve acquisition/filter semantics and detector metadata,
then implement a per-channel, per-run audit plus the classical PSD baseline before any
ML training.

## 2026-07-17 — Establish cross-chat shared memory

**Task:** Make work performed in sibling PSD_ML chats visible across the project.

**Changes:**

- Added root `AGENTS.md` with mandatory read/update and handoff rules.
- Added `docs/PROJECT_CONTEXT.md` as the concise current-state index.
- Added `docs/DECISIONS.md` for durable scientific and technical decisions.
- Added this shared chronological worklog.
- Linked the existing signal-processing plan and `gamma_n_data` resources from the project
  context.

**Verification:** confirmed that project files and `gamma_n_data` utilities are visible
from the common workspace.

**Limitation:** this mechanism shares durable findings and artifacts, not complete sibling
chat transcripts automatically. Relevant transcript content must be summarized or saved
under `docs/`.

**Next step:** every active or future project task should read these files and add its own
handoff entry after material work.

## 2026-07-17 — Import signal-processing plan

**Task:** Preserve the earlier planning chat in the project workspace.

**Changes:** added `docs/signal_processing_plan.md` containing the research goal, staged
audit and preprocessing plan, PSD and ML baselines, domain-shift checks, pseudo-labeling
strategy, validation rules, and required input metadata.

**Next step:** audit the uploaded experimental files and construct the run-level metadata
table before modeling.
