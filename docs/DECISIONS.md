# Project decisions

This file records durable decisions shared by all project tasks. New entries should state
the date, decision, rationale, and consequences. If a decision is superseded, retain it
and point to the replacing entry.

## 2026-07-17 — Use files as cross-chat project memory

**Decision:** `docs/PROJECT_CONTEXT.md`, `docs/WORKLOG.md`, and this file are the canonical
communication layer between Codex chats working in `PSD_ML`.

**Rationale:** sibling chats do not automatically share their full transcripts, while all
tasks in the project can read and update the same workspace.

**Consequence:** every material task must read shared memory at startup and publish a
handoff before finishing, as required by the root `AGENTS.md`.

## 2026-07-20 — Test Cf shape branches inside channel-specific Energy strata

**Decision:** join a sampled CSV waveform to ROOT `Energy` and `EnergyShort` by its
preserved source-entry number, and require exact equality of all 144 CSV/ROOT waveform
samples before using that join. Treat `Energy` as an uncalibrated instrument value until
an external calibration is available.

For the initial branch search, use a deterministic Cf-only sample of 5,000 events per
channel and analyze channels separately. Divide each channel into 12 equal-count Energy
intervals; calculate a peak-normalized late-area score (positive area in samples 40:100
divided by positive area in 15:100); and compare one- and two-Gaussian fits independently
inside each interval. Retain `low_snr` events, but exclude structural failures. Require
ΔBIC ≥ 10, separation ≥ 2, at least 8 assigned events per component, bootstrap support
≥ 0.70, at least three supported intervals including three consecutive intervals, and
median branch-shape cosine ≥ 0.80. Repeat the verdict for 8, 10, and 12 intervals and
require agreement in at least two of three binnings for binning-robust evidence.

**Rationale:** energy stratification prevents a simple amplitude/energy trend from being
misread as particle-dependent shape, while fixed mixture and repeatability criteria make
the branch claim auditable. Exact waveform verification turns the positional metadata
join into a checked interface. Low-SNR retention keeps the scientifically important
low-energy region in scope.

**Consequence:** call the fitted groups lower- and higher-tail shape components, not
gamma and neutron classes. Current stability is within one Cf acquisition across
neighboring energy intervals, bootstrap resamples, and bin counts; it is not independent
run reproducibility. Generated filenames include the sample size and seed so analyses
with different sampling scopes cannot silently overwrite one another.

## 2026-07-20 — Do not treat low-SNR events as permanently rejected

**Decision:** `low_snr` remains a diagnostic QC flag. Although the current convenience
mask `quality_ok` excludes events with `SNR < 20` from aggregate clean-shape summaries,
those events must remain stored, inspectable, and available as a separate analysis
stratum. Future PSD/model evaluation must be reported both with and without this cut and
must include dedicated low-energy metrics.

**Rationale:** the cut currently marks 144/10,000 sampled events and preferentially
selects small-amplitude pulses. Small amplitude is an energy proxy pending calibration,
so unconditional rejection can remove low-energy gamma and neutron events in precisely
the region motivating improvement over classical PSD.

**Consequence:** quantify QC acceptance versus amplitude and calibrated energy, validate
the amplitude–energy relationship using ROOT metadata, document threshold sensitivity,
and include the requirements in `docs/REPORT_NOTES.md`. Do not describe `quality_ok` as
the final scientific acceptance selection.

## 2026-07-17 — Keep implementation in a project Python library

**Decision:** new processing logic is implemented as functions in the project-local
`psd_ml/` package. Notebooks expose parameters, call those functions in conceptual order,
and display their results; they do not contain function implementations or processing
loops. This supersedes the implementation-in-notebook part of the earlier “Use one
notebook” decision while preserving `csv_data_processing.ipynb` as the reproducible
orchestration and research record.

**Rationale:** reusable Python functions can be tested and called outside Jupyter, while
short stage-level notebook calls keep the analysis legible without requiring readers to
inspect implementation details continuously.

**Consequence:** add functions only when an analysis stage needs them, give each public
function one clear scientific role, and keep all parameters that affect an experiment
visible in the notebook.

## 2026-07-17 — Initial waveform preprocessing based on technical diagnostics

**Decision:** for the current 144-sample waveforms, compute a per-event baseline as the
median of samples 0–11; subtract it and invert the negative pulse polarity; align each
valid event by linearly interpolating its first CFD-50 leading-edge crossing to sample
20. Preserve both an amplitude-retaining aligned branch and a peak-normalized branch for
shape-only comparisons. Do not smooth or fit/remove a linear baseline trend at this
stage.

Flag, but do not delete, ADC clipping, low SNR, anomalous baseline noise, invalid/early
alignment, tail non-recovery, and conservative multi-peak topology. Thresholds based on
noise are fitted per channel while pooling both source runs, so source identity does not
choose data-quality cuts.

**Rationale:** on the fixed 10,000-event source×channel-stratified audit sample, baseline
window estimates are stable through 12 samples; the 95th percentile absolute baseline
slope is only 0.584 ADC/sample; no event reaches either ADC rail; the raw CFD-50 1–99%
range is 16.62–25.10 samples; and 99% of relative tail residuals are below 3.11%. These
measurements justify baseline correction and time alignment but not smoothing, linear
detrending, or blanket tail correction.

**Consequence:** this is a technical conditioning stage, not a neutron/gamma selector.
The source labels remain acquisition-condition labels, suspicious events remain present
with QC columns, and parameters must be re-audited on independent runs before being
treated as stable production defaults.

## 2026-07-17 — Validate by complete experimental run

**Decision:** model evaluation and transformation fitting must split data by complete
series, run, or experimental day rather than randomly splitting individual pulses.

**Rationale:** pulses from one acquisition share technical conditions and can cause severe
data leakage.

**Consequence:** run identifiers and acquisition metadata are required fields in the audit
table and all evaluation datasets.

## 2026-07-17 — Use candidate terminology before independent labels

**Decision:** without independent neutron truth labels, model outputs are called
neutron-like scores or candidates rather than neutron probabilities.

**Rationale:** `G` versus `M` origin labels identify experimental mixtures, not the true
particle identity of each pulse.

## 2026-07-17 — Preserve ROOT event/sample order in waveform CSV exports

**Decision:** waveform CSV exports are headerless, contain exactly one pulse per row and
one ADC sample per column, and preserve both TTree entry order and sample order. Event
metadata remain outside these waveform-only files.

**Rationale:** this makes every line satisfy the requested one-row/one-pulse interface
without inserting a non-pulse header row, while retaining a deterministic mapping back
to the source ROOT entry.

**Consequence:** columns are interpreted positionally as samples 0 through 143. Metadata
must be read from the source ROOT tree or exported separately if later required.

## 2026-07-17 — Use one notebook as the reproducible CSV processing pipeline

**Decision:** subsequent CSV transformations and analysis steps are implemented in
`csv_data_processing.ipynb`, or accompanied by exact manual launch instructions when a
notebook implementation is impractical. Random operations expose and fix their seed.

**Rationale:** the complete data lineage, parameters, generated artifacts, and plots must
be reproducible manually through an ordered `Run All` workflow.

**Consequence:** avoid isolated exploratory scripts that cannot be replayed from the
notebook. Generated samples retain a separate provenance table mapping them to source
run, channel, file, and row.

## 2026-07-17 — Balance the exploratory sample by source

**Decision:** the small visualization/development sample contains equal numbers of
`60Co` and `252Cf` events. Sampling is uniform without replacement within each run; its
channel composition therefore follows the observed channel event counts within that run.

**Rationale:** equal source counts make side-by-side visual comparison legible and prevent
the much larger Co acquisition from dominating the exploratory sample.

**Consequence:** this is an intentionally balanced exploratory distribution, not an
estimate of real source priors or class probabilities. The provenance table remains the
authoritative mapping to original run/channel frequencies.

## 2026-07-17 — Supersede source-only balance with source-channel stratification

**Decision:** the exploratory visualization sample now contains exactly 1,000 events for
each `source × channel` pair: two sources and channels 0, 2, 3, 4, and 5, for 10,000
events total. This supersedes the source-only balancing decision above for the current
notebook output.

**Rationale:** pulse shapes and acquisition settings can differ by detector channel, so
combining channels hides structure and makes source comparisons ambiguous.

**Consequence:** generate and interpret ten separate source-channel plots. The sample is
deliberately stratified and must not be used to estimate natural source/channel priors.

## 2026-07-17 — Canonical detector labels and channel mapping

**Decision:** use the following channel labels in every plot, table, model report, and
future discussion:

- CH0: `PMT-9102B + T-Stlbn` — 40×40 mm trans-stilbene crystal;
- CH2: `PMT-9102B + T-Stlbn` — 40×40 mm trans-stilbene crystal;
- CH3: `PMT-R6094 + P-Trfnl` — 25×25 mm p-terphenyl crystal;
- CH4: `PMT-R6231 + T-Stlbn` — 40×40 mm trans-stilbene crystal;
- CH5: `PMT-R6231 + P-Trfnl` — 40×40 mm p-terphenyl crystal.

The experimental detector designations additionally describe detectors 0 and 2 as the
two PMT-9102B/trans-stilbene assemblies, detector 1 as PMT-R6231/trans-stilbene,
detector 3 as PMT-R6231/p-terphenyl, and detector 4 as PMT-R6094/p-terphenyl.

**Rationale:** file channel numbers and experimental detector numbers are not the same
numbering scheme for every assembly. A canonical composition label prevents ambiguous
comparisons.

**Consequence:** retain the file channel as a technical identifier, but display it with
the exact `<PMT> + <scintillator>` label above. Do not relabel CH3/CH4/CH5 merely from
their numeric order.

## 2026-07-17 — Publish code and reproducible workflow, not bulk data

**Decision:** the project is versioned in the public GitHub repository
`iv-gonch/PSD_ML`. Commit source code, notebooks, documentation, run metadata, settings,
and small reference screenshots. Exclude raw ROOT files, derived bulk CSV files,
generated samples/plots, `.venv`, and local application caches.

**Rationale:** raw and derived experimental files occupy multiple gigabytes and are
reproducible from the authoritative ROOT data; committing them would make the repository
impractical and risks exceeding GitHub file limits.

**Consequence:** `.gitignore` defines the publication boundary. Data distribution, if
needed later, must use a dedicated dataset release or external storage with checksums and
documented provenance.
