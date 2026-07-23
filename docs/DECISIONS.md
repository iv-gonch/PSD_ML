# Project decisions

This file records durable decisions shared by all project tasks. New entries should state
the date, decision, rationale, and consequences. If a decision is superseded, retain it
and point to the replacing entry.

## 2026-07-23 — Interpret VAE geometry through latent principal directions

**Decision:** interpret the three-dimensional VAE through channel-specific principal
directions of posterior means fitted on structural-ok Cf-validation events, not through
the arbitrary `z1`/`z2`/`z3` coordinate numbers. Order directions by latent covariance
eigenvalue. Orient their signs toward increasing 40:100 tail fraction when that
association is measurable; otherwise use a deterministic loading convention. Mark a
direction with less than 1% explained latent variance as collapsed and do not interpret
its correlations or traversal.

For each active direction, report raw Spearman associations and partial rank
associations after quadratic adjustment for Qlong rank. Compare the direction with
multiple observed shape measures, classical PSD, amplitude/SNR, baseline, raw CFD/peak
timing, tail recovery, reconstruction error, real waveform deciles, and decoder
traversal.

**Rationale:** VAE coordinates can rotate, permute, and change sign, while the observed
ball/ribbon/plane geometry concerns the covariance subspace. Qlong affects both waveform
statistics and latent position, so an unadjusted correlation can mistake an energy trend
for an independent shape factor. Tiny collapsed directions can show numerically large
but scientifically meaningless correlations.

**Consequence:** the resulting PCs are reproducible diagnostic directions for this fit,
not identified physical causes, particle labels, or universally transferable axes.
Technical timing associations and Co-vs-Cf run AUC must remain visible when evaluating a
PC as a future `new_PSD` candidate. Independent runs and controlled measurements remain
required.

## 2026-07-21 — Reframe the research around a benchmarked new PSD metric

**Decision:** the primary research objective is to discover and validate the most
effective pulse-shape separation metric, without committing in advance to VAE. AE/VAE,
physical features, time-series methods, metric learning, and compact supervised networks
must share one run-separated benchmark. The second objective is a minimal factorized
representation with an explicitly tested particle component; condition-aware PSD follows
only after controlled multi-condition data are collected.

**Rationale:** reconstruction does not optimize particle separation, standard VAE does
not guarantee interpretable/disentangled coordinates, and current Co/Cf mixture labels
cannot prove event-level particle identity. Explicit conditioning and particle-specific
losses address different questions and require appropriate experimental data.

**Consequence:** use the revised `docs/signal_processing_plan.md`; treat VAE as a
candidate, audit every latent against shape, charge, run, channel, QC and conditions, and
require independent runs/anchors before claiming `z_particle` or neutron efficiency. The
conceptual guide is `Идеи развития PSD_ML.md`.

## 2026-07-21 — Evaluate the colleague's VAE as a candidate shape score

**Decision:** integrate the reported three-latent VAE as a candidate representation and
diagnostic method in the common PSD benchmark, not as an already validated neutron/gamma
classifier or physical simulator. Use PSD_ML preprocessing/provenance and evaluate the
latent score by channel, Qlong stratum, complete run, QC, and low-SNR group. The preferred
joint extension is a conditional VAE with known Qlong/channel/operating conditions passed
explicitly and a residual `z_shape`.

**Rationale:** the report shows a promising latent coordinate controlling the decoded
tail, but does not document label provenance, run-separated validation, energy/channel
control, losses, or independent metrics. Smooth decoder traversal alone does not prove
physical intermediate events. Unconditional training could encode run, gain, energy, or
channel artifacts.

**Consequence:** obtain code, data contract, label origin, architecture/loss, seeds, and
run metadata from the colleague. Compare VAE, classical PSD, the current shape score,
physical features, and simple PCA/linear-AE baselines under one validation protocol.
Do not augment training with decoded pulses until their physical fidelity is validated.
The full review is `docs/colleague_vae_report_review.md`.

## 2026-07-20 — Separate condition-aware PSD, online adaptation, and detector control

**Decision:** develop the future real-time system as three distinct layers: fast
condition-aware pulse classification, slow guarded adaptation/calibration, and an
optional physical-control loop. Known PMT voltage and other measured conditions are
explicit model inputs. Domain adaptation is a comparative method after controlled
multi-condition data exist; unconstrained test-time self-training is not the initial
solution. MPC is considered only when the system can actuate detector settings and has
an identified dynamic model plus an independent calibration reference.

**Rationale:** measured conditions contain useful information that should not be erased
by forced domain invariance. Co and Cf have different and changing class proportions, so
unconditional marginal alignment can remove the rare physical component. MPC solves a
constrained control problem; it does not itself classify pulses.

**Consequence:** collect repeated, randomized voltage-sweep runs before adaptation work;
validate by complete run and held-out condition. Any online adapter requires a frozen
reference, bounded updates, OOD/unknown handling, audit logs, rollback, and shadow-mode
validation. The detailed plan is `docs/adaptive_psd_plan.md`.

## 2026-07-20 — Require rendered-image QA for dense Plotly layouts

**Decision:** figures with multiple subplots, annotations, or external legends must be
checked from an actual browser render at both a wide viewport and a notebook-like width
before the layout is accepted. For the current Cf figures, legends occupy a reserved
right margin and Energy-interval annotations occupy an internal label column rather than
negative paper coordinates.

**Rationale:** valid Plotly coordinates do not prove that browser-rendered text is
readable. Earlier legends overlapped or crossed the title, and annotations placed beyond
the paper boundary were clipped even though the notebook executed successfully.

**Consequence:** structural checks remain useful but cannot replace visual inspection.
Record the tested viewport sizes in the worklog; do not finish a visualization change
while known labels, legends, titles, or axes overlap or leave the visible canvas.

## 2026-07-17 — Use files as cross-chat project memory

**Decision:** `docs/PROJECT_CONTEXT.md`, `docs/WORKLOG.md`, and this file are the canonical
communication layer between Codex chats working in `PSD_ML`.

**Rationale:** sibling chats do not automatically share their full transcripts, while all
tasks in the project can read and update the same workspace.

**Consequence:** every material task must read shared memory at startup and publish a
handoff before finishing, as required by the root `AGENTS.md`.

## 2026-07-20 — Treat ROOT Energy as recorded QDC charge, not independent truth

**Decision:** describe ROOT `Energy` as the recorded firmware long-gate integral
`Qlong` and `EnergyShort` as `Qshort`. Use “ADC channels” until a physical calibration
has been established, despite the formal `keV` unit in `run.info`. In figures and
explanations, retain the ROOT branch name only in parentheses.

The current `shape_score = positive_area[40:100] / positive_area[15:100]` is a fixed,
interpretable demonstration feature. Its window boundaries are not optimized and it is
not the proposed final `new_PSD`.

**Rationale:** acquisition settings specify a 140 ns long QDC gate, 40 ns short gate
(60 ns on CH4/CH5), 10 ns pre-gate, `ADCCH` output, and identity calibration. An offline
check on 25,000 Cf waveforms reproduces `Energy` with samples `[15:85]` and
`EnergyShort` with `[15:35]` or `[15:45]`: correlations are 0.99997–1.00000 and fitted
scales are approximately 1/32. Thus Qlong and the waveform shape are statistically
coupled observations of the same pulse rather than independent axes.

**Consequence:** energy-stratified plots test residual shape differences at similar
recorded charge, not shape against independently measured particle energy. Future work
must test alternative energy proxies/calibration and optimize multi-window or learned
features using run-separated validation. Benchmark physical-feature models before
waveform ML, and defer a 1D CNN until labels and independent runs are adequate.

## 2026-07-20 — Pair score distributions with waveform bands in Cf branch plots

**Decision:** the primary per-channel visualization of the Energy-stratified Cf analysis
uses one row per Energy interval and two aligned columns. The left column shows the
observed late-area-score density and fitted components; the right column shows the
median and 10–90% band of the corresponding normalized waveforms. Branch colors appear
only when the interval passes every predeclared statistical criterion. Energy bounds,
event count, and the pass/fail status are displayed once in a dedicated row label.

**Rationale:** the former grid of waveform heatmaps obscured the actual test of
bimodality, repeated long subplot titles overlapped, and the legend was difficult to
find. The paired view directly separates two questions: whether two score components
exist at approximately fixed Energy, and where their waveform shapes differ.

**Consequence:** interpret a branch only from repeated agreement of both columns across
neighboring Energy rows. A colored fit in one row remains insufficient evidence, and
lower-/higher-tail labels remain geometric descriptions rather than particle identities.

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

## 2026-07-21 — Train the first real-data VAE per detector channel

**Decision:** the first reproduction of the colleague's VAE is an equivalent documented
three-dimensional VAE trained separately for each detector channel. Each channel uses
10,000 `252Cf` events, a deterministic Qlong-stratified 80/20 train/validation split,
and three model seeds. The `60Co` events are never used for fitting and serve only as an
external gamma-control acquisition.

**Rationale:** channel-specific models avoid asking the latent space to encode detector
hardware, while the untouched Co acquisition exposes run/domain separation. Three
coordinates reproduce the reported latent dimension without assuming that one fixed
coordinate is the only physically meaningful one. Multiple seeds expose permutation,
sign, rotation, and posterior-collapse ambiguity.

**Consequence:** this experiment is an unsupervised representation audit, not a validated
neutron/gamma classifier. Random event splitting within the only available Cf run cannot
measure run generalization. Co-vs-Cf AUC is reported explicitly as acquisition/run AUC,
never as particle AUC. Structural QC failures are excluded from fitting and aggregate
diagnostics, while `low_snr` events remain eligible.
