# PSD_ML project context

Last updated: 2026-07-20

## Goal

Investigate pulse-shape information for separating neutron-like, gamma-like, and other
events. The initial study compares three experimental mixtures:

- `B`: background without controlled neutron or gamma sources;
- `G`: gamma plus other events, without controlled neutrons;
- `M`: neutron plus gamma plus other events.

The immediate objective is to establish whether a reproducible shape component appears
in `M` and remains distinguishable from `G` after controlling for energy, run identity,
baseline, timing, and other acquisition artifacts.

## Current state

- The detailed analysis plan is saved in `docs/signal_processing_plan.md`.
- Experimental material and ROOT-file exploration utilities are under `gamma_n_data/`.
- A Python virtual environment exists at `.venv/`.
- Git is initialized on branch `main`; the public remote is
  `https://github.com/iv-gonch/PSD_ML`. Bulk data and local environments are excluded by
  `.gitignore`.
- Available acquisitions currently comprise one 30-minute `60Co` gamma-control run and
  one 60-minute `252Cf` mixed neutron/gamma run, on channels 0, 2, 3, 4, and 5.
- `Data_*.root` files contain a `Data` TTree with `Energy`, `EnergyShort`, timestamps,
  flags, and 144-sample waveforms. The recorded `sampleTime` setting is 2000; its unit
  still needs experimental confirmation.
- The current CoMPASS PSD is reproducible as
  `(Energy - EnergyShort) / Energy`; configured long/short gates are 140/40 ns, with a
  60 ns short-gate override on channels 4 and 5, and a 10 ns pre-gate.
- The `252Cf` run metadata reports nonzero PSD rejection and its settings contain
  channel-specific PSD cuts/thresholds absent from the `60Co` run. The exact semantics
  of `UNFILTERED` versus these acquisition cuts must be resolved before comparing runs.
- All ten waveform trees have a headerless CSV mirror under `gamma_n_data/CSV/`, with
  one pulse per row and 144 samples per row. The reproducible exporter is
  `gamma_n_data/export_waveforms_csv.py`.
- `csv_data_processing.ipynb` is the primary reproducible CSV pipeline. Its code cells
  expose parameters and call stage-level functions from the project mini-library
  `psd_ml/pipeline.py`; processing implementations are kept out of the notebook. Its
  first stage creates a seeded sample of 10,000 events: exactly 1,000 for every
  `source × channel` pair. It writes an aligned provenance table and ten channel-specific
  interactive HTML waveform plots under `gamma_n_data/samples/`; PNG generation is
  disabled. The same ten plots are stored directly in notebook outputs using Plotly MIME,
  without iframes.
- The same notebook now contains a completed technical audit and preprocessing stage for
  that fixed sample: per-pulse 12-sample median baseline subtraction, polarity inversion,
  sub-sample CFD-50 alignment to sample 20, amplitude-retaining and peak-normalized
  output branches, and non-destructive QC flags. Generated processed waveforms, features,
  group audit CSV, and five channel-level shape-summary HTML files are under
  `gamma_n_data/samples/`.
- Audit evidence: no sampled ADC clipping; median baseline RMS 3.136 ADC; 95% absolute
  baseline slope 0.584 ADC/sample; raw CFD-50 1–99% range 16.62–25.10 samples; relative
  tail residual 95/99% quantiles 1.31%/3.10%; post-alignment 99% CFD error 0.2355 sample.
  No smoothing or linear detrending is currently justified. Source labels are mixture/run
  conditions, not event-level particle truth.
- The notebook includes an interactive visual audit of all 192 events for which at least
  one QC flag is active. Seven non-empty source×channel plots show baseline-subtracted,
  sign-inverted forms before alignment or normalization; every event appears once and
  hover lists all of its flags and diagnostic values. Three groups with no QC events are
  reported explicitly. Matching standalone HTML files use the prefix `qc_flagged_` under
  `gamma_n_data/samples/`.
- The project-local Jupyter kernel spec is `.jupyter/kernels/psd-ml/kernel.json`; Plotly
  6.9.0 provides offline interactive zoom/pan plots and is installed in `.venv`.
- Canonical channel labels are: CH0/CH2 `PMT-9102B + T-Stlbn`, CH3
  `PMT-R6094 + P-Trfnl`, CH4 `PMT-R6231 + T-Stlbn`, and CH5
  `PMT-R6231 + P-Trfnl`. Crystal sizes and the detector-number mapping are recorded in
  `docs/DECISIONS.md`.
- No background (`B`) run, independent repeat runs, validated neutron labels, classical
  PSD benchmark, run-generalization validation, or trained classifier is present.

## Important paths

- `docs/signal_processing_plan.md` — full staged research plan.
- `docs/DECISIONS.md` — durable technical and scientific decisions.
- `docs/WORKLOG.md` — chronological handoffs from all Codex tasks.
- `docs/REPORT_NOTES.md` — mandatory caveats and analyses for the eventual project report.
- `gamma_n_data/` — data, run metadata, notebooks, and ROOT inspection/plotting tools.
- `gamma_n_data/CSV/` — waveform-only CSV exports and their format description.
- `gamma_n_data/export_waveforms_csv.py` — streaming ROOT-to-CSV exporter.
- `csv_data_processing.ipynb` — executable, ordered CSV processing pipeline.
- `psd_ml/pipeline.py` — reusable implementation of sampling, audit, preprocessing,
  validation, persistence, and Plotly stages called by the notebook.
- `gamma_n_data/samples/` — reproducibly generated test samples and figures.
- `gamma_n_data/JUPYTER.md` — notebook usage notes.
- `gamma_n_data/VIEW_ROOT_FILES.md` — ROOT file viewing instructions.
- `start_jupyter.sh` — project notebook launcher.
- `.jupyter/kernels/psd-ml/kernel.json` — kernel bound to this project's `.venv`.
- `.gitignore` — publication boundary excluding ROOT, derived CSV/samples, and local
  runtime state.

## Working principles

- Split train, validation, and test data by complete runs/series, never by individual
  pulses sampled from the same run.
- Treat PCA as diagnostic rather than proof of separability.
- Check domain/run leakage before interpreting `G` versus `M` separation physically.
- Until independent neutron labels exist, call outputs neutron-like scores or candidates,
  not calibrated neutron probabilities.
- Treat `low_snr` as a diagnostic stratum, not permanent rejection. Report QC acceptance
  and algorithm performance versus amplitude/calibrated energy, with and without the
  current `SNR < 20` cut.
- Establish an interpretable PSD baseline before using a 1D CNN.

## Next steps

1. Confirm the unit of `sampleTime=2000`, pulse polarity, source geometry, and the
   meaning of `RAW`/`FILTERED`/`UNFILTERED` in the presence of the Cf PSD cuts.
2. Add a background run plus independent repeated `60Co` and `252Cf` runs; complete-run
   validation is impossible with only one run of each condition.
3. Join waveforms to ROOT `Energy`, establish the amplitude–energy relation, and quantify
   QC acceptance versus energy, especially for the 144 current `low_snr` events.
4. Re-audit the chosen preprocessing on independent complete runs and add the remaining
   metadata/rejection and energy-stratified diagnostics per channel.
5. Implement the classical PSD benchmark described in `docs/signal_processing_plan.md`.
6. Only after leakage checks, build interpretable shape features and compare them with
   classical PSD; defer 1D CNN work until stable candidate labels exist.
