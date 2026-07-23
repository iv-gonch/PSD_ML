# PSD_ML project context

Last updated: 2026-07-21

## Goal

Investigate pulse-shape information for separating neutron-like, gamma-like, and other
events. The initial study compares three experimental mixtures:

- `B`: background without controlled neutron or gamma sources;
- `G`: gamma plus other events, without controlled neutrons;
- `M`: neutron plus gamma plus other events.

The primary objective is now to discover and validate a new pulse-shape metric that
outperforms classical PSD, without assuming in advance that VAE is the winning method.
The associated scientific objectives are to interpret learned factors through waveform
shape and physical conditions, find a minimal factorized latent space, isolate a
particle-informative/domain-robust component, and later build condition-aware PSD from
controlled multi-condition measurements.

## Current state

- The detailed analysis plan is saved in `docs/signal_processing_plan.md`.
- `Идеи развития PSD_ML.md` provides the project primer on AE, VAE, CVAE, latent
  interpretation, factorization, Domain Adaptation and cross-detector transfer.
- The proposed real-time extension is specified in `docs/adaptive_psd_plan.md`: measured
  PMT voltage/temperature/field enter a condition-aware classifier; guarded domain
  adaptation is a later residual correction, while MPC is reserved for a separate
  physical-control loop with actuators and an independent calibration reference.
- A colleague's 7-slide VAE report has been reviewed in
  `docs/colleague_vae_report_review.md`. Its three-dimensional VAE exposes a promising
  tail-controlling latent coordinate, but it lacks documented label provenance,
  energy/channel/run controls and independent validation. It will be treated as a
  candidate `new_PSD` score within the common benchmark; a Qlong/channel/condition-aware
  conditional VAE is the preferred joint extension.
- `real_data_vae.ipynb` now reproduces an equivalent three-dimensional VAE on real data.
  It trains five independent channel models on 10,000 Cf events per channel with three
  seeds (15 models total); Co is an external control and has no training split. The
  notebook calls only `psd_ml/vae.py`, verifies every sampled CSV waveform against ROOT,
  retains `low_snr`, exports event-level latents and audit tables, and embeds 30 Plotly
  figures. The generated artifacts live under ignored
  `gamma_n_data/samples/vae_real/`.
- The first VAE audit does not support a universal single latent coordinate: by the
  documented variance/KL diagnostic, all three coordinates are active for CH0 and CH2,
  two for CH4, and mainly one for CH3 and CH5. Coordinate matching across seeds is not
  uniformly stable, especially for CH2, so coordinate numbers cannot be assigned a fixed
  physical interpretation. Co-vs-Cf run AUC ranges only about 0.50–0.62 and is not a
  neutron/gamma metric.
- Experimental material and ROOT-file exploration utilities are under `gamma_n_data/`.
- A Python virtual environment exists at `.venv/`.
- Git is initialized on branch `main`; the public remote is
  `https://github.com/iv-gonch/PSD_ML`. Bulk data and local environments are excluded by
  `.gitignore`.
- Available acquisitions currently comprise one 30-minute `60Co` gamma-control run and
  one 60-minute `252Cf` mixed neutron/gamma run, on channels 0, 2, 3, 4, and 5.
- `Data_*.root` files contain a `Data` TTree with `Energy`, `EnergyShort`, timestamps,
  flags, and 144-sample waveforms. The recorded `sampleTime=2000` is consistent with
  2000 ps = 2 ns per sample: 70/20/30 waveform samples reproduce the configured
  140/40/60 ns QDC gates.
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
- The notebook now also performs a channel-separated Cf shape-branch search on a
  deterministic 25,000-event sample (5,000 per channel). CSV rows are joined to ROOT
  `Energy`/`EnergyShort` by provenance and all 144 waveform samples are verified exactly.
  Twelve equal-count Energy intervals per channel are analyzed with a peak-normalized
  late-area score, one-versus-two Gaussian BIC, component separation, minimum-size and
  bootstrap criteria. `low_snr` events are retained; only structural QC failures are
  excluded.
- Two shape components pass all 8/10/12 Energy-bin choices for CH2, CH3, CH4, and CH5.
  CH0 passes the formal two-of-three sensitivity rule at 8 and 10 bins, but fails the
  narrowest 12-bin primary analysis because supported intervals are not sufficiently
  consecutive; its evidence is therefore weaker and needs more statistics. The lowest
  Energy intervals generally remain unresolved. Higher-tail candidates are a minority
  and must not yet be interpreted as neutrons. Only one Cf run is available.
- The five `cf_energy_binned_form_figures` were redesigned for legibility. Each Energy
  interval is now one spacious row: an internal interval-label column, observed/fitted
  late-area-score density, and normalized waveform median with a 10–90% band. The legend
  occupies a reserved right margin instead of competing with the title. Score figures use
  the same right-margin legend policy. Browser renders were inspected at 1200–1800 px;
  legends, row labels, column headings, and axes remain visible without overlap.
- ROOT `Energy` is not an independent label: it is the recorded firmware `Qlong` charge
  integral; `EnergyShort` is `Qshort`. Settings use 140 ns long, 40 ns short gates
  (60 ns for CH4/CH5), 10 ns pre-gate, identity calibration, and `ADCCH` output. Offline
  integration of samples `[15:85]`, `[15:35]`, or `[15:45]` reproduces these fields with
  correlation 0.99997–1.00000 and scale approximately 1/32. Values are therefore ADC
  channels, not calibrated keV, even though `run.info` carries a formal `keV` label.
- The current `shape_score = positive_area[40:100] / positive_area[15:100]` is a
  deliberately simple PSD-like demonstration feature, not an optimized `new_PSD`. It is
  related to but not identical to CoMPASS PSD (correlations 0.6428–0.7695). The notebook
  now explains both score plots and records the required ML/window-optimization stage.
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
- `Идеи развития PSD_ML.md` — conceptual guide based on the VAE discussion.
- `docs/adaptive_psd_plan.md` — condition-aware inference, domain adaptation, online
  safety, voltage-sweep validation, and the optional MPC control layer.
- `docs/colleague_vae_report_review.md` — interpretation, limitations, questions for the
  colleague, and the concrete VAE/CVAE integration protocol.
- `docs/DECISIONS.md` — durable technical and scientific decisions.
- `docs/WORKLOG.md` — chronological handoffs from all Codex tasks.
- `docs/REPORT_NOTES.md` — mandatory caveats and analyses for the eventual project report.
- `gamma_n_data/` — data, run metadata, notebooks, and ROOT inspection/plotting tools.
- `gamma_n_data/CSV/` — waveform-only CSV exports and their format description.
- `gamma_n_data/export_waveforms_csv.py` — streaming ROOT-to-CSV exporter.
- `csv_data_processing.ipynb` — executable, ordered CSV processing pipeline.
- `real_data_vae.ipynb` — executable real-data VAE experiment and full latent audit.
- `psd_ml/pipeline.py` — reusable implementation of sampling, audit, preprocessing,
  validation, ROOT metadata joins, energy-stratified branch analysis, persistence, and
  Plotly stages called by the notebook.
- `psd_ml/vae.py` — channel-specific VAE, training, persistence, latent audit, and
  interactive result figures called by `real_data_vae.ipynb`.
- `requirements-vae.txt` — pinned additional PyTorch dependency for the VAE experiment.
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

1. Confirm pulse polarity, source geometry, and the meaning of
   `RAW`/`FILTERED`/`UNFILTERED` in the presence of the Cf PSD cuts.
2. Add a background run plus independent repeated `60Co` and `252Cf` runs; complete-run
   validation is impossible with only one run of each condition.
3. Calibrate recorded `Qlong` physically and quantify every QC acceptance curve versus
   calibrated energy, especially for the low-SNR stratum retained in the branch analysis.
4. Investigate whether the rare higher-tail component is physical or residual pileup /
   acquisition selection; compare it with Co and the full classical PSD distribution,
   and resolve the binning-sensitive CH0 result.
5. Re-audit preprocessing and the two-branch result on independent complete runs.
6. Implement the common `new_PSD` benchmark described in
   `docs/signal_processing_plan.md`: strong classical/multi-window baselines first, then
   PCA/linear AE, AE/VAE, time-series and supervised candidates under the same splits.
7. Add a standard latent audit against waveform shape, Qlong, classical PSD, run,
   channel, QC and low-SNR; compare dimensions and seed stability rather than assuming
   three latents or interpreting a coordinate by its index.
8. Before online adaptation, collect randomized repeated Co/Cf/background/calibration
   runs over a safe PMT-voltage grid, recording setpoint/readback and environmental
   telemetry. Compare gain correction, explicit condition inputs, adapters, and domain
   adaptation using held-out runs and held-out conditions.
9. Compare the completed real-data VAE with classical PSD and optimized multi-window
   baselines on independent complete runs and explicit low-Qlong strata. Obtain the
   colleague's code, label provenance and exact preprocessing before claiming an exact
   reproduction or interpreting latent separation physically.
