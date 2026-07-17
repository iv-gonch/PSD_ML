"""Reproducible data pipeline used by ``csv_data_processing.ipynb``.

The public functions deliberately correspond to the notebook's conceptual stages:
inventory, stratified sampling, raw audit, preprocessing, validation, and plotting.
The notebook exposes parameters and results; implementation details live here.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


DETECTOR_LABELS = {
    0: "PMT-9102B + T-Stlbn",
    2: "PMT-9102B + T-Stlbn",
    3: "PMT-R6094 + P-Trfnl",
    4: "PMT-R6231 + T-Stlbn",
    5: "PMT-R6231 + P-Trfnl",
}

RUN_ORDER = ("call_all_60Co", "call_all_252Cf")
PLOT_SPECS = {
    "call_all_60Co": {"title": "60Co: gamma", "color": "#1f77b4", "slug": "60Co"},
    "call_all_252Cf": {
        "title": "252Cf: neutron + gamma",
        "color": "#ff7f0e",
        "slug": "252Cf",
    },
}
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int = 20260717
    pulses_per_group: int = 1000
    expected_samples: int = 144
    baseline_samples: int = 12
    cfd_fraction: float = 0.50
    alignment_target: float = 20.0
    adc_min: int = 0
    adc_max: int = 16383
    low_snr_threshold: float = 20.0
    tail_ratio_threshold: float = 0.05
    chunk_size_bytes: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    csv_root: Path
    output_dir: Path
    source_files: tuple[Path, ...]


@dataclass(frozen=True)
class InventoryItem:
    path: Path
    rows: int
    bytes: int


@dataclass(frozen=True)
class Inventory:
    items: tuple[InventoryItem, ...]
    total_rows: int
    total_bytes: int
    mean_row_bytes: float
    test_pulses: int


@dataclass(frozen=True)
class Sample:
    waveforms: np.ndarray
    provenance: tuple[tuple[str, int, str, int], ...]
    runs: np.ndarray
    channels: np.ndarray


@dataclass(frozen=True)
class Audit:
    baseline: np.ndarray
    baseline_rms: np.ndarray
    baseline_slope: np.ndarray
    positive_raw: np.ndarray
    amplitude: np.ndarray
    peak_index: np.ndarray
    snr: np.ndarray
    tail_residual: np.ndarray
    tail_ratio: np.ndarray
    cfd_time: np.ndarray
    source_rows: np.ndarray
    baseline_sensitivity: tuple[tuple[int, float, float], ...]
    baseline_drift_correlations: dict[tuple[str, int], float]


@dataclass(frozen=True)
class Processed:
    aligned_amplitude: np.ndarray
    aligned_normalized: np.ndarray
    aligned_peak: np.ndarray
    clipped: np.ndarray
    low_snr: np.ndarray
    baseline_noisy: np.ndarray
    invalid_alignment: np.ndarray
    tail_not_recovered: np.ndarray
    possible_multipeak: np.ndarray
    quality_ok: np.ndarray
    baseline_noise_thresholds: dict[int, float]


@dataclass(frozen=True)
class Validation:
    aligned_cfd: np.ndarray
    summary_rows: tuple[dict[str, object], ...]
    summary_csv: Path
    alignment_error_q99: float


@dataclass(frozen=True)
class ShapeCharacteristics:
    processed_peak: np.ndarray
    decay50_time: np.ndarray
    late_area_fraction: np.ndarray
    rows: tuple[dict[str, object], ...]


def configure_plotly() -> None:
    """Use a native notebook MIME bundle instead of an iframe."""

    pio.renderers.default = "plotly_mimetype"


def discover_project(start: Path | None = None) -> ProjectPaths:
    """Locate project data from the current folder or any of its parents."""

    start = (start or Path.cwd()).resolve()
    root = next(
        (folder for folder in (start, *start.parents) if (folder / "gamma_n_data" / "CSV").is_dir()),
        None,
    )
    if root is None:
        raise FileNotFoundError("Не найдена папка gamma_n_data/CSV")
    csv_root = root / "gamma_n_data" / "CSV"
    output_dir = root / "gamma_n_data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = tuple(sorted(csv_root.glob("call_all_*/*.csv")))
    if not source_files:
        raise FileNotFoundError(f"В {csv_root} не найдены исходные CSV")
    return ProjectPaths(root, csv_root, output_dir, source_files)


def _count_lines(path: Path, chunk_size: int) -> int:
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            count += chunk.count(b"\n")
    return count


def inventory_sources(paths: ProjectPaths, config: PipelineConfig) -> Inventory:
    """Count source pulses without loading the multi-gigabyte dataset into RAM."""

    items = []
    for path in paths.source_files:
        rows = _count_lines(path, config.chunk_size_bytes)
        if rows == 0:
            raise ValueError(f"Пустой файл: {path}")
        items.append(InventoryItem(path, rows, path.stat().st_size))
    total_rows = sum(item.rows for item in items)
    total_bytes = sum(item.bytes for item in items)
    return Inventory(
        tuple(items),
        total_rows,
        total_bytes,
        total_bytes / total_rows,
        len(items) * config.pulses_per_group,
    )


def print_inventory(paths: ProjectPaths, inventory: Inventory, config: PipelineConfig) -> None:
    print(f"Корень проекта: {paths.root}")
    print(f"Найдено исходных файлов: {len(paths.source_files)}")
    for path in paths.source_files:
        print(" -", path.relative_to(paths.root))
    print(f"Всего импульсов: {inventory.total_rows:,}")
    print(f"Объём исходных CSV: {inventory.total_bytes / 1024**3:.3f} ГиБ")
    print(f"Средний размер строки: {inventory.mean_row_bytes:.1f} байт")
    print(f"Групп источник × канал: {len(inventory.items)}")
    print(
        f"Будет выбрано импульсов: {inventory.test_pulses:,} "
        f"({config.pulses_per_group:,} на группу)"
    )
    print(
        f"Ожидаемый размер: "
        f"{inventory.test_pulses * inventory.mean_row_bytes / 1024**2:.3f} МиБ"
    )


def _parse_source(path: Path) -> tuple[str, int]:
    match = re.search(r"Data_CH(\d+)@", path.name)
    if not match:
        raise ValueError(f"Не удалось определить канал из имени {path.name}")
    return path.parent.name, int(match.group(1))


def sample_waveforms(inventory: Inventory, config: PipelineConfig) -> Sample:
    """Draw the same number of random rows from every source × channel file."""

    rng = np.random.default_rng(config.random_seed)
    rows_by_file: dict[Path, list[int]] = {}
    for item in inventory.items:
        if item.rows < config.pulses_per_group:
            raise ValueError(
                f"В {item.path.name} недостаточно импульсов: "
                f"{item.rows} < {config.pulses_per_group}"
            )
        rows_by_file[item.path] = np.sort(
            rng.choice(item.rows, size=config.pulses_per_group, replace=False)
        ).tolist()

    waveforms: list[np.ndarray] = []
    provenance: list[tuple[str, int, str, int]] = []
    for item in inventory.items:
        path = item.path
        wanted = rows_by_file[path]
        wanted_position = 0
        run, channel = _parse_source(path)
        with path.open("r", encoding="ascii") as stream:
            for row_index, line in enumerate(stream):
                if row_index != wanted[wanted_position]:
                    continue
                pulse = np.fromstring(line, dtype=np.int16, sep=",")
                if pulse.size != config.expected_samples:
                    raise ValueError(
                        f"{path.name}, строка {row_index}: ожидалось "
                        f"{config.expected_samples} отсчётов, найдено {pulse.size}"
                    )
                waveforms.append(pulse)
                provenance.append((run, channel, path.name, row_index))
                wanted_position += 1
                if wanted_position == len(wanted):
                    break

    array = np.stack(waveforms)
    expected_shape = (inventory.test_pulses, config.expected_samples)
    if array.shape != expected_shape:
        raise AssertionError(f"Получена форма {array.shape}, ожидалась {expected_shape}")
    provenance_tuple = tuple(provenance)
    return Sample(
        array,
        provenance_tuple,
        np.array([row[0] for row in provenance_tuple]),
        np.array([row[1] for row in provenance_tuple]),
    )


def save_sample(
    sample: Sample, paths: ProjectPaths, config: PipelineConfig
) -> tuple[Path, Path]:
    stem = (
        f"test_waveforms_{config.pulses_per_group}_per_group_"
        f"seed_{config.random_seed}"
    )
    sample_csv = paths.output_dir / f"{stem}.csv"
    provenance_csv = paths.output_dir / f"{stem}_provenance.csv"
    np.savetxt(sample_csv, sample.waveforms, fmt="%d", delimiter=",")
    with provenance_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_row", "run", "channel", "source_file", "source_row"])
        for sample_row, fields in enumerate(sample.provenance):
            writer.writerow((sample_row, *fields))
    print(f"Импульсы: {sample_csv.relative_to(paths.root)} ({sample_csv.stat().st_size / 1024**2:.3f} МиБ)")
    print(f"Происхождение строк: {provenance_csv.relative_to(paths.root)}")
    return sample_csv, provenance_csv


def _segmented_xy(pulses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_pulses, n_samples = pulses.shape
    x = np.tile(np.append(np.arange(n_samples, dtype=float), np.nan), n_pulses)
    y = np.column_stack([pulses, np.full(n_pulses, np.nan)]).ravel()
    return x, y


def plot_raw_groups(sample: Sample, paths: ProjectPaths, config: PipelineConfig) -> list[go.Figure]:
    """Show and save one interactive raw-waveform figure per source × channel."""

    figures = []
    stem = paths.output_dir / (
        f"test_waveforms_{config.pulses_per_group}_per_group_"
        f"seed_{config.random_seed}"
    )
    for run in RUN_ORDER:
        spec = PLOT_SPECS[run]
        for channel in sorted(DETECTOR_LABELS):
            mask = (sample.runs == run) & (sample.channels == channel)
            pulses = sample.waveforms[mask]
            if len(pulses) != config.pulses_per_group:
                raise AssertionError(f"{run}, CH{channel}: найдено {len(pulses)} импульсов")
            x, y = _segmented_xy(pulses)
            fig = go.Figure(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="lines",
                    hoverinfo="skip",
                    line={"color": spec["color"], "width": 0.55},
                    opacity=0.12,
                )
            )
            fig.update_layout(
                title=(
                    f"{spec['title']} — CH{channel}: {DETECTOR_LABELS[channel]} — "
                    f"{len(pulses)} pulses, seed={config.random_seed}"
                ),
                xaxis_title="Sample index",
                yaxis_title="ADC value",
                template="plotly_white",
                height=650,
                dragmode="zoom",
                showlegend=False,
            )
            html_path = Path(f"{stem}_{spec['slug']}_CH{channel}_interactive.html")
            fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
            fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
            print("Интерактивно:", html_path.relative_to(paths.root))
            figures.append(fig)
    return figures


def _leading_crossings(
    signals: np.ndarray, amplitudes: np.ndarray, peaks: np.ndarray, fraction: float
) -> np.ndarray:
    result = np.full(len(signals), np.nan, dtype=float)
    for i, signal in enumerate(signals):
        target = fraction * amplitudes[i]
        hits = np.flatnonzero(signal[: peaks[i] + 1] >= target)
        if len(hits) == 0 or hits[0] == 0:
            continue
        j = int(hits[0])
        y0, y1 = signal[j - 1], signal[j]
        result[i] = j - 1 + (target - y0) / (y1 - y0) if y1 != y0 else float(j)
    return result


def audit_waveforms(sample: Sample, config: PipelineConfig) -> Audit:
    """Measure baseline, timing, noise, amplitude, and tail behavior before processing."""

    waveforms = sample.waveforms
    n_baseline = config.baseline_samples
    baseline_window = waveforms[:, :n_baseline].astype(float)
    baseline = np.median(baseline_window, axis=1)
    baseline_centered = baseline_window - baseline[:, None]
    baseline_rms = np.sqrt(np.mean(baseline_centered**2, axis=1))
    x0 = np.arange(n_baseline, dtype=float)
    x0_centered = x0 - x0.mean()
    baseline_slope = (baseline_centered * x0_centered).sum(axis=1) / (
        x0_centered**2
    ).sum()
    positive_raw = baseline[:, None] - waveforms.astype(float)
    amplitude = positive_raw.max(axis=1)
    peak_index = positive_raw.argmax(axis=1)
    snr = amplitude / np.maximum(baseline_rms, 0.5)
    tail_residual = np.median(waveforms[:, -20:].astype(float), axis=1) - baseline
    tail_ratio = np.abs(tail_residual) / np.maximum(amplitude, 1.0)
    cfd_time = _leading_crossings(positive_raw, amplitude, peak_index, config.cfd_fraction)
    source_rows = np.array([row[3] for row in sample.provenance])
    sensitivity = []
    for window in (8, 10, 12, 14, 16):
        alternative = np.median(waveforms[:, :window], axis=1)
        delta = np.abs(alternative - baseline)
        sensitivity.append((window, float(np.median(delta)), float(np.quantile(delta, 0.95))))
    correlations = {}
    for run in RUN_ORDER:
        for channel in sorted(DETECTOR_LABELS):
            group = (sample.runs == run) & (sample.channels == channel)
            correlations[(run, channel)] = float(
                np.corrcoef(source_rows[group], baseline[group])[0, 1]
            )
    return Audit(
        baseline,
        baseline_rms,
        baseline_slope,
        positive_raw,
        amplitude,
        peak_index,
        snr,
        tail_residual,
        tail_ratio,
        cfd_time,
        source_rows,
        tuple(sensitivity),
        correlations,
    )


def print_audit(sample: Sample, audit: Audit, config: PipelineConfig) -> None:
    waveforms = sample.waveforms
    print(f"Проверено импульсов: {len(waveforms):,}")
    print(f"Диапазон ADC в выборке: {waveforms.min()} … {waveforms.max()}")
    print(
        f"Точное достижение границ ADC: min="
        f"{np.sum(waveforms.min(axis=1) == config.adc_min)}, "
        f"max={np.sum(waveforms.max(axis=1) == config.adc_max)}"
    )
    print(f"Baseline RMS, медиана: {np.median(audit.baseline_rms):.3f} ADC")
    print(
        f"|наклон baseline|, 95%: "
        f"{np.quantile(np.abs(audit.baseline_slope), 0.95):.3f} ADC/sample"
    )
    print(f"Амплитуда, медиана: {np.median(audit.amplitude):.1f} ADC")
    print(
        f"SNR, медиана: {np.median(audit.snr):.1f}; событий SNR < "
        f"{config.low_snr_threshold:g}: {np.sum(audit.snr < config.low_snr_threshold)}"
    )
    print("CFD-50 время, 1/50/99%:", np.nanquantile(audit.cfd_time, [0.01, 0.5, 0.99]))
    print(
        "|остаток хвоста|/амплитуда, 50/95/99%:",
        np.quantile(audit.tail_ratio, [0.5, 0.95, 0.99]),
    )
    print(
        "Чувствительность baseline к длине начального окна "
        f"(относительно {config.baseline_samples} отсчётов):"
    )
    for window, median, q95 in audit.baseline_sensitivity:
        print(f"  {window:2d}: median={median:.3f}, q95={q95:.3f} ADC")
    print("Корреляция baseline с номером исходной строки (контроль дрейфа по времени):")
    for (run, channel), correlation in audit.baseline_drift_correlations.items():
        print(f"  {run:14s} CH{channel}: r={correlation:+.4f}")


def _conservative_multipeak_flags(signals: np.ndarray) -> np.ndarray:
    kernel = np.ones(5) / 5
    smooth = np.stack([np.convolve(signal, kernel, mode="same") for signal in signals])
    smooth_amp = smooth.max(axis=1)
    flags = np.zeros(len(signals), dtype=bool)
    for i, signal in enumerate(smooth):
        maxima = np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])) + 1
        maxima = maxima[signal[maxima] > 0.25 * smooth_amp[i]]
        for left_pos, left in enumerate(maxima):
            for right in maxima[left_pos + 1 :]:
                if right - left < 8:
                    continue
                valley = signal[left : right + 1].min()
                if valley < 0.5 * min(signal[left], signal[right]):
                    flags[i] = True
                    break
            if flags[i]:
                break
    return flags


def preprocess_waveforms(sample: Sample, audit: Audit, config: PipelineConfig) -> Processed:
    """Subtract per-pulse baseline, invert sign, align at CFD-50, and create a shape branch."""

    baseline_noisy = np.zeros(len(sample.waveforms), dtype=bool)
    thresholds = {}
    for channel in sorted(DETECTOR_LABELS):
        mask = sample.channels == channel
        center = np.median(audit.baseline_rms[mask])
        spread = 1.4826 * np.median(np.abs(audit.baseline_rms[mask] - center))
        threshold = center + 6 * max(spread, 0.25)
        thresholds[channel] = float(threshold)
        baseline_noisy[mask] = audit.baseline_rms[mask] > threshold

    clipped = (sample.waveforms.min(axis=1) <= config.adc_min) | (
        sample.waveforms.max(axis=1) >= config.adc_max
    )
    low_snr = audit.snr < config.low_snr_threshold
    invalid_alignment = ~np.isfinite(audit.cfd_time) | (
        audit.cfd_time < config.baseline_samples
    )
    tail_not_recovered = audit.tail_ratio > config.tail_ratio_threshold
    possible_multipeak = _conservative_multipeak_flags(audit.positive_raw)

    sample_axis = np.arange(config.expected_samples, dtype=float)
    aligned = np.empty_like(audit.positive_raw, dtype=float)
    for i, signal in enumerate(audit.positive_raw):
        if invalid_alignment[i]:
            aligned[i] = np.nan
            continue
        source_positions = sample_axis + (audit.cfd_time[i] - config.alignment_target)
        aligned[i] = np.interp(source_positions, sample_axis, signal, left=0.0, right=0.0)
    aligned_peak = np.full(len(sample.waveforms), np.nan)
    aligned_peak[~invalid_alignment] = np.max(aligned[~invalid_alignment], axis=1)
    normalized = aligned / np.maximum(aligned_peak[:, None], 1.0)
    quality_ok = ~(
        clipped
        | low_snr
        | baseline_noisy
        | invalid_alignment
        | tail_not_recovered
        | possible_multipeak
    )
    return Processed(
        aligned,
        normalized,
        aligned_peak,
        clipped,
        low_snr,
        baseline_noisy,
        invalid_alignment,
        tail_not_recovered,
        possible_multipeak,
        quality_ok,
        thresholds,
    )


def save_processed_data(
    sample: Sample, audit: Audit, processed: Processed, paths: ProjectPaths, config: PipelineConfig
) -> tuple[Path, Path, Path]:
    amplitude_csv = paths.output_dir / f"processed_aligned_amplitude_seed_{config.random_seed}.csv"
    normalized_csv = paths.output_dir / f"processed_aligned_normalized_seed_{config.random_seed}.csv"
    features_csv = paths.output_dir / f"processed_features_seed_{config.random_seed}.csv"
    np.savetxt(amplitude_csv, processed.aligned_amplitude, fmt="%.6f", delimiter=",")
    np.savetxt(normalized_csv, processed.aligned_normalized, fmt="%.8f", delimiter=",")
    header = [
        "sample_row", "run", "channel", "detector_label", "source_file", "source_row",
        "baseline", "baseline_rms", "baseline_slope", "amplitude", "snr", "peak_index",
        "cfd50_time", "applied_shift", "tail_residual", "tail_ratio", "clipped", "low_snr",
        "baseline_noisy", "invalid_alignment", "tail_not_recovered", "possible_multipeak",
        "quality_ok",
    ]
    with features_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for i, (run, channel, source_file, source_row) in enumerate(sample.provenance):
            writer.writerow([
                i, run, channel, DETECTOR_LABELS[channel], source_file, source_row,
                audit.baseline[i], audit.baseline_rms[i], audit.baseline_slope[i],
                audit.amplitude[i], audit.snr[i], audit.peak_index[i], audit.cfd_time[i],
                audit.cfd_time[i] - config.alignment_target, audit.tail_residual[i],
                audit.tail_ratio[i], int(processed.clipped[i]), int(processed.low_snr[i]),
                int(processed.baseline_noisy[i]), int(processed.invalid_alignment[i]),
                int(processed.tail_not_recovered[i]), int(processed.possible_multipeak[i]),
                int(processed.quality_ok[i]),
            ])
    for path in (amplitude_csv, normalized_csv, features_csv):
        print("Сохранено:", path.relative_to(paths.root))
    return amplitude_csv, normalized_csv, features_csv


def summarize_and_validate(
    sample: Sample,
    audit: Audit,
    processed: Processed,
    paths: ProjectPaths,
    config: PipelineConfig,
) -> Validation:
    """Check preprocessing invariants and save per-group audit statistics."""

    valid = ~processed.invalid_alignment
    aligned_peak_index = np.full(len(sample.waveforms), -1, dtype=int)
    aligned_peak_index[valid] = np.argmax(processed.aligned_amplitude[valid], axis=1)
    aligned_cfd = np.full(len(sample.waveforms), np.nan)
    aligned_cfd[valid] = _leading_crossings(
        processed.aligned_amplitude[valid],
        processed.aligned_peak[valid],
        aligned_peak_index[valid],
        config.cfd_fraction,
    )
    if processed.aligned_amplitude.shape != sample.waveforms.shape:
        raise AssertionError("Размер amplitude-ветви изменился")
    if processed.aligned_normalized.shape != sample.waveforms.shape:
        raise AssertionError("Размер normalized-ветви изменился")
    normalized_error = np.max(
        np.abs(np.max(processed.aligned_normalized[valid], axis=1) - 1.0)
    )
    if normalized_error >= 1e-12:
        raise AssertionError(f"Нарушена нормировка: ошибка {normalized_error}")
    alignment_error_q99 = float(
        np.nanquantile(np.abs(aligned_cfd[valid] - config.alignment_target), 0.99)
    )
    if alignment_error_q99 >= 0.30:
        raise AssertionError(f"Недостаточная точность выравнивания: {alignment_error_q99}")

    rows = []
    for run in RUN_ORDER:
        for channel in sorted(DETECTOR_LABELS):
            mask = (sample.runs == run) & (sample.channels == channel)
            rows.append({
                "run": run,
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "n": int(mask.sum()),
                "baseline_median": float(np.median(audit.baseline[mask])),
                "baseline_rms_median": float(np.median(audit.baseline_rms[mask])),
                "amplitude_q05": float(np.quantile(audit.amplitude[mask], 0.05)),
                "amplitude_median": float(np.median(audit.amplitude[mask])),
                "amplitude_q95": float(np.quantile(audit.amplitude[mask], 0.95)),
                "snr_q05": float(np.quantile(audit.snr[mask], 0.05)),
                "raw_cfd50_q01": float(np.nanquantile(audit.cfd_time[mask], 0.01)),
                "raw_cfd50_median": float(np.nanmedian(audit.cfd_time[mask])),
                "raw_cfd50_q99": float(np.nanquantile(audit.cfd_time[mask], 0.99)),
                "baseline_source_row_corr": float(
                    np.corrcoef(audit.source_rows[mask], audit.baseline[mask])[0, 1]
                ),
                "tail_ratio_q95": float(np.quantile(audit.tail_ratio[mask], 0.95)),
                "low_snr": int(processed.low_snr[mask].sum()),
                "baseline_noisy": int(processed.baseline_noisy[mask].sum()),
                "invalid_alignment": int(processed.invalid_alignment[mask].sum()),
                "tail_not_recovered": int(processed.tail_not_recovered[mask].sum()),
                "possible_multipeak": int(processed.possible_multipeak[mask].sum()),
                "quality_ok": int(processed.quality_ok[mask].sum()),
            })
    summary_csv = paths.output_dir / f"pulse_audit_summary_seed_{config.random_seed}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return Validation(aligned_cfd, tuple(rows), summary_csv, alignment_error_q99)


def print_quality_summary(validation: Validation, paths: ProjectPaths, config: PipelineConfig) -> None:
    print("run/channel | baseline | noise | amplitude q05/50/95 | CFD50 q01/50/99 | flags low/noisy/align/tail/multi | OK")
    for row in validation.summary_rows:
        print(
            f"{row['run']:14s} CH{row['channel']} | {row['baseline_median']:7.1f} | "
            f"{row['baseline_rms_median']:4.2f} | "
            f"{row['amplitude_q05']:6.1f}/{row['amplitude_median']:6.1f}/{row['amplitude_q95']:6.1f} | "
            f"{row['raw_cfd50_q01']:5.2f}/{row['raw_cfd50_median']:5.2f}/{row['raw_cfd50_q99']:5.2f} | "
            f"{row['low_snr']:3d}/{row['baseline_noisy']:2d}/{row['invalid_alignment']:2d}/"
            f"{row['tail_not_recovered']:2d}/{row['possible_multipeak']:2d} | {row['quality_ok']:4d}"
        )
    print(
        f"После выравнивания CFD-{config.cfd_fraction * 100:g}: "
        f"median={np.nanmedian(validation.aligned_cfd):.6f}, "
        f"99% |ошибки|={validation.alignment_error_q99:.6f} sample"
    )
    print("Сводка сохранена:", validation.summary_csv.relative_to(paths.root))


def _trailing_crossings(
    normalized_signals: np.ndarray, peaks: np.ndarray, level: float
) -> np.ndarray:
    result = np.full(len(normalized_signals), np.nan)
    for i, signal in enumerate(normalized_signals):
        start = peaks[i]
        hits = np.flatnonzero(signal[start:] <= level)
        if len(hits) == 0 or hits[0] == 0:
            continue
        j = start + int(hits[0])
        y0, y1 = signal[j - 1], signal[j]
        result[i] = j - 1 + (level - y0) / (y1 - y0) if y1 != y0 else float(j)
    return result


def characterize_processed_shapes(
    sample: Sample, processed: Processed
) -> ShapeCharacteristics:
    """Calculate neutral shape descriptors without assigning particle identity."""

    processed_peak = np.argmax(
        np.nan_to_num(processed.aligned_normalized, nan=-np.inf), axis=1
    )
    decay50_time = _trailing_crossings(processed.aligned_normalized, processed_peak, 0.5)
    positive_for_area = np.clip(processed.aligned_amplitude, 0, None)
    total_area = np.nansum(positive_for_area[:, 15:100], axis=1)
    late_area_fraction = np.nansum(positive_for_area[:, 40:100], axis=1) / np.maximum(
        total_area, 1.0
    )
    rows = []
    for run in RUN_ORDER:
        for channel in sorted(DETECTOR_LABELS):
            mask = (sample.runs == run) & (sample.channels == channel) & processed.quality_ok
            rows.append({
                "run": run,
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "n": int(mask.sum()),
                "peak_sample_median": float(np.median(processed_peak[mask])),
                "decay50_median": float(np.nanmedian(decay50_time[mask])),
                "late_area_fraction_median": float(np.nanmedian(late_area_fraction[mask])),
            })
    return ShapeCharacteristics(processed_peak, decay50_time, late_area_fraction, tuple(rows))


def print_characteristics(characteristics: ShapeCharacteristics) -> None:
    print("Характеристики причёсанных импульсов (медианы по группам):")
    for row in characteristics.rows:
        print(
            f"{row['run']:14s} CH{row['channel']} {row['detector_label']:24s} | "
            f"n={row['n']:4d} | peak sample={row['peak_sample_median']:5.1f} | "
            f"decay50={row['decay50_median']:6.2f} | "
            f"late-area fraction={row['late_area_fraction_median']:.4f}"
        )


def plot_processed_shapes(
    sample: Sample, processed: Processed, paths: ProjectPaths, config: PipelineConfig
) -> list[go.Figure]:
    """Show and save median normalized shapes with 10–90% bands by channel."""

    sample_axis = np.arange(config.expected_samples, dtype=float)
    figures = []
    for channel in sorted(DETECTOR_LABELS):
        fig = go.Figure()
        for run in RUN_ORDER:
            mask = (sample.runs == run) & (sample.channels == channel) & processed.quality_ok
            q10, median_shape, q90 = np.nanquantile(
                processed.aligned_normalized[mask], [0.10, 0.50, 0.90], axis=0
            )
            spec = PLOT_SPECS[run]
            fig.add_trace(go.Scatter(
                x=sample_axis, y=q90, mode="lines", line={"width": 0},
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=sample_axis,
                y=q10,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=(
                    "rgba(31,119,180,0.18)"
                    if run == "call_all_60Co"
                    else "rgba(255,127,14,0.18)"
                ),
                name=f"{spec['title']} 10–90%",
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=sample_axis,
                y=median_shape,
                mode="lines",
                line={"color": spec["color"], "width": 2},
                name=f"{spec['title']} median",
            ))
        fig.update_layout(
            title=f"Processed normalized shapes — CH{channel}: {DETECTOR_LABELS[channel]}",
            xaxis_title="Aligned sample index",
            yaxis_title="Peak-normalized amplitude",
            template="plotly_white",
            height=550,
            dragmode="zoom",
        )
        html_path = paths.output_dir / (
            f"processed_shape_summary_CH{channel}_seed_{config.random_seed}.html"
        )
        fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
        fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
        print("Интерактивно:", html_path.relative_to(paths.root))
        figures.append(fig)
    return figures
