"""Reproducible data pipeline used by ``csv_data_processing.ipynb``.

The public functions deliberately correspond to the notebook's conceptual stages:
inventory, stratified sampling, raw audit, preprocessing, validation, and plotting.
The notebook exposes parameters and results; implementation details live here.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


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

QC_FLAG_SPECS = (
    ("clipped", "достигнута граница ADC", "#d62728"),
    ("invalid_alignment", "невалидное CFD-выравнивание", "#9467bd"),
    ("possible_multipeak", "возможная многопиковая форма", "#8c564b"),
    ("tail_not_recovered", "хвост не вернулся к baseline", "#ff7f0e"),
    ("low_snr", "низкий SNR", "#7f7f7f"),
    ("baseline_noisy", "шумный baseline", "#17becf"),
)


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
class EnergyShapeConfig:
    run: str = "call_all_252Cf"
    pulses_per_channel: int = 5000
    energy_bins: int = 12
    integration_start: int = 15
    tail_start: int = 40
    integration_stop: int = 100
    amplitude_density_bins: int = 72
    bic_delta_threshold: float = 10.0
    separation_threshold: float = 2.0
    min_component_events: int = 8
    bootstrap_iterations: int = 40
    bootstrap_support_threshold: float = 0.70
    stable_min_bins: int = 3
    stable_min_consecutive_bins: int = 3
    stable_min_shape_cosine: float = 0.80
    sensitivity_energy_bins: tuple[int, ...] = (8, 10, 12)
    sensitivity_bootstrap_iterations: int = 40


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
class RootEventMetadata:
    energy: np.ndarray
    energy_short: np.ndarray
    waveform_matches: np.ndarray


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


@dataclass(frozen=True)
class EnergyShapeAnalysis:
    energy: np.ndarray
    classical_psd: np.ndarray
    shape_score: np.ndarray
    shape_usable: np.ndarray
    energy_bin: np.ndarray
    mixture_component: np.ndarray
    bin_edges: dict[int, np.ndarray]
    bin_rows: tuple[dict[str, object], ...]
    channel_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class EnergyBinningSensitivity:
    rows: tuple[dict[str, object], ...]
    channel_rows: tuple[dict[str, object], ...]


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


def _sample_inventory_items(
    items: Sequence[InventoryItem],
    config: PipelineConfig,
    pulses_per_group: int,
) -> Sample:
    rng = np.random.default_rng(config.random_seed)
    rows_by_file: dict[Path, list[int]] = {}
    for item in items:
        if item.rows < pulses_per_group:
            raise ValueError(
                f"В {item.path.name} недостаточно импульсов: "
                f"{item.rows} < {pulses_per_group}"
            )
        rows_by_file[item.path] = np.sort(
            rng.choice(item.rows, size=pulses_per_group, replace=False)
        ).tolist()

    waveforms: list[np.ndarray] = []
    provenance: list[tuple[str, int, str, int]] = []
    for item in items:
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
    expected_shape = (len(items) * pulses_per_group, config.expected_samples)
    if array.shape != expected_shape:
        raise AssertionError(f"Получена форма {array.shape}, ожидалась {expected_shape}")
    provenance_tuple = tuple(provenance)
    return Sample(
        array,
        provenance_tuple,
        np.array([row[0] for row in provenance_tuple]),
        np.array([row[1] for row in provenance_tuple]),
    )


def sample_waveforms(inventory: Inventory, config: PipelineConfig) -> Sample:
    """Draw the same number of random rows from every source × channel file."""

    return _sample_inventory_items(
        inventory.items, config, config.pulses_per_group
    )


def sample_run_waveforms(
    inventory: Inventory,
    config: PipelineConfig,
    run: str,
    pulses_per_channel: int,
) -> Sample:
    """Draw a reproducible, channel-balanced sample from one acquisition run."""

    items = tuple(item for item in inventory.items if item.path.parent.name == run)
    if not items:
        raise ValueError(f"В inventory не найден запуск {run}")
    return _sample_inventory_items(items, config, pulses_per_channel)


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


def _sample_artifact_tag(config: PipelineConfig) -> str:
    return f"{config.pulses_per_group}_per_group_seed_{config.random_seed}"


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
    for run in sorted(set(sample.runs)):
        for channel in sorted(DETECTOR_LABELS):
            group = (sample.runs == run) & (sample.channels == channel)
            if group.sum() < 2:
                continue
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


def load_root_event_metadata(
    sample: Sample,
    paths: ProjectPaths,
    verify_waveforms: bool = True,
) -> RootEventMetadata:
    """Load ROOT metadata for sampled rows and verify CSV↔ROOT event alignment.

    Provenance row numbers are interpreted as TTree entry numbers. When verification is
    enabled, all 144 samples are compared event-by-event with the waveform already loaded
    from CSV. This makes the Energy join an audited positional join rather than an
    assumption based only on file naming.
    """

    try:
        import ROOT
    except ImportError as error:
        raise RuntimeError("Для чтения Energy требуется PyROOT из окружения проекта") from error

    ROOT.gROOT.SetBatch(True)
    n_events = len(sample.waveforms)
    energy = np.empty(n_events, dtype=np.uint16)
    energy_short = np.empty(n_events, dtype=np.uint16)
    waveform_matches = np.zeros(n_events, dtype=bool)

    file_groups: dict[tuple[str, str], list[int]] = {}
    for sample_row, (run, _, source_file, _) in enumerate(sample.provenance):
        file_groups.setdefault((run, source_file), []).append(sample_row)

    for (run, source_file), sample_rows in file_groups.items():
        root_path = (
            paths.root
            / "gamma_n_data"
            / run
            / "UNFILTERED"
            / Path(source_file).with_suffix(".root")
        )
        root_file = ROOT.TFile.Open(str(root_path), "READ")
        if not root_file or root_file.IsZombie():
            raise OSError(f"Не удалось открыть ROOT-файл: {root_path}")
        tree = root_file.Get("Data")
        if not tree or not tree.InheritsFrom("TTree"):
            root_file.Close()
            raise ValueError(f"В {root_path} не найдено дерево Data")

        tree.SetBranchStatus("*", False)
        for branch in ("Energy", "EnergyShort", "Channel"):
            tree.SetBranchStatus(branch, True)
        if verify_waveforms:
            tree.SetBranchStatus("Samples", True)

        for sample_row in sample_rows:
            _, expected_channel, _, source_row = sample.provenance[sample_row]
            if source_row >= tree.GetEntries():
                root_file.Close()
                raise IndexError(
                    f"ROOT entry {source_row} вне диапазона {root_path.name}"
                )
            if tree.GetEntry(source_row) <= 0:
                root_file.Close()
                raise OSError(f"Не удалось прочитать {root_path.name}, entry {source_row}")
            if int(tree.Channel) != expected_channel:
                root_file.Close()
                raise AssertionError(
                    f"Несовпадение канала: {root_path.name}, entry {source_row}"
                )
            energy[sample_row] = int(tree.Energy)
            energy_short[sample_row] = int(tree.EnergyShort)
            if verify_waveforms:
                root_samples = np.fromiter(
                    (tree.Samples.At(i) for i in range(tree.Samples.GetSize())),
                    dtype=np.int16,
                    count=tree.Samples.GetSize(),
                )
                waveform_matches[sample_row] = np.array_equal(
                    root_samples, sample.waveforms[sample_row]
                )
            else:
                waveform_matches[sample_row] = True
        root_file.Close()

    if not np.all(waveform_matches):
        failed = np.flatnonzero(~waveform_matches)
        raise AssertionError(
            f"CSV↔ROOT waveform mismatch для {len(failed)} событий; "
            f"первые sample_row: {failed[:10].tolist()}"
        )
    return RootEventMetadata(energy, energy_short, waveform_matches)


def save_root_event_metadata(
    sample: Sample,
    metadata: RootEventMetadata,
    paths: ProjectPaths,
    config: PipelineConfig,
) -> Path:
    """Save the audited Energy join next to other generated sample artifacts."""

    output = paths.output_dir / f"root_metadata_{_sample_artifact_tag(config)}.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_row", "run", "channel", "source_file", "source_row",
            "energy", "energy_short", "waveform_matches_root",
        ])
        for sample_row, provenance in enumerate(sample.provenance):
            writer.writerow([
                sample_row,
                *provenance,
                int(metadata.energy[sample_row]),
                int(metadata.energy_short[sample_row]),
                int(metadata.waveform_matches[sample_row]),
            ])
    print("ROOT metadata:", output.relative_to(paths.root))
    print(f"Проверено совпадение CSV↔ROOT: {metadata.waveform_matches.sum():,} событий")
    return output


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
    artifact_tag = _sample_artifact_tag(config)
    amplitude_csv = paths.output_dir / f"processed_aligned_amplitude_{artifact_tag}.csv"
    normalized_csv = paths.output_dir / f"processed_aligned_normalized_{artifact_tag}.csv"
    features_csv = paths.output_dir / f"processed_features_{artifact_tag}.csv"
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
    summary_csv = paths.output_dir / (
        f"pulse_audit_summary_{_sample_artifact_tag(config)}.csv"
    )
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


def _qc_flag_arrays(
    processed: Processed,
) -> tuple[tuple[str, str, str, np.ndarray], ...]:
    return tuple(
        (name, label, color, getattr(processed, name))
        for name, label, color in QC_FLAG_SPECS
    )


def summarize_qc_flags(sample: Sample, processed: Processed) -> tuple[dict[str, object], ...]:
    """Count every QC flag and the union of flagged events per source × channel."""

    flag_arrays = _qc_flag_arrays(processed)
    any_flag = np.logical_or.reduce([values for _, _, _, values in flag_arrays])
    if not np.array_equal(any_flag, ~processed.quality_ok):
        raise AssertionError("quality_ok не совпадает с объединением QC-флагов")

    rows = []
    for run in RUN_ORDER:
        for channel in sorted(DETECTOR_LABELS):
            group = (sample.runs == run) & (sample.channels == channel)
            row: dict[str, object] = {
                "run": run,
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "n_group": int(group.sum()),
                "n_flagged": int((group & any_flag).sum()),
            }
            for name, _, _, values in flag_arrays:
                row[name] = int((group & values).sum())
            rows.append(row)
    return tuple(rows)


def print_qc_flag_summary(rows: tuple[dict[str, object], ...]) -> None:
    """Print a compact accounting table; flag columns can overlap."""

    print(
        "run/channel | flagged | clipped/invalid/multipeak/tail/noisy/low_snr "
        "(один импульс может иметь несколько флагов)"
    )
    for row in rows:
        print(
            f"{row['run']:14s} CH{row['channel']} | "
            f"{row['n_flagged']:3d}/{row['n_group']:4d} | "
            f"{row['clipped']:2d}/{row['invalid_alignment']:2d}/"
            f"{row['possible_multipeak']:2d}/{row['tail_not_recovered']:2d}/"
            f"{row['baseline_noisy']:2d}/{row['low_snr']:3d}"
        )


def plot_qc_flagged_waveforms(
    sample: Sample,
    audit: Audit,
    processed: Processed,
    paths: ProjectPaths,
    config: PipelineConfig,
) -> list[go.Figure]:
    """Plot every flagged event without alignment or amplitude normalization.

    Only the per-event baseline is removed and the negative acquisition polarity is
    inverted. This keeps low amplitude, timing failures, and multi-peak topology visible.
    Each event appears exactly once; the legend and hover show all of its active flags.
    """

    flag_arrays = _qc_flag_arrays(processed)
    any_flag = np.logical_or.reduce([values for _, _, _, values in flag_arrays])
    sample_axis = np.arange(config.expected_samples)
    figures = []

    for run in RUN_ORDER:
        spec = PLOT_SPECS[run]
        for channel in sorted(DETECTOR_LABELS):
            group = (sample.runs == run) & (sample.channels == channel)
            indices = np.flatnonzero(group & any_flag)
            if len(indices) == 0:
                print(f"{run:14s} CH{channel}: QC-событий нет")
                continue

            fig = go.Figure()
            shown_combinations: set[str] = set()
            for index in indices:
                active = [
                    (name, label, color)
                    for name, label, color, values in flag_arrays
                    if values[index]
                ]
                combination = " + ".join(name for name, _, _ in active)
                combination_label = " + ".join(label for _, label, _ in active)
                color = active[0][2]
                source_file = sample.provenance[index][2]
                source_row = sample.provenance[index][3]
                fig.add_trace(go.Scattergl(
                    x=sample_axis,
                    y=audit.positive_raw[index],
                    mode="lines",
                    name=combination_label,
                    legendgroup=combination,
                    showlegend=combination not in shown_combinations,
                    line={"color": color, "width": 1.1},
                    opacity=0.72,
                    hovertemplate=(
                        f"sample_row={index}<br>source_row={source_row}<br>"
                        f"source_file={source_file}<br>flags={combination}<br>"
                        f"amplitude={audit.amplitude[index]:.1f} ADC<br>"
                        f"SNR={audit.snr[index]:.1f}<br>"
                        f"baseline RMS={audit.baseline_rms[index]:.2f} ADC<br>"
                        f"tail ratio={audit.tail_ratio[index]:.4f}<br>"
                        "sample=%{x}<br>baseline-subtracted amplitude=%{y:.1f}"
                        "<extra></extra>"
                    ),
                ))
                shown_combinations.add(combination)

            fig.add_hline(y=0, line={"color": "#666", "width": 1})
            fig.update_layout(
                title=(
                    f"QC-flagged waveforms — {spec['title']} — CH{channel}: "
                    f"{DETECTOR_LABELS[channel]} — {len(indices)} events"
                ),
                xaxis_title="Sample index",
                yaxis_title="Baseline-subtracted, sign-inverted amplitude (ADC)",
                template="plotly_white",
                height=620,
                dragmode="zoom",
                legend_title_text="Active QC flags",
            )
            html_path = paths.output_dir / (
                f"qc_flagged_{spec['slug']}_CH{channel}_"
                f"{_sample_artifact_tag(config)}.html"
            )
            fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
            fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
            print("Интерактивно:", html_path.relative_to(paths.root))
            figures.append(fig)
    return figures


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
            f"processed_shape_summary_CH{channel}_{_sample_artifact_tag(config)}.html"
        )
        fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
        fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
        print("Интерактивно:", html_path.relative_to(paths.root))
        figures.append(fig)
    return figures


def _fit_two_gaussian_mixture(values: np.ndarray) -> dict[str, object]:
    """Fit a deterministic 1D two-Gaussian mixture with several initializations."""

    x = np.asarray(values, dtype=float)
    if len(x) < 10 or not np.all(np.isfinite(x)):
        raise ValueError("Для GMM нужны как минимум 10 конечных значений")
    scale = max(float(np.std(x)), 1e-8)
    variance_floor = max(scale**2 * 1e-6, 1e-12)
    initial_quantiles = ((0.20, 0.80), (0.30, 0.70), (0.10, 0.90), (0.50, 0.95))
    best: dict[str, object] | None = None

    for quantiles in initial_quantiles:
        means = np.quantile(x, quantiles).astype(float)
        sigmas = np.full(2, scale, dtype=float)
        weights = np.full(2, 0.5, dtype=float)
        responsibilities = np.full((len(x), 2), 0.5, dtype=float)

        for _ in range(250):
            log_prob = np.column_stack([
                np.log(max(weights[k], 1e-12))
                - np.log(sigmas[k])
                - 0.5 * ((x - means[k]) / sigmas[k]) ** 2
                for k in range(2)
            ])
            row_max = log_prob.max(axis=1, keepdims=True)
            probability = np.exp(log_prob - row_max)
            responsibilities = probability / probability.sum(axis=1, keepdims=True)
            component_n = responsibilities.sum(axis=0)
            new_weights = component_n / len(x)
            new_means = (responsibilities * x[:, None]).sum(axis=0) / component_n
            new_variances = (
                responsibilities * (x[:, None] - new_means) ** 2
            ).sum(axis=0) / component_n
            new_sigmas = np.sqrt(np.maximum(new_variances, variance_floor))
            change = max(
                np.max(np.abs(new_means - means)),
                np.max(np.abs(new_sigmas - sigmas)),
                np.max(np.abs(new_weights - weights)),
            )
            weights, means, sigmas = new_weights, new_means, new_sigmas
            if change < 1e-9:
                break

        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        sigmas = sigmas[order]
        responsibilities = responsibilities[:, order]
        density = np.column_stack([
            weights[k]
            / (sigmas[k] * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x - means[k]) / sigmas[k]) ** 2)
            for k in range(2)
        ])
        responsibilities = density / np.maximum(
            density.sum(axis=1, keepdims=True), 1e-300
        )
        log_likelihood = float(np.log(np.maximum(density.sum(axis=1), 1e-300)).sum())
        candidate = {
            "weights": weights,
            "means": means,
            "sigmas": sigmas,
            "responsibilities": responsibilities,
            "log_likelihood": log_likelihood,
        }
        if best is None or log_likelihood > float(best["log_likelihood"]):
            best = candidate

    assert best is not None
    mean_one = float(np.mean(x))
    sigma_one = max(float(np.std(x)), 1e-8)
    log_likelihood_one = float(
        (
            -np.log(sigma_one * np.sqrt(2 * np.pi))
            - 0.5 * ((x - mean_one) / sigma_one) ** 2
        ).sum()
    )
    bic_one = 2 * np.log(len(x)) - 2 * log_likelihood_one
    bic_two = 5 * np.log(len(x)) - 2 * float(best["log_likelihood"])
    means = np.asarray(best["means"])
    sigmas = np.asarray(best["sigmas"])
    separation = float(
        abs(means[1] - means[0]) / np.sqrt((sigmas[0] ** 2 + sigmas[1] ** 2) / 2)
    )
    assignments = np.asarray(best["responsibilities"]).argmax(axis=1)
    best.update({
        "bic_delta": float(bic_one - bic_two),
        "separation": separation,
        "assignments": assignments,
        "counts": np.bincount(assignments, minlength=2),
    })
    return best


def _mixture_passes_thresholds(
    mixture: dict[str, object], config: EnergyShapeConfig
) -> bool:
    counts = np.asarray(mixture["counts"])
    return bool(
        float(mixture["bic_delta"]) >= config.bic_delta_threshold
        and float(mixture["separation"]) >= config.separation_threshold
        and int(counts.min()) >= config.min_component_events
    )


def _maximum_consecutive_true(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def analyze_energy_binned_shapes(
    sample: Sample,
    metadata: RootEventMetadata,
    audit: Audit,
    processed: Processed,
    config: EnergyShapeConfig,
    random_seed: int,
) -> EnergyShapeAnalysis:
    """Search for two repeatable shape branches within narrow Cf Energy intervals.

    The score is calculated only from peak-normalized waveforms. Equal-count Energy
    intervals are fitted independently with one- and two-Gaussian models. A bin supports
    two branches only when BIC, component separation, minimum size, and bootstrap
    repeatability all pass fixed thresholds. Low-SNR events remain included; only
    structural QC failures are excluded from the branch fit.
    """

    if not (
        0 <= config.integration_start < config.tail_start < config.integration_stop
        <= sample.waveforms.shape[1]
    ):
        raise ValueError("Некорректные границы интегральных окон формы")
    if config.energy_bins < 3:
        raise ValueError("Нужно как минимум три энергетических интервала")

    energy = metadata.energy.astype(float)
    classical_psd = np.full(len(sample.waveforms), np.nan)
    positive_energy = energy > 0
    classical_psd[positive_energy] = (
        energy[positive_energy] - metadata.energy_short[positive_energy]
    ) / energy[positive_energy]

    normalized_positive = np.clip(processed.aligned_normalized, 0, None)
    shape_score = np.full(len(sample.waveforms), np.nan)
    denominator = np.nansum(
        normalized_positive[:, config.integration_start : config.integration_stop],
        axis=1,
    )
    numerator = np.nansum(
        normalized_positive[:, config.tail_start : config.integration_stop], axis=1
    )
    finite_shape = np.isfinite(processed.aligned_normalized).all(axis=1)
    shape_score[finite_shape] = numerator[finite_shape] / np.maximum(
        denominator[finite_shape], 1e-12
    )

    structural_failure = (
        processed.clipped
        | processed.invalid_alignment
        | processed.baseline_noisy
        | processed.tail_not_recovered
        | processed.possible_multipeak
    )
    shape_usable = (
        (sample.runs == config.run)
        & positive_energy
        & finite_shape
        & ~structural_failure
    )
    energy_bin = np.full(len(sample.waveforms), -1, dtype=int)
    mixture_component = np.full(len(sample.waveforms), -1, dtype=int)
    bin_edges: dict[int, np.ndarray] = {}
    bin_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []

    for channel in sorted(DETECTOR_LABELS):
        channel_indices = np.flatnonzero(shape_usable & (sample.channels == channel))
        channel_energy = energy[channel_indices]
        edges = np.unique(
            np.quantile(channel_energy, np.linspace(0, 1, config.energy_bins + 1))
        )
        if len(edges) != config.energy_bins + 1:
            raise ValueError(
                f"CH{channel}: дискретная Energy дала только {len(edges) - 1} "
                f"уникальных интервалов вместо {config.energy_bins}"
            )
        bins = np.searchsorted(edges[1:-1], channel_energy, side="right")
        energy_bin[channel_indices] = bins
        bin_edges[channel] = edges
        channel_differences: list[np.ndarray] = []
        supported_flags: list[bool] = []

        for bin_index in range(config.energy_bins):
            local = bins == bin_index
            indices = channel_indices[local]
            values = shape_score[indices]
            mixture = _fit_two_gaussian_mixture(values)
            assignments = np.asarray(mixture["assignments"], dtype=int)
            mixture_component[indices] = assignments
            base_supported = _mixture_passes_thresholds(mixture, config)

            rng = np.random.default_rng(
                random_seed + channel * 1000 + bin_index
            )
            bootstrap_passes = 0
            for _ in range(config.bootstrap_iterations):
                bootstrap_values = rng.choice(values, size=len(values), replace=True)
                bootstrap_fit = _fit_two_gaussian_mixture(bootstrap_values)
                bootstrap_passes += _mixture_passes_thresholds(
                    bootstrap_fit, config
                )
            bootstrap_support = bootstrap_passes / config.bootstrap_iterations
            supported = bool(
                base_supported
                and bootstrap_support >= config.bootstrap_support_threshold
            )
            supported_flags.append(supported)

            component_counts = np.asarray(mixture["counts"], dtype=int)
            weights = np.asarray(mixture["weights"], dtype=float)
            means = np.asarray(mixture["means"], dtype=float)
            sigmas = np.asarray(mixture["sigmas"], dtype=float)
            if supported:
                low_shape = np.nanmedian(
                    processed.aligned_normalized[indices[assignments == 0]], axis=0
                )
                high_shape = np.nanmedian(
                    processed.aligned_normalized[indices[assignments == 1]], axis=0
                )
                channel_differences.append(
                    high_shape[config.integration_start : config.integration_stop]
                    - low_shape[config.integration_start : config.integration_stop]
                )

            bin_rows.append({
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "energy_bin": bin_index,
                "energy_low": float(edges[bin_index]),
                "energy_high": float(edges[bin_index + 1]),
                "energy_median": float(np.median(energy[indices])),
                "n": int(len(indices)),
                "low_snr": int(processed.low_snr[indices].sum()),
                "component_0_n": int(component_counts[0]),
                "component_1_n": int(component_counts[1]),
                "component_0_weight": float(weights[0]),
                "component_1_weight": float(weights[1]),
                "component_0_mean": float(means[0]),
                "component_1_mean": float(means[1]),
                "component_0_sigma": float(sigmas[0]),
                "component_1_sigma": float(sigmas[1]),
                "bic_delta": float(mixture["bic_delta"]),
                "separation": float(mixture["separation"]),
                "bootstrap_support": float(bootstrap_support),
                "two_branch_supported": supported,
            })

        difference_cosine = np.nan
        if len(channel_differences) == 1:
            difference_cosine = 1.0
        elif len(channel_differences) > 1:
            reference = np.mean(channel_differences, axis=0)
            reference_norm = np.linalg.norm(reference)
            cosines = [
                float(np.dot(difference, reference) / (
                    np.linalg.norm(difference) * reference_norm
                ))
                for difference in channel_differences
                if np.linalg.norm(difference) > 0 and reference_norm > 0
            ]
            if cosines:
                difference_cosine = float(np.median(cosines))

        channel_bin_rows = [
            row for row in bin_rows if row["channel"] == channel
        ]
        supported_rows = [
            row for row in channel_bin_rows if row["two_branch_supported"]
        ]
        max_consecutive = _maximum_consecutive_true(supported_flags)
        stable = bool(
            len(supported_rows) >= config.stable_min_bins
            and max_consecutive >= config.stable_min_consecutive_bins
            and np.isfinite(difference_cosine)
            and difference_cosine >= config.stable_min_shape_cosine
        )
        channel_rows.append({
            "channel": channel,
            "detector_label": DETECTOR_LABELS[channel],
            "n_shape_usable": int(len(channel_indices)),
            "n_low_snr_retained": int(processed.low_snr[channel_indices].sum()),
            "energy_min": float(channel_energy.min()),
            "energy_max": float(channel_energy.max()),
            "amplitude_energy_correlation": float(
                np.corrcoef(audit.amplitude[channel_indices], channel_energy)[0, 1]
            ),
            "shape_score_classical_psd_correlation": float(
                np.corrcoef(shape_score[channel_indices], classical_psd[channel_indices])[0, 1]
            ),
            "supported_bins": int(len(supported_rows)),
            "max_consecutive_supported_bins": int(max_consecutive),
            "median_supported_separation": float(
                np.median([row["separation"] for row in supported_rows])
            ) if supported_rows else np.nan,
            "median_supported_bootstrap": float(
                np.median([row["bootstrap_support"] for row in supported_rows])
            ) if supported_rows else np.nan,
            "median_branch_shape_cosine": float(difference_cosine),
            "stable_two_branch_evidence": stable,
        })

    return EnergyShapeAnalysis(
        energy,
        classical_psd,
        shape_score,
        shape_usable,
        energy_bin,
        mixture_component,
        bin_edges,
        tuple(bin_rows),
        tuple(channel_rows),
    )


def assess_energy_binning_sensitivity(
    sample: Sample,
    metadata: RootEventMetadata,
    audit: Audit,
    processed: Processed,
    primary_analysis: EnergyShapeAnalysis,
    config: EnergyShapeConfig,
    random_seed: int,
) -> EnergyBinningSensitivity:
    """Repeat the branch verdict for neighboring choices of Energy-bin count."""

    if config.energy_bins not in config.sensitivity_energy_bins:
        raise ValueError("Основное число интервалов должно входить в sensitivity_energy_bins")
    rows = []
    for energy_bins in config.sensitivity_energy_bins:
        if energy_bins == config.energy_bins:
            current = primary_analysis
        else:
            sensitivity_config = replace(
                config,
                energy_bins=energy_bins,
                bootstrap_iterations=config.sensitivity_bootstrap_iterations,
            )
            current = analyze_energy_binned_shapes(
                sample,
                metadata,
                audit,
                processed,
                sensitivity_config,
                random_seed,
            )
        for channel_row in current.channel_rows:
            rows.append({
                "energy_bins": energy_bins,
                "channel": channel_row["channel"],
                "detector_label": channel_row["detector_label"],
                "supported_bins": channel_row["supported_bins"],
                "max_consecutive_supported_bins": channel_row[
                    "max_consecutive_supported_bins"
                ],
                "median_branch_shape_cosine": channel_row[
                    "median_branch_shape_cosine"
                ],
                "stable_two_branch_evidence": channel_row[
                    "stable_two_branch_evidence"
                ],
            })

    channel_rows = []
    required = int(np.ceil(len(config.sensitivity_energy_bins) * 2 / 3))
    for channel in sorted(DETECTOR_LABELS):
        channel_sensitivity = [row for row in rows if row["channel"] == channel]
        stable_count = sum(
            bool(row["stable_two_branch_evidence"])
            for row in channel_sensitivity
        )
        channel_rows.append({
            "channel": channel,
            "detector_label": DETECTOR_LABELS[channel],
            "tested_energy_bin_counts": "/".join(
                str(row["energy_bins"]) for row in channel_sensitivity
            ),
            "stable_verdict_count": stable_count,
            "required_stable_verdicts": required,
            "stable_across_binnings": stable_count >= required,
        })
    return EnergyBinningSensitivity(tuple(rows), tuple(channel_rows))


def print_energy_shape_analysis(analysis: EnergyShapeAnalysis) -> None:
    """Print channel-level verdicts and auditable per-bin mixture diagnostics."""

    print("Cf: проверка двух ветвей формы внутри равностатистических Energy-интервалов")
    print(
        "channel | usable (low-SNR kept) | E range | r(amplitude,E) | "
        "r(shape,classic PSD) | supported/consecutive bins | shape cosine | stable"
    )
    for row in analysis.channel_rows:
        cosine = row["median_branch_shape_cosine"]
        cosine_text = f"{cosine:.3f}" if np.isfinite(cosine) else "  nan"
        print(
            f"CH{row['channel']} {row['detector_label']:24s} | "
            f"{row['n_shape_usable']:5d} ({row['n_low_snr_retained']:4d}) | "
            f"{row['energy_min']:6.0f}–{row['energy_max']:<6.0f} | "
            f"{row['amplitude_energy_correlation']:.4f} | "
            f"{row['shape_score_classical_psd_correlation']:.4f} | "
            f"{row['supported_bins']}/{row['max_consecutive_supported_bins']} | "
            f"{cosine_text} | {bool(row['stable_two_branch_evidence'])}"
        )

    print("\nПоканальные интервалы (Energy — приборные, некалиброванные единицы):")
    for row in analysis.bin_rows:
        print(
            f"CH{row['channel']} bin{row['energy_bin']} "
            f"[{row['energy_low']:.0f}, {row['energy_high']:.0f}] n={row['n']:4d} | "
            f"weights={row['component_0_weight']:.3f}/{row['component_1_weight']:.3f} | "
            f"means={row['component_0_mean']:.4f}/{row['component_1_mean']:.4f} | "
            f"ΔBIC={row['bic_delta']:.1f} D={row['separation']:.2f} "
            f"bootstrap={row['bootstrap_support']:.2f} | "
            f"two branches={bool(row['two_branch_supported'])}"
        )


def print_energy_binning_sensitivity(
    sensitivity: EnergyBinningSensitivity,
) -> None:
    """Print whether the channel verdict survives neighboring Energy binnings."""

    print("\nЧувствительность вывода к числу равностатистических Energy-интервалов:")
    for row in sensitivity.rows:
        cosine = row["median_branch_shape_cosine"]
        cosine_text = f"{cosine:.3f}" if np.isfinite(cosine) else "nan"
        print(
            f"CH{row['channel']} bins={row['energy_bins']:2d} | "
            f"supported={row['supported_bins']:2d}, "
            f"consecutive={row['max_consecutive_supported_bins']:2d}, "
            f"shape cosine={cosine_text} | "
            f"stable={bool(row['stable_two_branch_evidence'])}"
        )
    print("Итог по устойчивости к разбиению:")
    for row in sensitivity.channel_rows:
        print(
            f"CH{row['channel']} {row['detector_label']:24s} | "
            f"{row['stable_verdict_count']}/{len(sensitivity.rows) // len(DETECTOR_LABELS)} "
            f"разбиений дали stable | "
            f"stable across binnings={bool(row['stable_across_binnings'])}"
        )


def save_energy_binning_sensitivity(
    sensitivity: EnergyBinningSensitivity,
    paths: ProjectPaths,
    config: EnergyShapeConfig,
    random_seed: int,
) -> tuple[Path, Path]:
    """Save per-binning and aggregate robustness verdicts."""

    tag = f"{config.pulses_per_channel}_per_channel_seed_{random_seed}"
    rows_csv = paths.output_dir / f"cf_energy_binning_sensitivity_{tag}.csv"
    channels_csv = (
        paths.output_dir / f"cf_energy_binning_robustness_channels_{tag}.csv"
    )
    for output, rows in (
        (rows_csv, sensitivity.rows),
        (channels_csv, sensitivity.channel_rows),
    ):
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("Сохранено:", output.relative_to(paths.root))
    return rows_csv, channels_csv


def save_energy_shape_analysis(
    sample: Sample,
    metadata: RootEventMetadata,
    processed: Processed,
    analysis: EnergyShapeAnalysis,
    paths: ProjectPaths,
    config: EnergyShapeConfig,
    random_seed: int,
) -> tuple[Path, Path, Path]:
    """Save event assignments and both levels of branch-evidence summaries."""

    tag = f"{config.pulses_per_channel}_per_channel_seed_{random_seed}"
    events_csv = paths.output_dir / f"cf_energy_shape_events_{tag}.csv"
    bins_csv = paths.output_dir / f"cf_energy_shape_bins_{tag}.csv"
    channels_csv = paths.output_dir / f"cf_energy_shape_channels_{tag}.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_row", "run", "channel", "detector_label", "source_file",
            "source_row", "energy", "energy_short", "classical_psd", "shape_score",
            "shape_usable", "low_snr", "energy_bin", "mixture_component",
        ])
        for sample_row, provenance in enumerate(sample.provenance):
            run, channel, source_file, source_row = provenance
            writer.writerow([
                sample_row, run, channel, DETECTOR_LABELS[channel], source_file,
                source_row, int(metadata.energy[sample_row]),
                int(metadata.energy_short[sample_row]), analysis.classical_psd[sample_row],
                analysis.shape_score[sample_row], int(analysis.shape_usable[sample_row]),
                int(processed.low_snr[sample_row]), int(analysis.energy_bin[sample_row]),
                int(analysis.mixture_component[sample_row]),
            ])
    for output, rows in (
        (bins_csv, analysis.bin_rows),
        (channels_csv, analysis.channel_rows),
    ):
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for output in (events_csv, bins_csv, channels_csv):
        print("Сохранено:", output.relative_to(paths.root))
    return events_csv, bins_csv, channels_csv


def plot_energy_shape_scores(
    sample: Sample,
    processed: Processed,
    analysis: EnergyShapeAnalysis,
    paths: ProjectPaths,
    config: EnergyShapeConfig,
    random_seed: int,
) -> list[go.Figure]:
    """Plot waveform-only shape score versus uncalibrated ROOT Energy by Cf channel."""

    figures = []
    row_lookup = {
        (int(row["channel"]), int(row["energy_bin"])): row
        for row in analysis.bin_rows
    }
    tag = f"{config.pulses_per_channel}_per_channel_seed_{random_seed}"
    for channel in sorted(DETECTOR_LABELS):
        indices = np.flatnonzero(
            analysis.shape_usable & (sample.channels == channel)
        )
        fig = go.Figure()
        custom = np.column_stack([
            indices,
            np.array([sample.provenance[index][3] for index in indices]),
            analysis.energy_bin[indices],
            processed.low_snr[indices].astype(int),
        ])
        fig.add_trace(go.Scattergl(
            x=analysis.energy[indices],
            y=analysis.shape_score[indices],
            mode="markers",
            name="all shape-usable",
            marker={"size": 4, "color": "#7f7f7f", "opacity": 0.25},
            customdata=custom,
            hovertemplate=(
                "Energy=%{x:.0f}<br>shape score=%{y:.4f}<br>"
                "sample_row=%{customdata[0]}<br>source_row=%{customdata[1]}<br>"
                "energy bin=%{customdata[2]}<br>low_snr=%{customdata[3]}"
                "<extra></extra>"
            ),
        ))

        supported_high = []
        for bin_index in range(config.energy_bins):
            row = row_lookup[(channel, bin_index)]
            if not row["two_branch_supported"]:
                continue
            event_mask = (
                analysis.shape_usable
                & (sample.channels == channel)
                & (analysis.energy_bin == bin_index)
                & (analysis.mixture_component == 1)
            )
            supported_high.extend(np.flatnonzero(event_mask).tolist())
            for component, color, label in (
                (0, "#1f77b4", "lower-score component"),
                (1, "#ff7f0e", "higher-score component"),
            ):
                fig.add_trace(go.Scatter(
                    x=[row["energy_low"], row["energy_high"]],
                    y=[row[f"component_{component}_mean"]] * 2,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=bin_index == min(
                        int(item["energy_bin"])
                        for item in analysis.bin_rows
                        if item["channel"] == channel
                        and item["two_branch_supported"]
                    ),
                    line={"color": color, "width": 3},
                    hovertemplate=(
                        f"bin {bin_index}: Energy "
                        f"[{row['energy_low']:.0f}, {row['energy_high']:.0f}]<br>"
                        f"mean={row[f'component_{component}_mean']:.4f}"
                        "<extra></extra>"
                    ),
                ))
        if supported_high:
            high_indices = np.array(supported_high, dtype=int)
            fig.add_trace(go.Scattergl(
                x=analysis.energy[high_indices],
                y=analysis.shape_score[high_indices],
                mode="markers",
                name="higher-score candidates in supported bins",
                marker={"size": 6, "color": "#ff7f0e", "symbol": "diamond"},
                hovertemplate="Energy=%{x:.0f}<br>shape score=%{y:.4f}<extra></extra>",
            ))
        fig.update_layout(
            title=(
                f"Cf waveform shape score vs ROOT Energy — CH{channel}: "
                f"{DETECTOR_LABELS[channel]}"
            ),
            xaxis_title="ROOT Energy (uncalibrated instrument units, log scale)",
            yaxis_title="Normalized waveform late-area score",
            xaxis_type="log",
            template="plotly_white",
            height=620,
            dragmode="zoom",
        )
        html_path = paths.output_dir / f"cf_energy_shape_score_CH{channel}_{tag}.html"
        fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
        fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
        print("Интерактивно:", html_path.relative_to(paths.root))
        figures.append(fig)
    return figures


def plot_energy_binned_shape_distributions(
    sample: Sample,
    processed: Processed,
    analysis: EnergyShapeAnalysis,
    paths: ProjectPaths,
    config: EnergyShapeConfig,
    random_seed: int,
) -> list[go.Figure]:
    """Plot conditional densities of normalized forms in each Cf Energy interval."""

    figures = []
    sample_axis = np.arange(config.integration_start, config.integration_stop)
    amplitude_edges = np.linspace(-0.10, 1.05, config.amplitude_density_bins + 1)
    amplitude_centers = (amplitude_edges[:-1] + amplitude_edges[1:]) / 2
    tag = f"{config.pulses_per_channel}_per_channel_seed_{random_seed}"

    for channel in sorted(DETECTOR_LABELS):
        channel_rows = [
            row for row in analysis.bin_rows if row["channel"] == channel
        ]
        subplot_titles = [
            (
                f"bin {row['energy_bin']}: E {row['energy_low']:.0f}–"
                f"{row['energy_high']:.0f}, n={row['n']}, "
                f"2-branch={'yes' if row['two_branch_supported'] else 'no'}"
            )
            for row in channel_rows
        ]
        subplot_cols = 3 if config.energy_bins > 8 else 2
        subplot_rows = int(np.ceil(config.energy_bins / subplot_cols))
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.065,
            horizontal_spacing=0.06,
        )
        legend_seen: set[str] = set()
        for position, row_data in enumerate(channel_rows):
            row = position // subplot_cols + 1
            col = position % subplot_cols + 1
            bin_index = int(row_data["energy_bin"])
            indices = np.flatnonzero(
                analysis.shape_usable
                & (sample.channels == channel)
                & (analysis.energy_bin == bin_index)
            )
            shapes = processed.aligned_normalized[
                indices, config.integration_start : config.integration_stop
            ]
            density = np.empty(
                (config.amplitude_density_bins, len(sample_axis)), dtype=float
            )
            for sample_position in range(len(sample_axis)):
                counts, _ = np.histogram(
                    shapes[:, sample_position], bins=amplitude_edges
                )
                density[:, sample_position] = counts / len(shapes)
            fig.add_trace(go.Heatmap(
                x=sample_axis,
                y=amplitude_centers,
                z=density,
                coloraxis="coloraxis",
                hovertemplate=(
                    "sample=%{x}<br>normalized amplitude=%{y:.3f}<br>"
                    "fraction/bin=%{z:.3f}<extra></extra>"
                ),
            ), row=row, col=col)

            overall_name = "overall median"
            fig.add_trace(go.Scatter(
                x=sample_axis,
                y=np.nanmedian(shapes, axis=0),
                mode="lines",
                name=overall_name,
                legendgroup=overall_name,
                showlegend=overall_name not in legend_seen,
                line={"color": "#111111", "width": 2},
            ), row=row, col=col)
            legend_seen.add(overall_name)

            if row_data["two_branch_supported"]:
                local_components = analysis.mixture_component[indices]
                for component, color, name in (
                    (0, "#1f77b4", "lower-score median"),
                    (1, "#ff7f0e", "higher-score median"),
                ):
                    component_shapes = shapes[local_components == component]
                    fig.add_trace(go.Scatter(
                        x=sample_axis,
                        y=np.nanmedian(component_shapes, axis=0),
                        mode="lines",
                        name=name,
                        legendgroup=name,
                        showlegend=name not in legend_seen,
                        line={"color": color, "width": 2.5},
                    ), row=row, col=col)
                    legend_seen.add(name)

        fig.update_layout(
            title=(
                f"Cf normalized-form densities by ROOT Energy — CH{channel}: "
                f"{DETECTOR_LABELS[channel]}"
            ),
            template="plotly_white",
            height=300 * subplot_rows + 150,
            dragmode="zoom",
            coloraxis={
                "colorscale": "Viridis",
                "colorbar": {"title": "event fraction / amplitude bin"},
            },
        )
        fig.update_xaxes(title_text="Aligned sample index", row=subplot_rows)
        fig.update_yaxes(title_text="Peak-normalized amplitude", col=1)
        html_path = paths.output_dir / f"cf_energy_binned_forms_CH{channel}_{tag}.html"
        fig.write_html(html_path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
        fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)
        print("Интерактивно:", html_path.relative_to(paths.root))
        figures.append(fig)
    return figures
