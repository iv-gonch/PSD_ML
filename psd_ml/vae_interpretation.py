"""Interpret channel-specific VAE principal directions using measured properties."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .pipeline import DETECTOR_LABELS, PLOTLY_CONFIG
from .vae import CF_RUN, CO_RUN, LatentAudit, RealVAEConfig, RealVAEData, VAEEnsemble


METRIC_SPECS = (
    ("log10_Qlong", "log10 Qlong", "energy/scale"),
    ("amplitude", "Amplitude", "energy/scale"),
    ("log10_snr", "log10 SNR", "energy/scale"),
    ("classical_psd", "Classical PSD", "pulse shape"),
    ("tail_fraction_40_100", "Tail fraction 40:100", "pulse shape"),
    ("prompt_fraction_15_30", "Prompt fraction 15:30", "pulse shape"),
    ("mid_fraction_30_50", "Middle fraction 30:50", "pulse shape"),
    ("very_late_fraction_60_100", "Very-late fraction 60:100", "pulse shape"),
    ("charge_centroid_15_100", "Charge centroid 15:100", "pulse shape"),
    ("rise_10_90_samples", "Rise 10–90%, samples", "pulse shape"),
    ("decay_50_samples", "Peak-to-50% decay, samples", "pulse shape"),
    ("decay_10_samples", "Peak-to-10% decay, samples", "pulse shape"),
    ("baseline_rms", "Baseline RMS", "technical"),
    ("baseline_slope", "Baseline slope", "technical"),
    ("raw_cfd_shift", "Raw CFD-50 shift", "technical"),
    ("tail_residual_ratio", "Tail residual / amplitude", "technical"),
    ("raw_peak_index", "Raw peak index", "technical"),
    ("reconstruction_mse", "Reconstruction MSE", "model diagnostic"),
)


@dataclass(frozen=True)
class PrincipalDirectionAudit:
    bases: dict[int, dict[str, np.ndarray]]
    metrics: dict[str, np.ndarray]
    summary_rows: tuple[dict[str, object], ...]
    association_rows: tuple[dict[str, object], ...]
    contrast_rows: tuple[dict[str, object], ...]
    output_dir: Path


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        tied = inverse == group
        ranks[tied] = np.mean(ranks[tied])
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5:
        return float("nan")
    rx, ry = _rankdata(x[finite]), _rankdata(y[finite])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _qlong_adjusted_spearman(
    x: np.ndarray,
    y: np.ndarray,
    qlong: np.ndarray,
) -> float:
    """Partial rank correlation after quadratic regression on the Qlong rank."""

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(qlong) & (qlong > 0)
    if finite.sum() < 10:
        return float("nan")
    rx, ry, rq = (_rankdata(values[finite]) for values in (x, y, qlong))
    design = np.column_stack([np.ones(len(rq)), rq, rq**2])
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(residual_x) < 1e-10 or np.std(residual_y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _crossing_time(pulses: np.ndarray, level: float, rising: bool) -> np.ndarray:
    result = np.full(len(pulses), np.nan, dtype=float)
    for row, pulse in enumerate(pulses):
        if not np.any(np.isfinite(pulse)):
            continue
        peak = int(np.nanargmax(pulse))
        segment = pulse[: peak + 1] if rising else pulse[peak:]
        hits = np.flatnonzero(segment >= level if rising else segment <= level)
        if not len(hits):
            continue
        right = int(hits[0]) if rising else peak + int(hits[0])
        if right == 0 or (not rising and right == peak):
            result[row] = float(right)
            continue
        left = right - 1
        y0, y1 = pulse[left], pulse[right]
        result[row] = (
            left + (level - y0) / (y1 - y0) if y1 != y0 else float(right)
        )
    return result


def _pulse_shape_metrics(
    data: RealVAEData,
    latent_audit: LatentAudit,
    primary_seed: int,
) -> dict[str, np.ndarray]:
    pulses = np.clip(data.processed.aligned_normalized.astype(float), 0, None)
    area = np.sum(pulses[:, 15:100], axis=1)

    def fraction(start: int, stop: int) -> np.ndarray:
        return np.divide(
            np.sum(pulses[:, start:stop], axis=1),
            area,
            out=np.full(len(area), np.nan),
            where=area > 0,
        )

    sample_axis = np.arange(15, 100, dtype=float)
    centroid = np.divide(
        np.sum(pulses[:, 15:100] * sample_axis[None, :], axis=1),
        area,
        out=np.full(len(area), np.nan),
        where=area > 0,
    )
    rise_10 = _crossing_time(pulses, 0.10, rising=True)
    rise_90 = _crossing_time(pulses, 0.90, rising=True)
    decay_50 = _crossing_time(pulses, 0.50, rising=False)
    decay_10 = _crossing_time(pulses, 0.10, rising=False)
    peak = np.argmax(pulses, axis=1).astype(float)
    reconstruction_mse = np.full(len(pulses), np.nan)
    for channel in sorted(DETECTOR_LABELS):
        channel_indices = np.flatnonzero(data.sample.channels == channel)
        reconstruction_mse[channel_indices] = latent_audit.encodings[
            (channel, primary_seed)
        ]["reconstruction_mse"]
    return {
        "log10_Qlong": np.log10(np.maximum(data.root.energy.astype(float), 1.0)),
        "amplitude": data.audit.amplitude.astype(float),
        "log10_snr": np.log10(np.maximum(data.audit.snr.astype(float), 1e-6)),
        "classical_psd": data.classical_psd.astype(float),
        "tail_fraction_40_100": fraction(40, 100),
        "prompt_fraction_15_30": fraction(15, 30),
        "mid_fraction_30_50": fraction(30, 50),
        "very_late_fraction_60_100": fraction(60, 100),
        "charge_centroid_15_100": centroid,
        "rise_10_90_samples": rise_90 - rise_10,
        "decay_50_samples": decay_50 - peak,
        "decay_10_samples": decay_10 - peak,
        "baseline_rms": data.audit.baseline_rms.astype(float),
        "baseline_slope": data.audit.baseline_slope.astype(float),
        "raw_cfd_shift": data.audit.cfd_time.astype(float) - 20.0,
        "tail_residual_ratio": data.audit.tail_ratio.astype(float),
        "raw_peak_index": data.audit.peak_index.astype(float),
        "reconstruction_mse": reconstruction_mse,
    }


def _rank_auc(control: np.ndarray, mixture: np.ndarray) -> float:
    values = np.concatenate([control, mixture])
    labels = np.concatenate([np.zeros(len(control)), np.ones(len(mixture))])
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    ranks = _rankdata(values)
    n1, n0 = int(labels.sum()), int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1.0 - auc))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_vae_principal_directions(
    data: RealVAEData,
    ensemble: VAEEnsemble,
    latent_audit: LatentAudit,
    config: RealVAEConfig,
) -> PrincipalDirectionAudit:
    """Fit latent PCs on Cf-validation and audit their measured associations."""

    del ensemble  # Kept in the public signature to mirror the experiment stages.
    primary_seed = config.model_seeds[0]
    metrics = _pulse_shape_metrics(data, latent_audit, primary_seed)
    specs = {name: (label, family) for name, label, family in METRIC_SPECS}
    bases: dict[int, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    basis_json: dict[str, object] = {}

    for channel in sorted(DETECTOR_LABELS):
        channel_indices = np.flatnonzero(data.sample.channels == channel)
        encoded = latent_audit.encodings[(channel, primary_seed)]
        validation_local = np.flatnonzero(data.split[channel_indices] == "validation")
        validation_global = channel_indices[validation_local]
        validation_mu = encoded["mu"][validation_local].astype(float)
        center = np.mean(validation_mu, axis=0)
        eigenvalues, components = np.linalg.eigh(np.cov(validation_mu, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, components = eigenvalues[order], components[:, order]

        initial_scores = (validation_mu - center) @ components
        tail = metrics["tail_fraction_40_100"][validation_global]
        sign_references: list[str] = []
        for direction in range(config.latent_dim):
            correlation = _spearman(initial_scores[:, direction], tail)
            if np.isfinite(correlation) and abs(correlation) >= 0.05:
                if correlation < 0:
                    components[:, direction] *= -1
                sign_references.append("positive tail_fraction_40_100")
            else:
                loading = int(np.argmax(np.abs(components[:, direction])))
                if components[loading, direction] < 0:
                    components[:, direction] *= -1
                sign_references.append(f"positive largest loading z{loading + 1}")

        scores = (encoded["mu"].astype(float) - center) @ components
        explained = eigenvalues / np.sum(eigenvalues)
        effective_dimension = float(
            np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2)
        )
        bases[channel] = {
            "center": center,
            "components": components,
            "eigenvalues": eigenvalues,
            "explained_fraction": explained,
            "scores": scores,
            "validation_local": validation_local,
            "validation_global": validation_global,
        }
        basis_json[str(channel)] = {
            "seed": primary_seed,
            "center": center.tolist(),
            "components_columns_are_pc_directions": components.tolist(),
            "eigenvalues": eigenvalues.tolist(),
            "explained_fraction": explained.tolist(),
            "effective_dimension": effective_dimension,
            "sign_references": sign_references,
        }

        structural = data.structural_ok[channel_indices]
        co_mask = (data.sample.runs[channel_indices] == CO_RUN) & structural
        cf_mask = (data.sample.runs[channel_indices] == CF_RUN) & structural
        qlong_validation = data.root.energy[validation_global].astype(float)
        for direction in range(config.latent_dim):
            pc_name = f"PC{direction + 1}"
            validation_score = scores[validation_local, direction]
            summary_rows.append({
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "seed": primary_seed,
                "direction": pc_name,
                "eigenvalue": float(eigenvalues[direction]),
                "explained_fraction": float(explained[direction]),
                "cumulative_fraction": float(np.sum(explained[: direction + 1])),
                "effective_dimension": effective_dimension,
                "active_ge_1pct": int(explained[direction] >= 0.01),
                "loading_z1": float(components[0, direction]),
                "loading_z2": float(components[1, direction]),
                "loading_z3": float(components[2, direction]),
                "sign_reference": sign_references[direction],
                "co_cf_run_auc": _rank_auc(
                    scores[co_mask, direction], scores[cf_mask, direction]
                ),
                "n_cf_validation": len(validation_score),
            })
            low_edge, high_edge = np.quantile(validation_score, [0.10, 0.90])
            low, high = validation_score <= low_edge, validation_score >= high_edge
            for metric_name, metric_values in metrics.items():
                label, family = specs[metric_name]
                validation_metric = metric_values[validation_global]
                raw = _spearman(validation_score, validation_metric)
                adjusted = (
                    float("nan")
                    if metric_name == "log10_Qlong"
                    else _qlong_adjusted_spearman(
                        validation_score, validation_metric, qlong_validation
                    )
                )
                association_rows.append({
                    "channel": channel,
                    "detector_label": DETECTOR_LABELS[channel],
                    "seed": primary_seed,
                    "direction": pc_name,
                    "metric": metric_name,
                    "metric_label": label,
                    "family": family,
                    "spearman": raw,
                    "spearman_qlong_adjusted": adjusted,
                    "n": int(np.sum(
                        np.isfinite(validation_score) & np.isfinite(validation_metric)
                    )),
                })
                finite_metric = validation_metric[np.isfinite(validation_metric)]
                iqr = (
                    float(np.subtract(*np.quantile(finite_metric, [0.75, 0.25])))
                    if len(finite_metric) else float("nan")
                )
                low_median, high_median = (
                    float(np.nanmedian(validation_metric[selected]))
                    for selected in (low, high)
                )
                contrast_rows.append({
                    "channel": channel,
                    "direction": pc_name,
                    "metric": metric_name,
                    "low_decile_median": low_median,
                    "high_decile_median": high_median,
                    "high_minus_low": high_median - low_median,
                    "difference_over_iqr": (
                        (high_median - low_median) / iqr
                        if np.isfinite(iqr) and iqr > 0 else float("nan")
                    ),
                    "n_low": int(np.sum(low)),
                    "n_high": int(np.sum(high)),
                })

        score_rows = []
        for local, global_index in enumerate(channel_indices):
            score_rows.append({
                "sample_row": int(global_index),
                "run": str(data.sample.runs[global_index]),
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "source_file": data.sample.provenance[global_index][2],
                "source_row": int(data.sample.provenance[global_index][3]),
                "split": str(data.split[global_index]),
                "structural_ok": int(data.structural_ok[global_index]),
                "PC1": float(scores[local, 0]),
                "PC2": float(scores[local, 1]),
                "PC3": float(scores[local, 2]),
            })
        _write_rows(
            latent_audit.output_dir / f"principal_scores_CH{channel}.csv", score_rows
        )

    _write_rows(latent_audit.output_dir / "principal_direction_summary.csv", summary_rows)
    _write_rows(
        latent_audit.output_dir / "principal_direction_associations.csv",
        association_rows,
    )
    _write_rows(
        latent_audit.output_dir / "principal_direction_decile_contrasts.csv",
        contrast_rows,
    )
    (latent_audit.output_dir / "principal_direction_bases.json").write_text(
        json.dumps(basis_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return PrincipalDirectionAudit(
        bases,
        metrics,
        tuple(summary_rows),
        tuple(association_rows),
        tuple(contrast_rows),
        latent_audit.output_dir,
    )


def _primary_model(
    ensemble: VAEEnsemble,
    channel: int,
    seed: int,
) -> torch.nn.Module:
    return next(
        run.model for run in ensemble.runs
        if run.channel == channel and run.seed == seed
    )


def _save_show(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
    fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)


def plot_vae_principal_direction_results(
    data: RealVAEData,
    ensemble: VAEEnsemble,
    direction_audit: PrincipalDirectionAudit,
    config: RealVAEConfig,
) -> list[go.Figure]:
    """Plot property heatmaps and real/decoded movement along each latent PC."""

    figures: list[go.Figure] = []
    metric_names = [name for name, _, _ in METRIC_SPECS]
    metric_labels = [label for _, label, _ in METRIC_SPECS]
    traversal_quantiles = (0.05, 0.25, 0.50, 0.75, 0.95)
    quantile_colors = ("#313695", "#74add1", "#777777", "#f46d43", "#a50026")
    for channel in sorted(DETECTOR_LABELS):
        basis = direction_audit.bases[channel]
        explained = basis["explained_fraction"]
        pc_labels = [
            (
                f"PC{i + 1} ({100 * explained[i]:.1f}%)"
                if explained[i] >= 0.01
                else f"PC{i + 1} ({100 * explained[i]:.1f}%, collapsed)"
            )
            for i in range(3)
        ]
        rows = [
            row for row in direction_audit.association_rows
            if row["channel"] == channel
        ]

        def matrix(field: str) -> np.ndarray:
            values = np.array([
                [
                    next(
                        row[field] for row in rows
                        if row["direction"] == f"PC{pc + 1}"
                        and row["metric"] == metric
                    )
                    for pc in range(3)
                ]
                for metric in metric_names
            ], dtype=float)
            values[:, explained < 0.01] = np.nan
            return values

        heatmap = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Raw Spearman ρ", "Spearman ρ after Qlong adjustment"),
            horizontal_spacing=0.16,
        )
        for column, values in enumerate(
            (matrix("spearman"), matrix("spearman_qlong_adjusted")), start=1
        ):
            text = np.empty(values.shape, dtype=object)
            text[:] = "—"
            text[np.isfinite(values)] = [
                f"{value:.2f}" for value in values[np.isfinite(values)]
            ]
            heatmap.add_trace(go.Heatmap(
                z=values,
                x=pc_labels,
                y=metric_labels,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}<br>ρ=%{z:.3f}<extra></extra>",
                colorbar={"title": "ρ", "len": 0.80} if column == 2 else None,
                showscale=column == 2,
            ), row=1, col=column)
        heatmap.update_layout(
            title=(
                f"Properties associated with latent principal directions — "
                f"CH{channel}: {DETECTOR_LABELS[channel]}"
            ),
            template="plotly_white",
            height=760,
            margin={"l": 180, "r": 90, "t": 100, "b": 80},
        )
        heatmap.update_yaxes(showticklabels=False, row=1, col=2)
        _save_show(
            heatmap,
            direction_audit.output_dir
            / f"principal_direction_associations_CH{channel}.html",
        )
        figures.append(heatmap)

        waveform_fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=tuple(
                [f"{label}<br>Real Cf extremes" for label in pc_labels]
                + [f"{label}<br>Decoder traversal" for label in pc_labels]
            ),
            shared_yaxes=True,
            vertical_spacing=0.16,
        )
        validation_local = basis["validation_local"].astype(int)
        validation_global = basis["validation_global"].astype(int)
        validation_scores = basis["scores"][validation_local]
        model = _primary_model(ensemble, channel, config.model_seeds[0])
        for direction in range(3):
            if explained[direction] < 0.01:
                subplot_x = (0.14444444444444446, 0.5, 0.8555555555555556)
                for subplot_y in (0.79, 0.21):
                    waveform_fig.add_annotation(
                        text="Collapsed; not interpreted",
                        x=subplot_x[direction],
                        y=subplot_y,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                    )
                continue
            values = validation_scores[:, direction]
            low_edge, high_edge = np.quantile(values, [0.10, 0.90])
            for selected, name, color in (
                (values <= low_edge, "Low 10% real median", "#2166ac"),
                (values >= high_edge, "High 10% real median", "#b2182b"),
            ):
                waveform_fig.add_trace(go.Scatter(
                    x=np.arange(144),
                    y=np.median(
                        data.processed.aligned_normalized[
                            validation_global[selected]
                        ],
                        axis=0,
                    ),
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=direction == 0,
                    line={"color": color},
                    hovertemplate="sample=%{x}<br>amplitude=%{y:.4f}<extra></extra>",
                ), row=1, col=direction + 1)

            score_quantiles = np.quantile(values, traversal_quantiles)
            latent_vectors = (
                basis["center"][None, :]
                + score_quantiles[:, None]
                * basis["components"][:, direction][None, :]
            )
            with torch.no_grad():
                decoded = model.decode(
                    torch.from_numpy(latent_vectors.astype(np.float32))
                ).numpy()
            for quantile, score, pulse, color in zip(
                traversal_quantiles, score_quantiles, decoded, quantile_colors
            ):
                name = f"Decoder q={quantile:.0%}"
                waveform_fig.add_trace(go.Scatter(
                    x=np.arange(144),
                    y=pulse,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=direction == 0,
                    line={"color": color},
                    hovertemplate=(
                        f"PC{direction + 1}={score:.3f}<br>"
                        "sample=%{x}<br>decoded=%{y:.4f}<extra></extra>"
                    ),
                ), row=2, col=direction + 1)
        waveform_fig.update_layout(
            title=(
                f"Waveform change along latent principal directions — "
                f"CH{channel}: {DETECTOR_LABELS[channel]}"
            ),
            template="plotly_white",
            height=820,
            margin={"l": 80, "r": 205, "t": 110, "b": 75},
            legend={"orientation": "v", "x": 1.01, "y": 1.0},
        )
        waveform_fig.update_xaxes(title_text="Aligned sample index")
        waveform_fig.update_yaxes(
            title_text="Real peak-normalized amplitude", row=1, col=1
        )
        waveform_fig.update_yaxes(title_text="Decoded amplitude", row=2, col=1)
        _save_show(
            waveform_fig,
            direction_audit.output_dir
            / f"principal_direction_waveforms_CH{channel}.html",
        )
        figures.append(waveform_fig)
    return figures


def print_vae_principal_direction_findings(
    direction_audit: PrincipalDirectionAudit,
) -> None:
    """Print strongest associations without assigning causal or particle meaning."""

    print(
        "PC signs are conventional. Associations use structural-ok Cf-validation; "
        "adjusted ρ is a partial rank association controlling Qlong."
    )
    for channel in sorted(DETECTOR_LABELS):
        print(f"CH{channel} {DETECTOR_LABELS[channel]}:")
        summaries = [
            row for row in direction_audit.summary_rows
            if row["channel"] == channel
        ]
        for summary in summaries:
            direction = summary["direction"]
            if not summary["active_ge_1pct"]:
                print(
                    f"  {direction}: "
                    f"{100 * summary['explained_fraction']:.1f}% "
                    "(inactive/collapsed); associations are not interpreted"
                )
                continue
            rows = [
                row for row in direction_audit.association_rows
                if row["channel"] == channel and row["direction"] == direction
            ]
            qlong = next(row for row in rows if row["metric"] == "log10_Qlong")
            shape = max(
                (
                    row for row in rows
                    if row["family"] == "pulse shape"
                    and np.isfinite(row["spearman_qlong_adjusted"])
                ),
                key=lambda row: abs(row["spearman_qlong_adjusted"]),
            )
            technical = max(
                (
                    row for row in rows
                    if row["family"] == "technical"
                    and np.isfinite(row["spearman_qlong_adjusted"])
                ),
                key=lambda row: abs(row["spearman_qlong_adjusted"]),
            )
            print(
                f"  {direction}: {100 * summary['explained_fraction']:.1f}% (active); "
                f"Qlong ρ={qlong['spearman']:+.3f}; "
                f"shape={shape['metric_label']} "
                f"(adjusted ρ={shape['spearman_qlong_adjusted']:+.3f}); "
                f"technical={technical['metric_label']} "
                f"(adjusted ρ={technical['spearman_qlong_adjusted']:+.3f}); "
                f"run AUC={summary['co_cf_run_auc']:.3f}"
            )
    print("These are associations, not identified physical causes or particle labels.")
