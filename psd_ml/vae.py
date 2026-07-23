"""Three-latent VAE experiment on real PSD_ML pulse waveforms.

The public functions in this module are intentionally stage-sized so that
``real_data_vae.ipynb`` remains an orchestration and interpretation document.
Co is never used for fitting: five channel-specific VAEs learn the real Cf
mixture without event labels, and Co is retained as an external run control.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import nn

from .pipeline import (
    DETECTOR_LABELS,
    PLOTLY_CONFIG,
    Audit,
    PipelineConfig,
    Processed,
    ProjectPaths,
    RootEventMetadata,
    Sample,
    audit_waveforms,
    inventory_sources,
    load_root_event_metadata,
    preprocess_waveforms,
    sample_waveforms,
)


CF_RUN = "call_all_252Cf"
CO_RUN = "call_all_60Co"


@dataclass(frozen=True)
class RealVAEConfig:
    data_seed: int = 20260717
    pulses_per_group: int = 10_000
    model_seeds: tuple[int, ...] = (20260717, 20260718, 20260719)
    latent_dim: int = 3
    hidden_dims: tuple[int, int] = (128, 64)
    validation_fraction: float = 0.20
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 200
    patience: int = 20
    beta_max: float = 0.01
    beta_warmup_epochs: int = 40
    tail_start: int = 40
    integration_start: int = 15
    integration_stop: int = 100
    qlong_bins: int = 8
    traversal_quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

    def __post_init__(self) -> None:
        if self.latent_dim != 3:
            raise ValueError("Этот эксперимент фиксирует ровно три латентные переменные")
        if len(self.model_seeds) != 3:
            raise ValueError("Для проверки устойчивости требуется ровно три model seed")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction должна лежать между 0 и 1")


@dataclass(frozen=True)
class RealVAEData:
    sample: Sample
    audit: Audit
    processed: Processed
    root: RootEventMetadata
    structural_ok: np.ndarray
    split: np.ndarray
    classical_psd: np.ndarray
    shape_score: np.ndarray


@dataclass(frozen=True)
class VAETrainingRun:
    channel: int
    seed: int
    model: "PulseVAE"
    history: tuple[dict[str, float], ...]
    best_epoch: int
    checkpoint_path: Path


@dataclass(frozen=True)
class VAEEnsemble:
    runs: tuple[VAETrainingRun, ...]


@dataclass(frozen=True)
class LatentAudit:
    encodings: dict[tuple[int, int], dict[str, np.ndarray]]
    correlation_rows: tuple[dict[str, object], ...]
    stability_rows: tuple[dict[str, object], ...]
    qlong_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    output_dir: Path


class PulseVAE(nn.Module):
    """Compact MLP VAE for 144-point peak-normalized pulses."""

    def __init__(
        self,
        input_dim: int = 144,
        hidden_dims: tuple[int, int] = (128, 64),
        latent_dim: int = 3,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.SiLU(),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.SiLU(),
            nn.Linear(h2, h1),
            nn.SiLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), torch.clamp(self.logvar(hidden), -10.0, 10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = torch.sum((reconstruction - target) ** 2, dim=1).mean()
    kl_loss = -0.5 * torch.sum(
        1.0 + logvar - mu.square() - torch.exp(logvar), dim=1
    ).mean()
    total = reconstruction_loss + beta * kl_loss
    return total, reconstruction_loss, kl_loss


def _structural_ok(processed: Processed) -> np.ndarray:
    return ~(
        processed.clipped
        | processed.baseline_noisy
        | processed.invalid_alignment
        | processed.tail_not_recovered
        | processed.possible_multipeak
    )


def _stratified_cf_split(
    sample: Sample,
    root: RootEventMetadata,
    structural_ok: np.ndarray,
    config: RealVAEConfig,
) -> np.ndarray:
    split = np.full(len(sample.waveforms), "excluded", dtype="U10")
    split[(sample.runs == CO_RUN) & structural_ok] = "control"
    for channel in sorted(DETECTOR_LABELS):
        indices = np.flatnonzero(
            (sample.runs == CF_RUN) & (sample.channels == channel) & structural_ok
        )
        energy = root.energy[indices].astype(float)
        edges = np.unique(np.quantile(energy, np.linspace(0, 1, 11)))
        strata = np.digitize(energy, edges[1:-1], right=True)
        rng = np.random.default_rng(config.data_seed + channel)
        for stratum in np.unique(strata):
            local = indices[strata == stratum].copy()
            rng.shuffle(local)
            n_validation = max(1, int(round(len(local) * config.validation_fraction)))
            split[local[:n_validation]] = "validation"
            split[local[n_validation:]] = "train"
    return split


def prepare_real_vae_data(
    paths: ProjectPaths,
    config: RealVAEConfig,
    verify_root_waveforms: bool = True,
) -> RealVAEData:
    """Sample, preprocess, and attach audited ROOT metadata for the VAE experiment."""

    pipeline_config = PipelineConfig(
        random_seed=config.data_seed,
        pulses_per_group=config.pulses_per_group,
    )
    inventory = inventory_sources(paths, pipeline_config)
    sample = sample_waveforms(inventory, pipeline_config)
    audit = audit_waveforms(sample, pipeline_config)
    processed = preprocess_waveforms(sample, audit, pipeline_config)
    root = load_root_event_metadata(
        sample, paths, verify_waveforms=verify_root_waveforms
    )
    if not np.all(root.waveform_matches):
        raise AssertionError("Не все sampled CSV waveform совпали с ROOT")
    structural_ok = _structural_ok(processed)
    split = _stratified_cf_split(sample, root, structural_ok, config)
    energy = root.energy.astype(float)
    classical_psd = np.divide(
        energy - root.energy_short.astype(float),
        energy,
        out=np.full(len(energy), np.nan),
        where=energy > 0,
    )
    positive = np.clip(processed.aligned_normalized, 0, None)
    denominator = np.sum(
        positive[:, config.integration_start : config.integration_stop], axis=1
    )
    shape_score = np.divide(
        np.sum(positive[:, config.tail_start : config.integration_stop], axis=1),
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 0,
    )
    data = RealVAEData(
        sample,
        audit,
        processed,
        root,
        structural_ok,
        split,
        classical_psd,
        shape_score,
    )
    _validate_real_vae_data(data, config)
    return data


def _validate_real_vae_data(data: RealVAEData, config: RealVAEConfig) -> None:
    if data.processed.aligned_normalized.shape != (
        10 * config.pulses_per_group,
        144,
    ):
        raise AssertionError("Неожиданная форма VAE dataset")
    if np.any((data.split == "train") & (data.sample.runs != CF_RUN)):
        raise AssertionError("Co обнаружен в train split")
    if np.any((data.split == "validation") & (data.sample.runs != CF_RUN)):
        raise AssertionError("Co обнаружен в validation split")
    if np.any(data.processed.invalid_alignment[data.structural_ok]):
        raise AssertionError("structural_ok содержит невалидное выравнивание")
    for channel in sorted(DETECTOR_LABELS):
        if not np.any((data.sample.channels == channel) & (data.split == "train")):
            raise AssertionError(f"CH{channel}: пустой train split")
        if not np.any((data.sample.channels == channel) & (data.split == "validation")):
            raise AssertionError(f"CH{channel}: пустой validation split")


def print_real_vae_data_summary(data: RealVAEData) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    print("VAE учится только на Cf; Co используется только как внешний run-control.")
    for channel in sorted(DETECTOR_LABELS):
        row: dict[str, object] = {
            "channel": channel,
            "detector": DETECTOR_LABELS[channel],
        }
        for split_name in ("train", "validation", "control", "excluded"):
            row[split_name] = int(
                np.sum((data.sample.channels == channel) & (data.split == split_name))
            )
        row["low_snr_retained"] = int(
            np.sum(
                (data.sample.channels == channel)
                & data.structural_ok
                & data.processed.low_snr
            )
        )
        rows.append(row)
        print(
            f"CH{channel} {DETECTOR_LABELS[channel]}: "
            f"train={row['train']}, validation={row['validation']}, "
            f"Co-control={row['control']}, excluded={row['excluded']}, "
            f"low-SNR retained={row['low_snr_retained']}"
        )
    return tuple(rows)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _evaluate_loss(
    model: PulseVAE,
    array: np.ndarray,
    config: RealVAEConfig,
    beta: float,
) -> tuple[float, float, float]:
    model.eval()
    totals = np.zeros(3, dtype=float)
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(array), config.batch_size):
            batch = torch.from_numpy(array[start : start + config.batch_size])
            mu, logvar = model.encode(batch)
            reconstruction = model.decode(mu)
            values = vae_loss(reconstruction, batch, mu, logvar, beta)
            totals += [float(value) for value in values]
            n_batches += 1
    return tuple((totals / max(n_batches, 1)).tolist())


def _save_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_channel_vae_ensemble(
    data: RealVAEData,
    paths: ProjectPaths,
    config: RealVAEConfig,
) -> VAEEnsemble:
    """Train three deterministic initializations for every detector channel."""

    output_dir = paths.output_dir / "vae_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runs: list[VAETrainingRun] = []
    for channel in sorted(DETECTOR_LABELS):
        train_indices = np.flatnonzero(
            (data.sample.channels == channel) & (data.split == "train")
        )
        validation_indices = np.flatnonzero(
            (data.sample.channels == channel) & (data.split == "validation")
        )
        train_array = np.ascontiguousarray(
            data.processed.aligned_normalized[train_indices], dtype=np.float32
        )
        validation_array = np.ascontiguousarray(
            data.processed.aligned_normalized[validation_indices], dtype=np.float32
        )
        for seed in config.model_seeds:
            _set_deterministic_seed(seed)
            model = PulseVAE(144, config.hidden_dims, config.latent_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            generator = torch.Generator().manual_seed(seed)
            history: list[dict[str, float]] = []
            best_state: dict[str, torch.Tensor] | None = None
            best_epoch = 0
            best_validation = float("inf")
            stale_epochs = 0
            for epoch in range(1, config.max_epochs + 1):
                model.train()
                permutation = torch.randperm(len(train_array), generator=generator).numpy()
                beta = config.beta_max * min(1.0, epoch / config.beta_warmup_epochs)
                for start in range(0, len(permutation), config.batch_size):
                    batch_indices = permutation[start : start + config.batch_size]
                    batch = torch.from_numpy(train_array[batch_indices])
                    optimizer.zero_grad(set_to_none=True)
                    reconstruction, mu, logvar = model(batch)
                    total, _, _ = vae_loss(reconstruction, batch, mu, logvar, beta)
                    total.backward()
                    optimizer.step()
                train_values = _evaluate_loss(model, train_array, config, beta)
                validation_values = _evaluate_loss(
                    model, validation_array, config, beta
                )
                history.append({
                    "epoch": float(epoch),
                    "beta": float(beta),
                    "train_total": train_values[0],
                    "train_reconstruction": train_values[1],
                    "train_kl": train_values[2],
                    "validation_total": validation_values[0],
                    "validation_reconstruction": validation_values[1],
                    "validation_kl": validation_values[2],
                })
                if validation_values[0] < best_validation - 1e-8:
                    best_validation = validation_values[0]
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
            if best_state is None:
                raise AssertionError("VAE training не сохранил best state")
            model.load_state_dict(best_state)
            checkpoint_path = output_dir / f"vae_CH{channel}_seed_{seed}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "channel": channel,
                "seed": seed,
                "input_dim": 144,
                "hidden_dims": config.hidden_dims,
                "latent_dim": config.latent_dim,
                "best_epoch": best_epoch,
                "config": asdict(config),
            }, checkpoint_path)
            history_path = output_dir / f"history_CH{channel}_seed_{seed}.csv"
            _save_history(history_path, history)
            print(
                f"CH{channel}, seed={seed}: best epoch={best_epoch}, "
                f"validation loss={best_validation:.6f}"
            )
            runs.append(VAETrainingRun(
                channel,
                seed,
                model,
                tuple(history),
                best_epoch,
                checkpoint_path,
            ))
    return VAEEnsemble(tuple(runs))


def _encode(model: PulseVAE, array: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    mus, logvars, errors, reconstructions = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(array), batch_size):
            batch = torch.from_numpy(array[start : start + batch_size])
            mu, logvar = model.encode(batch)
            reconstruction = model.decode(mu)
            mus.append(mu.numpy())
            logvars.append(logvar.numpy())
            reconstructions.append(reconstruction.numpy())
            errors.append(torch.mean((reconstruction - batch) ** 2, dim=1).numpy())
    return {
        "mu": np.concatenate(mus),
        "logvar": np.concatenate(logvars),
        "reconstruction": np.concatenate(reconstructions),
        "reconstruction_mse": np.concatenate(errors),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def _rank_auc(control: np.ndarray, mixture: np.ndarray) -> float:
    values = np.concatenate([control, mixture])
    labels = np.concatenate([np.zeros(len(control)), np.ones(len(mixture))])
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for group in np.flatnonzero(counts > 1):
        tied = np.flatnonzero(inverse == group)
        ranks[tied] = ranks[tied].mean()
    n1, n0 = int(labels.sum()), int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    auc = (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def _alignment_rows(
    channel: int,
    config: RealVAEConfig,
    data: RealVAEData,
    encodings: dict[tuple[int, int], dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    reference_seed = config.model_seeds[0]
    validation = np.flatnonzero(
        (data.sample.channels == channel) & (data.split == "validation")
    )
    channel_indices = np.flatnonzero(data.sample.channels == channel)
    local_validation = np.searchsorted(channel_indices, validation)
    reference = encodings[(channel, reference_seed)]["mu"][local_validation]
    rows: list[dict[str, object]] = []
    for seed in config.model_seeds:
        candidate = encodings[(channel, seed)]["mu"][local_validation]
        best: tuple[float, tuple[int, ...], tuple[float, ...]] | None = None
        for permutation in itertools.permutations(range(config.latent_dim)):
            correlations = tuple(
                _pearson(reference[:, i], candidate[:, permutation[i]])
                for i in range(config.latent_dim)
            )
            score = float(np.nansum(np.abs(correlations)))
            if best is None or score > best[0]:
                best = (score, permutation, correlations)
        assert best is not None
        _, permutation, correlations = best
        for reference_latent, (candidate_latent, correlation) in enumerate(
            zip(permutation, correlations), start=1
        ):
            rows.append({
                "channel": channel,
                "reference_seed": reference_seed,
                "seed": seed,
                "reference_latent": reference_latent,
                "matched_latent": candidate_latent + 1,
                "sign": 1 if correlation >= 0 else -1,
                "correlation": correlation,
            })
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_real_vae_latents(
    data: RealVAEData,
    ensemble: VAEEnsemble,
    paths: ProjectPaths,
    config: RealVAEConfig,
) -> LatentAudit:
    """Encode all real pulses and audit every latent against technical factors."""

    output_dir = paths.output_dir / "vae_real"
    encodings: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    correlation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    qlong_rows: list[dict[str, object]] = []
    metrics = {
        "Qlong": data.root.energy.astype(float),
        "classical_psd": data.classical_psd,
        "shape_score": data.shape_score,
        "amplitude": data.audit.amplitude,
        "snr": data.audit.snr,
        "baseline_rms": data.audit.baseline_rms,
        "cfd_shift": data.audit.cfd_time - 20.0,
    }
    for training_run in ensemble.runs:
        channel = training_run.channel
        channel_indices = np.flatnonzero(data.sample.channels == channel)
        array = np.ascontiguousarray(
            data.processed.aligned_normalized[channel_indices], dtype=np.float32
        )
        encoded = _encode(training_run.model, array, config.batch_size)
        encodings[(channel, training_run.seed)] = encoded
        latent_rows: list[dict[str, object]] = []
        for local, global_index in enumerate(channel_indices):
            row: dict[str, object] = {
                "sample_row": int(global_index),
                "run": str(data.sample.runs[global_index]),
                "channel": channel,
                "detector_label": DETECTOR_LABELS[channel],
                "source_file": data.sample.provenance[global_index][2],
                "source_row": int(data.sample.provenance[global_index][3]),
                "split": str(data.split[global_index]),
                "Qlong": int(data.root.energy[global_index]),
                "Qshort": int(data.root.energy_short[global_index]),
                "classical_psd": data.classical_psd[global_index],
                "shape_score": data.shape_score[global_index],
                "amplitude": data.audit.amplitude[global_index],
                "snr": data.audit.snr[global_index],
                "low_snr": int(data.processed.low_snr[global_index]),
                "structural_ok": int(data.structural_ok[global_index]),
                "reconstruction_mse": encoded["reconstruction_mse"][local],
            }
            for latent in range(config.latent_dim):
                row[f"z{latent + 1}_mu"] = encoded["mu"][local, latent]
                row[f"z{latent + 1}_logvar"] = encoded["logvar"][local, latent]
            latent_rows.append(row)
        _write_rows(
            output_dir / f"latents_CH{channel}_seed_{training_run.seed}.csv",
            latent_rows,
        )
        validation_global = np.flatnonzero(
            (data.sample.channels == channel) & (data.split == "validation")
        )
        validation_local = np.searchsorted(channel_indices, validation_global)
        structural_mask = data.structural_ok[channel_indices]
        for latent in range(config.latent_dim):
            values = encoded["mu"][:, latent]
            for metric_name, metric_values in metrics.items():
                correlation_rows.append({
                    "channel": channel,
                    "seed": training_run.seed,
                    "latent": latent + 1,
                    "metric": metric_name,
                    "pearson": _pearson(
                        values[structural_mask],
                        metric_values[channel_indices][structural_mask],
                    ),
                })
            cf_values = values[
                (data.sample.runs[channel_indices] == CF_RUN) & structural_mask
            ]
            co_values = values[
                (data.sample.runs[channel_indices] == CO_RUN) & structural_mask
            ]
            summary_rows.append({
                "channel": channel,
                "seed": training_run.seed,
                "latent": latent + 1,
                "posterior_mu_variance_validation": float(
                    np.var(encoded["mu"][validation_local, latent])
                ),
                "mean_kl_validation": float(np.mean(
                    -0.5 * (
                        1
                        + encoded["logvar"][validation_local, latent]
                        - encoded["mu"][validation_local, latent] ** 2
                        - np.exp(encoded["logvar"][validation_local, latent])
                    )
                )),
                "co_cf_run_auc": _rank_auc(co_values, cf_values),
            })
        if training_run.seed == config.model_seeds[0]:
            for run in (CO_RUN, CF_RUN):
                run_mask = (
                    (data.sample.runs[channel_indices] == run)
                    & data.structural_ok[channel_indices]
                )
                run_global = channel_indices[run_mask]
                run_local = np.flatnonzero(run_mask)
                energies = data.root.energy[run_global].astype(float)
                edges = np.unique(np.quantile(
                    energies, np.linspace(0, 1, config.qlong_bins + 1)
                ))
                bins = np.digitize(energies, edges[1:-1], right=True)
                for bin_index in np.unique(bins):
                    selected = run_local[bins == bin_index]
                    for latent in range(config.latent_dim):
                        values = encoded["mu"][selected, latent]
                        q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
                        qlong_values = data.root.energy[channel_indices[selected]]
                        qlong_rows.append({
                            "channel": channel,
                            "run": run,
                            "qlong_bin": int(bin_index),
                            "qlong_low": int(np.min(qlong_values)),
                            "qlong_high": int(np.max(qlong_values)),
                            "qlong_median": float(np.median(qlong_values)),
                            "latent": latent + 1,
                            "n": len(values),
                            "q10": float(q10),
                            "median": float(q50),
                            "q90": float(q90),
                        })
    stability_rows: list[dict[str, object]] = []
    for channel in sorted(DETECTOR_LABELS):
        stability_rows.extend(_alignment_rows(channel, config, data, encodings))
    _write_rows(output_dir / "latent_correlations.csv", correlation_rows)
    _write_rows(output_dir / "latent_stability.csv", stability_rows)
    _write_rows(output_dir / "latent_summary.csv", summary_rows)
    _write_rows(output_dir / "latent_qlong_bins.csv", qlong_rows)
    return LatentAudit(
        encodings,
        tuple(correlation_rows),
        tuple(stability_rows),
        tuple(qlong_rows),
        tuple(summary_rows),
        output_dir,
    )


def _save_show(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, config=PLOTLY_CONFIG, include_plotlyjs="directory")
    fig.show(renderer="plotly_mimetype", config=PLOTLY_CONFIG)


def _primary_run(ensemble: VAEEnsemble, channel: int, seed: int) -> VAETrainingRun:
    return next(run for run in ensemble.runs if run.channel == channel and run.seed == seed)


def plot_real_vae_results(
    data: RealVAEData,
    ensemble: VAEEnsemble,
    latent_audit: LatentAudit,
    config: RealVAEConfig,
) -> list[go.Figure]:
    """Create interactive training, reconstruction, latent, and traversal audits."""

    figures: list[go.Figure] = []
    primary_seed = config.model_seeds[0]
    colors = {CO_RUN: "#1f77b4", CF_RUN: "#ff7f0e"}
    labels = {CO_RUN: "60Co gamma-control", CF_RUN: "252Cf mixture"}
    for channel in sorted(DETECTOR_LABELS):
        channel_runs = [run for run in ensemble.runs if run.channel == channel]
        history_fig = go.Figure()
        for run in channel_runs:
            history_fig.add_trace(go.Scatter(
                x=[row["epoch"] for row in run.history],
                y=[row["validation_total"] for row in run.history],
                mode="lines",
                name=f"seed {run.seed}",
            ))
        history_fig.update_layout(
            title=f"VAE validation loss — CH{channel}: {DETECTOR_LABELS[channel]}",
            xaxis_title="Epoch",
            yaxis_title="Validation objective",
            template="plotly_white",
            height=520,
            legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"},
        )
        _save_show(history_fig, latent_audit.output_dir / f"training_CH{channel}.html")
        figures.append(history_fig)

        channel_indices = np.flatnonzero(data.sample.channels == channel)
        encoded = latent_audit.encodings[(channel, primary_seed)]
        reconstruction_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("60Co external control", "252Cf validation"),
            shared_yaxes=True,
        )
        for column, (run_name, split_name) in enumerate(
            ((CO_RUN, "control"), (CF_RUN, "validation")), start=1
        ):
            global_mask = (
                (data.sample.channels == channel) & (data.split == split_name)
            )
            global_indices = np.flatnonzero(global_mask)
            local_indices = np.searchsorted(channel_indices, global_indices)
            original = data.processed.aligned_normalized[global_indices]
            reconstructed = encoded["reconstruction"][local_indices]
            for values, name, dash in (
                (np.median(original, axis=0), "Real median", "solid"),
                (np.median(reconstructed, axis=0), "VAE median reconstruction", "dash"),
            ):
                reconstruction_fig.add_trace(go.Scatter(
                    x=np.arange(144),
                    y=values,
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    line={"dash": dash},
                ), row=1, col=column)
        reconstruction_fig.update_layout(
            title=f"Real versus reconstructed pulses — CH{channel}: {DETECTOR_LABELS[channel]}",
            template="plotly_white",
            height=520,
            margin={"l": 75, "r": 35, "t": 95, "b": 70},
            legend={"orientation": "h", "y": 1.04, "yanchor": "bottom"},
        )
        reconstruction_fig.update_xaxes(title_text="Aligned sample index")
        reconstruction_fig.update_yaxes(title_text="Peak-normalized amplitude", row=1, col=1)
        _save_show(
            reconstruction_fig,
            latent_audit.output_dir / f"reconstruction_CH{channel}.html",
        )
        figures.append(reconstruction_fig)

        histogram_fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("z1", "z2", "z3"),
        )
        for latent in range(3):
            for run_name in (CO_RUN, CF_RUN):
                mask = (
                    (data.sample.runs[channel_indices] == run_name)
                    & data.structural_ok[channel_indices]
                )
                histogram_fig.add_trace(go.Histogram(
                    x=encoded["mu"][mask, latent],
                    histnorm="probability density",
                    opacity=0.55,
                    nbinsx=60,
                    name=labels[run_name],
                    legendgroup=run_name,
                    showlegend=latent == 0,
                    marker_color=colors[run_name],
                ), row=1, col=latent + 1)
        histogram_fig.update_layout(
            title=f"All three posterior means by acquisition — CH{channel}: {DETECTOR_LABELS[channel]}",
            barmode="overlay",
            template="plotly_white",
            height=520,
            margin={"l": 70, "r": 35, "t": 100, "b": 70},
            legend={"orientation": "h", "y": 1.04, "yanchor": "bottom"},
        )
        histogram_fig.update_xaxes(title_text="Posterior mean")
        histogram_fig.update_yaxes(title_text="Probability density", row=1, col=1)
        _save_show(histogram_fig, latent_audit.output_dir / f"latent_hist_CH{channel}.html")
        figures.append(histogram_fig)

        scatter_fig = go.Figure()
        for run_name in (CO_RUN, CF_RUN):
            eligible = np.flatnonzero(
                (data.sample.runs[channel_indices] == run_name)
                & data.structural_ok[channel_indices]
            )
            rng = np.random.default_rng(config.data_seed + channel)
            if len(eligible) > 1500:
                eligible = np.sort(rng.choice(eligible, 1500, replace=False))
            scatter_fig.add_trace(go.Scatter3d(
                x=encoded["mu"][eligible, 0],
                y=encoded["mu"][eligible, 1],
                z=encoded["mu"][eligible, 2],
                mode="markers",
                name=labels[run_name],
                marker={"size": 2.5, "opacity": 0.5, "color": colors[run_name]},
                customdata=data.root.energy[channel_indices[eligible]],
                hovertemplate="z1=%{x:.3f}<br>z2=%{y:.3f}<br>z3=%{z:.3f}<br>Qlong=%{customdata}<extra></extra>",
            ))
        scatter_fig.update_layout(
            title=f"Three-dimensional VAE representation — CH{channel}: {DETECTOR_LABELS[channel]}",
            template="plotly_white",
            height=650,
            margin={"l": 20, "r": 190, "t": 80, "b": 20},
            scene={
                "xaxis_title": "z1 posterior mean",
                "yaxis_title": "z2 posterior mean",
                "zaxis_title": "z3 posterior mean",
            },
            legend={"orientation": "v", "x": 1.02, "y": 1.0},
        )
        _save_show(scatter_fig, latent_audit.output_dir / f"latent_3d_CH{channel}.html")
        figures.append(scatter_fig)

        qlong_fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("z1", "z2", "z3"),
        )
        for latent in range(1, 4):
            for run_name in (CO_RUN, CF_RUN):
                rows = [
                    row for row in latent_audit.qlong_rows
                    if row["channel"] == channel
                    and row["latent"] == latent
                    and row["run"] == run_name
                ]
                x = np.array([row["qlong_median"] for row in rows])
                median = np.array([row["median"] for row in rows])
                q10 = np.array([row["q10"] for row in rows])
                q90 = np.array([row["q90"] for row in rows])
                qlong_fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([q90, q10[::-1]]),
                    fill="toself",
                    fillcolor=("rgba(31,119,180,0.12)" if run_name == CO_RUN else "rgba(255,127,14,0.12)"),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=run_name,
                ), row=1, col=latent)
                qlong_fig.add_trace(go.Scatter(
                    x=x,
                    y=median,
                    mode="lines+markers",
                    name=labels[run_name],
                    showlegend=latent == 1,
                    legendgroup=run_name,
                    line={"color": colors[run_name]},
                ), row=1, col=latent)
        qlong_fig.update_layout(
            title=f"Latents within Qlong strata — CH{channel}: {DETECTOR_LABELS[channel]}",
            template="plotly_white",
            height=520,
            margin={"l": 70, "r": 35, "t": 100, "b": 75},
            legend={"orientation": "h", "y": 1.04, "yanchor": "bottom"},
        )
        qlong_fig.update_xaxes(title_text="Recorded Qlong, ADC channels", type="log")
        qlong_fig.update_yaxes(title_text="Posterior mean; median and 10–90%", row=1, col=1)
        _save_show(qlong_fig, latent_audit.output_dir / f"latent_qlong_CH{channel}.html")
        figures.append(qlong_fig)

        primary = _primary_run(ensemble, channel, primary_seed)
        cf_validation_global = np.flatnonzero(
            (data.sample.channels == channel) & (data.split == "validation")
        )
        cf_validation_local = np.searchsorted(channel_indices, cf_validation_global)
        validation_mu = encoded["mu"][cf_validation_local]
        traversal_fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Vary z1", "Vary z2", "Vary z3"),
            shared_yaxes=True,
        )
        base = np.median(validation_mu, axis=0)
        for latent in range(3):
            values = np.quantile(
                validation_mu[:, latent], config.traversal_quantiles
            )
            latent_vectors = np.repeat(base[None, :], len(values), axis=0)
            latent_vectors[:, latent] = values
            with torch.no_grad():
                decoded = primary.model.decode(
                    torch.from_numpy(latent_vectors.astype(np.float32))
                ).numpy()
            for quantile, value, pulse in zip(
                config.traversal_quantiles, values, decoded
            ):
                traversal_fig.add_trace(go.Scatter(
                    x=np.arange(144),
                    y=pulse,
                    mode="lines",
                    name=f"q={quantile:.0%}, z={value:.3f}",
                    legendgroup=f"z{latent + 1}",
                    showlegend=latent == 0,
                    hovertemplate=f"z{latent + 1}={value:.3f}<br>sample=%{{x}}<br>decoded=%{{y:.3f}}<extra></extra>",
                ), row=1, col=latent + 1)
        traversal_fig.update_layout(
            title=f"Traversal of every real posterior coordinate — CH{channel}: {DETECTOR_LABELS[channel]}",
            template="plotly_white",
            height=560,
            margin={"l": 75, "r": 35, "t": 100, "b": 75},
            legend={"orientation": "v", "x": 1.01, "y": 1.0},
        )
        traversal_fig.update_xaxes(title_text="Aligned sample index")
        traversal_fig.update_yaxes(title_text="Decoded peak-normalized amplitude", row=1, col=1)
        _save_show(traversal_fig, latent_audit.output_dir / f"traversal_CH{channel}.html")
        figures.append(traversal_fig)
    return figures


def print_real_vae_findings(
    latent_audit: LatentAudit,
    config: RealVAEConfig,
) -> None:
    print("Все три латентные переменные сохранены; координаты не считаются физическими метками.")
    print("Согласование seed выполнено только по перестановке и знаку на Cf-validation.")
    for channel in sorted(DETECTOR_LABELS):
        rows = [
            row for row in latent_audit.summary_rows
            if row["channel"] == channel and row["seed"] == config.model_seeds[0]
        ]
        print(f"CH{channel} {DETECTOR_LABELS[channel]}:")
        for row in rows:
            print(
                f"  z{row['latent']}: var(mu)={row['posterior_mu_variance_validation']:.4g}, "
                f"mean KL={row['mean_kl_validation']:.4g}, "
                f"Co-vs-Cf run AUC={row['co_cf_run_auc']:.3f}"
            )
    print(
        "Важно: Co-vs-Cf AUC измеряет различимость запусков, а не истинную "
        "gamma/neutron классификацию."
    )
