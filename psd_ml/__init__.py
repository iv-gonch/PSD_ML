"""Project-local tools for reproducible pulse-shape analysis."""

from .pipeline import (
    DETECTOR_LABELS,
    PipelineConfig,
    audit_waveforms,
    characterize_processed_shapes,
    configure_plotly,
    discover_project,
    inventory_sources,
    plot_processed_shapes,
    plot_raw_groups,
    preprocess_waveforms,
    print_audit,
    print_characteristics,
    print_inventory,
    print_quality_summary,
    sample_waveforms,
    save_processed_data,
    save_sample,
    summarize_and_validate,
)

__all__ = [name for name in globals() if not name.startswith("_")]
