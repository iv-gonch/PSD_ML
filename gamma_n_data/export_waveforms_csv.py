#!/usr/bin/env python3
"""Export CoMPASS waveform TTrees to headerless CSV files.

Each CSV row is one pulse; columns are sample values in acquisition order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import ROOT
except ImportError:
    sys.exit("PyROOT is not installed. Use the project .venv with CERN ROOT available.")

ROOT.gROOT.SetBatch(True)


def export_file(source: Path, destination: Path, tree_name: str = "Data") -> tuple[int, int]:
    root_file = ROOT.TFile.Open(str(source), "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Cannot open ROOT file: {source}")

    tree = root_file.Get(tree_name)
    if not tree or not tree.InheritsFrom("TTree"):
        root_file.Close()
        raise KeyError(f"TTree '{tree_name}' not found in {source}")
    if not tree.GetBranch("Samples"):
        root_file.Close()
        raise KeyError(f"Branch 'Samples' not found in {source}:{tree_name}")

    tree.SetBranchStatus("*", False)
    tree.SetBranchStatus("Samples", True)
    entries = int(tree.GetEntries())
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    sample_count: int | None = None
    try:
        with partial.open("w", encoding="ascii", newline="", buffering=1024 * 1024) as stream:
            for index in range(entries):
                if tree.GetEntry(index) <= 0:
                    raise OSError(f"Failed to read entry {index} from {source}")
                samples = tree.Samples
                current_count = len(samples)
                if sample_count is None:
                    sample_count = current_count
                elif current_count != sample_count:
                    raise ValueError(
                        f"Waveform length changed at entry {index}: "
                        f"expected {sample_count}, got {current_count}"
                    )
                stream.write(",".join(str(int(value)) for value in samples))
                stream.write("\n")
        partial.replace(destination)
    finally:
        root_file.Close()

    return entries, sample_count or 0


def discover_inputs(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("call_all_*/UNFILTERED/Data_*.root"))


def destination_for(source: Path, data_dir: Path, output_dir: Path) -> Path:
    run_name = source.relative_to(data_dir).parts[0]
    return output_dir / run_name / f"{source.stem}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ROOT Samples branches: one headerless CSV row per pulse"
    )
    parser.add_argument(
        "files", nargs="*", type=Path, help="Data_*.root files; default: discover all runs"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent,
        help="gamma_n_data directory used for discovery and run names",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="output root (default: DATA_DIR/CSV)"
    )
    parser.add_argument("--tree", default="Data", help="TTree name (default: Data)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir / "CSV").resolve()
    sources = [path.resolve() for path in args.files] or discover_inputs(data_dir)
    if not sources:
        parser.error("no input Data_*.root files found")

    for source in sources:
        if not source.is_file():
            parser.error(f"file not found: {source}")
        destination = destination_for(source, data_dir, output_dir)
        print(f"Exporting {source} -> {destination}", flush=True)
        entries, samples = export_file(source, destination, args.tree)
        print(f"Done: {entries:,} pulses, {samples} samples each", flush=True)


if __name__ == "__main__":
    main()
