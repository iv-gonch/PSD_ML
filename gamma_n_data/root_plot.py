#!/usr/bin/env python3
"""Save a ROOT histogram, tree expression, or waveform as a PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import ROOT
except ImportError:
    sys.exit("PyROOT is not installed. Install CERN ROOT: https://root.cern/install/")

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1110)


def open_file(path: str):
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        sys.exit(f"Cannot open ROOT file: {path}")
    return root_file


def save(canvas, output: str):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(output)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot ROOT data to PNG/PDF")
    parser.add_argument("file", help="path to a .root file")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--object", help="histogram path, e.g. Energy/EnergyCH0@DT5730SB_27616")
    modes.add_argument("--expr", help="TTree::Draw expression, e.g. Energy or (Energy-EnergyShort)/Energy:Energy")
    modes.add_argument("--waveform", type=int, metavar="ENTRY", help="plot Samples for one event")
    parser.add_argument("--tree", default="Data", help="TTree path (default: Data)")
    parser.add_argument("--cut", default="", help="TTree selection, e.g. Energy>1000")
    parser.add_argument("--bins", default="", help="histogram bins, e.g. 400,0,32000 or 300,0,30000,200,0,1")
    parser.add_argument("-o", "--output", default="root_plot.png", help="output PNG/PDF")
    args = parser.parse_args()

    root_file = open_file(args.file)
    canvas = ROOT.TCanvas("canvas", "ROOT viewer", 1100, 750)
    canvas.SetRightMargin(0.14)

    if args.object:
        obj = root_file.Get(args.object)
        if not obj:
            sys.exit(f"Object not found: {args.object}")
        if not obj.InheritsFrom("TH1"):
            sys.exit(f"Object is {obj.ClassName()}, not a TH1/TH2 histogram")
        obj.SetDirectory(0)
        obj.SetTitle(args.object)
        obj.Draw("COLZ" if obj.InheritsFrom("TH2") else "HIST")
    elif args.expr:
        tree = root_file.Get(args.tree)
        if not tree or not tree.InheritsFrom("TTree"):
            sys.exit(f"TTree not found: {args.tree}")
        hist_spec = f">>view({args.bins})" if args.bins else ">>view"
        drawn = tree.Draw(args.expr + hist_spec, args.cut, "COLZ" if ":" in args.expr else "HIST")
        if drawn < 0:
            sys.exit("ROOT could not draw the expression; check branch names and --bins")
        ROOT.gPad.SetGrid()
    else:
        tree = root_file.Get(args.tree)
        if not tree or not tree.InheritsFrom("TTree"):
            sys.exit(f"TTree not found: {args.tree}")
        if args.waveform < 0 or args.waveform >= tree.GetEntries():
            sys.exit(f"Entry must be between 0 and {tree.GetEntries() - 1}")
        tree.GetEntry(args.waveform)
        try:
            samples = tree.Samples
        except AttributeError:
            sys.exit("The tree has no Samples branch")
        graph = ROOT.TGraph(len(samples))
        for i, value in enumerate(samples):
            graph.SetPoint(i, i, value)
        graph.SetTitle(f"Waveform, entry {args.waveform};sample;ADC value")
        graph.SetLineColor(ROOT.kBlue + 1)
        graph.Draw("AL")
        ROOT.gPad.SetGrid()

    canvas.Update()
    save(canvas, args.output)


if __name__ == "__main__":
    main()
