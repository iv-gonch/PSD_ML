#!/usr/bin/env python3
"""Text-mode inspector for CERN ROOT files (uses the installed PyROOT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import ROOT
except ImportError:
    sys.exit("PyROOT is not installed. Install CERN ROOT: https://root.cern/install/")

ROOT.gROOT.SetBatch(True)


def open_file(path: str):
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        sys.exit(f"Cannot open ROOT file: {path}")
    return root_file


def walk(directory, prefix: str = ""):
    for key in directory.GetListOfKeys():
        name = key.GetName()
        full_name = f"{prefix}/{name}" if prefix else name
        class_name = key.GetClassName()
        print(f"{full_name:<60} {class_name}")
        cls = ROOT.TClass.GetClass(class_name)
        if cls and cls.InheritsFrom("TDirectory"):
            subdir = key.ReadObj()
            walk(subdir, full_name)


def find_trees(directory, prefix: str = ""):
    trees = []
    for key in directory.GetListOfKeys():
        name = key.GetName()
        full_name = f"{prefix}/{name}" if prefix else name
        class_name = key.GetClassName()
        cls = ROOT.TClass.GetClass(class_name)
        if cls and cls.InheritsFrom("TTree"):
            trees.append(full_name)
        elif cls and cls.InheritsFrom("TDirectory"):
            trees.extend(find_trees(key.ReadObj(), full_name))
    return trees


def get_tree(root_file, name: str | None):
    if name:
        tree = root_file.Get(name)
        if not tree or not tree.InheritsFrom("TTree"):
            sys.exit(f"Object '{name}' is not a TTree")
        return tree
    names = find_trees(root_file)
    if not names:
        sys.exit("This file contains no TTrees")
    if len(names) > 1:
        sys.exit("Several TTrees found; select one with --tree: " + ", ".join(names))
    return root_file.Get(names[0])


def describe_tree(tree):
    print(f"Tree: {tree.GetName()}  entries: {tree.GetEntries():,}")
    print(f"{'branch':<24} {'type':<28} {'compressed size':>15}")
    for branch in tree.GetListOfBranches():
        leaf = branch.GetLeaf(branch.GetName())
        kind = leaf.GetTypeName() if leaf else branch.GetClassName() or "object/array"
        print(f"{branch.GetName():<24} {kind:<28} {branch.GetZipBytes():>15,}")


def short_value(value, limit: int = 8):
    if isinstance(value, (str, bytes, int, float, bool)):
        return repr(value)
    try:
        size = len(value)
        shown = [value[i] for i in range(min(size, limit))]
        suffix = f", ... ({size} values)" if size > limit else ""
        return "[" + ", ".join(map(str, shown)) + suffix + "]"
    except (TypeError, AttributeError):
        return str(value)


def show_entries(tree, count: int, branches: list[str] | None):
    available = {b.GetName() for b in tree.GetListOfBranches()}
    selected = branches or list(available)
    unknown = set(selected) - available
    if unknown:
        sys.exit("Unknown branches: " + ", ".join(sorted(unknown)))
    for branch in available:
        tree.SetBranchStatus(branch, branch in selected)
    for index in range(min(count, tree.GetEntries())):
        tree.GetEntry(index)
        values = "  ".join(f"{name}={short_value(getattr(tree, name))}" for name in selected)
        print(f"[{index}] {values}")


def main():
    parser = argparse.ArgumentParser(description="Inspect ROOT file contents in a terminal")
    parser.add_argument("file", help="path to a .root file")
    parser.add_argument("--tree", help="TTree path (auto-selected when there is only one)")
    parser.add_argument("--entries", type=int, metavar="N", help="print the first N tree entries")
    parser.add_argument("--branches", help="comma-separated branches used with --entries")
    args = parser.parse_args()

    if not Path(args.file).is_file():
        parser.error(f"file not found: {args.file}")
    root_file = open_file(args.file)
    print(f"File: {args.file}  ({Path(args.file).stat().st_size / 1024**2:.1f} MiB)")
    print(f"{'object':<60} class")
    walk(root_file)

    trees = find_trees(root_file)
    if trees:
        print()
        tree = get_tree(root_file, args.tree)
        describe_tree(tree)
        if args.entries is not None:
            branches = args.branches.split(",") if args.branches else None
            print()
            show_entries(tree, max(args.entries, 0), branches)


if __name__ == "__main__":
    main()
