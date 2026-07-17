#!/usr/bin/env python3
"""Open a ROOT file in ROOT's interactive TBrowser GUI."""

import argparse
import sys

try:
    import ROOT
except ImportError:
    sys.exit("PyROOT is not installed. Install CERN ROOT: https://root.cern/install/")


def main():
    parser = argparse.ArgumentParser(description="Open an interactive ROOT browser")
    parser.add_argument("file", help="path to a .root file")
    args = parser.parse_args()
    root_file = ROOT.TFile.Open(args.file, "READ")
    if not root_file or root_file.IsZombie():
        sys.exit(f"Cannot open ROOT file: {args.file}")
    # Supplying width and height avoids an ambiguous PyROOT overload.  The
    # TObject overload is: name, object, title, width, height.
    browser = ROOT.TBrowser("browser", root_file, args.file, 1100, 750)
    print("Double-click a histogram or TTree in the browser. Close the window to exit.")
    ROOT.gApplication.Run()
    _ = browser, root_file


if __name__ == "__main__":
    main()
