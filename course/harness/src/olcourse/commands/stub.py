"""ol stub <file> [--id ID]   print the compiling stub of a marked file (maintainer aid)"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import markers


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ol stub")
    ap.add_argument("file", type=Path)
    ap.add_argument("--id")
    a = ap.parse_args(argv)
    print(markers.stub(a.file.read_text(), str(a.file), a.id), end="")
    return 0
