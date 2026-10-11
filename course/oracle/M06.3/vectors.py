# /// script
# requires-python = ">=3.11"
# ///
"""Maintainer generator for course/fixtures/M06.3/pcg32.vectors.json.

The course tests of M06.3 and M07.0 need the reference vectors of
spec/pcg32.md, but a test cannot reach the contracts directory once it is
vendored by `ol export` (only fixtures travel with it). This script rebuilds
the vectors from spec/pcg32.md with the independent stdlib generator in
course/oracle/contracts/pcg32_vectors.py, checks that they equal the published
contract file byte for byte, and writes the same text as a fixture.

    uv run --script course/oracle/M06.3/vectors.py          # rewrite + MANIFEST row
    uv run --script course/oracle/M06.3/vectors.py --check  # compare only
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "course" / "contracts" / "spec" / "pcg32.vectors.json"
OUT = ROOT / "course" / "fixtures" / "M06.3" / "pcg32.vectors.json"
GEN = ROOT / "course" / "oracle" / "contracts" / "pcg32_vectors.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("pcg32_vectors", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.render(mod.build())
    if text != SPEC.read_text():
        print(f"{SPEC} is stale: regenerate it first", file=sys.stderr)
        return 1
    if "--check" in sys.argv:
        return 0 if OUT.read_text() == text else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    data = OUT.read_bytes()
    print(
        "\t".join(
            [
                OUT.relative_to(ROOT).as_posix(),
                hashlib.sha256(data).hexdigest(),
                str(len(data)),
                "course/oracle/M06.3/vectors.py",
                "-",
                "spec/pcg32.md",
                "Apache-2.0",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
