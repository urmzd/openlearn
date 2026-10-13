"""Cross-platform libm rounding must not rewrite published RNG fixtures."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("changed", ["roundoff", "integer", "normal"])
def test_rng_fixture_uses_contract_tolerance(tmp_path, monkeypatch, changed):
    course = Path(__file__).resolve().parents[2]
    path = course / "oracle/M06.3/vectors.py"
    spec = importlib.util.spec_from_file_location("fixture_copy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = json.loads((course / "contracts/spec/pcg32.vectors.json").read_text())
    if changed == "roundoff":
        data["normal"]["0"][0] = math.nextafter(data["normal"]["0"][0], math.inf)
    elif changed == "integer":
        data["next_u32"]["0"][0] += 1
    else:
        data["normal"]["0"][0] += 0.01
    published = tmp_path / "contract.json"
    published.write_text(json.dumps(data, indent=2) + "\n")
    out = tmp_path / "fixture.json"
    monkeypatch.setattr(mod, "SPEC", published)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(path)])
    assert mod.main() == (0 if changed == "roundoff" else 1)
    if changed == "roundoff":
        assert out.read_bytes() == published.read_bytes()
        monkeypatch.setattr(sys, "argv", [str(path), "--check"])
        assert mod.main() == 0
        out.write_text("{}\n")
        assert mod.main() == 1
    else:
        assert not out.exists()
