import re
from pathlib import Path

import pytest

from olcourse import HarnessError, ids, registry, tomlw

ROOT = Path(__file__).resolve().parents[3]


def test_course_id_regex_matches_bash():
    ss = (ROOT / "practice" / "bin" / "ol").read_text()
    m = re.search(r"^COURSE_ID_RE='([^']+)'", ss, re.M)
    assert m, "COURSE_ID_RE missing from practice/bin/ol"
    assert m.group(1) == ids.COURSE_ID_RE


@pytest.mark.parametrize(
    "good",
    [
        "M03.1",
        "M04.12",
        "L0.0",
        "L10.2",
        "C1",
        "C2",
        "S-M07",
        "S-M07a",
        "lang.01",
        "lang.11",
        "ds.09",
        "rt.01",
        "craft.13",
        "iv.01",
        "sq.multi-lora",
        "MS-P1",
        "MS-gateway",
        "L9.1+cuda",
    ],
)
def test_ids_accepted(good):
    assert ids.is_course_id(good)


@pytest.mark.parametrize(
    "bad",
    [
        "predict",
        "build",
        "reattempt",
        "M3.1",
        "L100.1",
        "C3",
        "S-M7",
        "S-M07ab",
        "lang.1",
        "xx.01",
        "sq.Upper",
        "02",
        "L9.1+rocm",
    ],
)
def test_ids_rejected(bad):
    assert not ids.is_course_id(bad)


def test_milestones_are_not_modules():
    assert ids.is_course_id("MS-P1") and not ids.is_module_id("MS-P1")


def test_range_expansion():
    assert ids.expand(["ds.01 to ds.03", "L0.1"]) == ["ds.01", "ds.02", "ds.03", "L0.1"]
    assert ids.expand(["L0.1 to L0.3"]) == ["L0.1", "L0.2", "L0.3"]
    with pytest.raises(ValueError):
        ids.expand(["ds.01 to rt.03"])


def test_underscore():
    assert ids.underscore("dur.06") == "dur_06"
    assert ids.underscore("L10.2") == "l10_2"


def _write(d: Path, mid: str, body: str) -> None:
    (d / "modules").mkdir(parents=True, exist_ok=True)
    (d / "modules" / f"{mid}.toml").write_text(f'id = "{mid}"\ntitle = "t"\n' + body)


BASE = {
    "M90.1": 'kind = "build"\nlang = ["python"]\npass = 1\nunits = ["python/a.py"]\nused_by = ["M90.2"]\n',
    "M90.2": 'kind = "build"\nlang = ["python"]\npass = 2\nunits = ["python/b.py"]\ndeps = ["M90.1"]\nused_by = ["craft.90"]\n',
    "craft.90": 'kind = "practice"\nlang = ["docs"]\npass = 3\ndeps = ["M90.2"]\n',
}


def _reg(tmp_path: Path, **override) -> registry.Registry:
    mods = dict(BASE)
    mods.update(override)
    for mid, body in mods.items():
        if body is not None:
            _write(tmp_path, mid, body)
    return registry.load(tmp_path)


def test_valid_registry_has_no_violations(tmp_path):
    reg = _reg(tmp_path)
    assert registry.invariants(reg) == []
    assert reg.closure("craft.90") == ["M90.1", "M90.2"]
    assert reg.dependents("M90.1") == ["M90.2", "craft.90"]


def test_mirror_rule_missing_and_extra(tmp_path):
    reg = _reg(
        tmp_path,
        **{
            "M90.1": BASE["M90.1"].replace(
                'used_by = ["M90.2"]', 'used_by = ["craft.90"]'
            )
        },
    )
    errs = "\n".join(registry.invariants(reg))
    assert "M90.1: used_by is missing ['M90.2']" in errs
    assert "M90.1: used_by lists ['craft.90']" in errs


def test_pass_order(tmp_path):
    reg = _reg(tmp_path, **{"M90.1": BASE["M90.1"].replace("pass = 1", "pass = 5")})
    errs = "\n".join(registry.invariants(reg))
    assert "dep M90.1 is taught later" in errs


def test_build_without_call_site(tmp_path):
    reg = _reg(
        tmp_path,
        **{
            "M90.2": BASE["M90.2"].replace('used_by = ["craft.90"]', "used_by = []"),
            "craft.90": None,
        },
    )
    assert registry.call_site_errors(reg, reg.get("M90.2"))


def test_ownership_and_upgrades(tmp_path):
    reg = _reg(
        tmp_path,
        **{
            "M90.3": 'kind = "build"\nlang = ["python"]\npass = 2\nupgrades = ["python/a.py"]\ndeps = ["M90.1"]\nused_by = ["M90.2"]\n',
            "M90.1": BASE["M90.1"].replace(
                'used_by = ["M90.2"]', 'used_by = ["M90.2", "M90.3"]'
            ),
        },
    )
    assert registry.invariants(reg) == []
    assert reg.unit_chain("python/a.py") == ["M90.1", "M90.3"]
    assert reg.superseded_by("M90.1") == {"python/a.py": "M90.3"}
    _write(
        tmp_path,
        "M90.4",
        'kind = "build"\nlang = ["python"]\npass = 2\nunits = ["python/a.py"]\nused_by = []\n',
    )
    errs = "\n".join(registry.invariants(registry.load(tmp_path)))
    assert "one owner, later ones use `upgrades`" in errs


def test_parse_errors(tmp_path):
    _write(tmp_path, "M90.1", 'kind = "bild"\nlang = ["cobol"]\npass = 1\n')
    with pytest.raises(HarnessError, match="kind 'bild'"):
        registry.load(tmp_path)
    (tmp_path / "modules" / "M90.1.toml").unlink()
    (tmp_path / "modules" / "zz.01.toml").write_text(
        'id = "zz.01"\ntitle = "t"\nkind = "build"\nlang = ["c"]\npass = 1\n'
    )
    with pytest.raises(HarnessError, match="course id grammar"):
        registry.load(tmp_path)


def test_tsv_round(tmp_path):
    reg = _reg(tmp_path)
    assert not registry.tsv_current(reg)
    registry.write_tsv(reg)
    assert registry.tsv_current(reg)
    rows = (tmp_path / "modules.tsv").read_text().splitlines()
    assert rows[1].split("\t")[:4] == ["M90.1", "build", "python", "1"]


def test_toml_writer_roundtrip():
    import tomllib

    doc = {
        "package": {"name": "x", "publish": False},
        "dependencies": {"a": {"path": "../a"}, "b": "1.0"},
        "workspace": {"members": ["a", "ol-tests"], "resolver": "2"},
    }
    assert tomllib.loads(tomlw.dumps(doc)) == doc
