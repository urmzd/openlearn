"""ref-thresholds (DESIGN 5.7, 5.11): `calibrated` json-last-line metrics are
held to the reference mean + 3 sd over 5 seeds; --record-thresholds writes the
rows (and their MANIFEST row); --smoke milestones (C1's 200-step config) use
`smoke_vars` and their own smoke thresholds; learning tests use
_lib.thresholds through `ol verify course --record-thresholds`."""

TRAINER = """import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int)
ap.add_argument("--seed", type=int)
ap.add_argument("--worse", type=float, default=0.0)
a = ap.parse_args()
loss = 2.0 + 0.001 * a.seed + (0.5 if a.steps < 1000 else 0.0) + WORSE
print(json.dumps({"loss": loss, "steps": a.steps}))
"""


def _setup(ss, worse: float = 0.0):
    ss.add_extras("thresholds")
    ss.init()
    ss.install_system()
    (ss.learner / "train.py").write_text(TRAINER.replace("WORSE", repr(worse)))
    st = ss.learner / "system.toml"
    st.write_text(
        st.read_text().replace("[entry]\n", '[entry]\nctl = ["python3", "train.py"]\n')
    )
    ss.commit_learner("feat: add the trainer")


def test_calibrated_metric_needs_recorded_thresholds(ss):
    _setup(ss)
    out = ss("milestone", "MS-T90", rc=5).out
    assert "no reference threshold for MS-T90/train loss (full)" in out
    out = ss("milestone", "MS-T90", "--record-thresholds", rc=0).out
    assert "MS-T90/train loss (full): mean 2.002" in out and "over 5 seeds" in out
    table = (ss.course / "fixtures/ref-thresholds.tsv").read_text()
    assert (
        "MS-T90/train\tloss\tmax\tfull\t2.002\t" in table
        and "2,2.001,2.002,2.003,2.004" in table
    )
    man = (ss.course / "fixtures/MANIFEST.tsv").read_text()
    assert "course/fixtures/ref-thresholds.tsv\t" in man
    out = ss("milestone", "MS-T90", rc=0).out
    assert "loss: 2 <= 2.00674" in out
    # the smoke run is calibrated apart: 200 steps (smoke_vars) lose 0.5 more
    assert "(smoke)" in ss("milestone", "MS-T90", "--smoke", rc=5).out
    ss("milestone", "MS-T90", "--smoke", "--record-thresholds", rc=0)
    out = ss("milestone", "MS-T90", "--smoke", rc=0).out
    assert "loss: 2.5 <= 2.50674" in out


def test_a_worse_learner_fails_the_calibrated_bar(ss):
    _setup(ss)
    ss("milestone", "MS-T90", "--record-thresholds", rc=0)
    (ss.learner / "train.py").write_text(TRAINER.replace("WORSE", "0.05"))
    out = ss("milestone", "MS-T90", rc=1).out
    assert "loss: 2.05 > 2.00674" in out and "FAIL" in out


def test_learning_tests_record_and_check(ss):
    t = ss.course / "tests/M90.1/test_learning.py"
    t.write_text(
        "import os\n\nfrom _lib.thresholds import check\nfrom tinyllm.demo.scale import scale\n\n\n"
        "def test_learns_a_scale():\n"
        "    # WHY: a toy 'learning' metric that scatters with the seed.\n"
        "    # KIND: learning\n"
        "    seed = int(os.environ.get('OL_SEED', '0'))\n"
        "    loss = sum(scale([0.1, 0.2], 1.0 + 0.01 * seed))\n"
        "    check('M90.1/test_learns_a_scale', 'loss', loss, direction='max')\n"
    )
    out = ss("verify", "course", "M90.1", "--record-thresholds", rc=0).out
    assert "M90.1/test_learns_a_scale loss: mean 0.306" in out
    rows = (ss.course / "fixtures/ref-thresholds.tsv").read_text()
    assert "M90.1/test_learns_a_scale\tloss\tmax\tfull" in rows
    ss.init()
    ss("start", "M90.1", rc=0)
    ss.implement("python/tinyllm/demo/scale.py", owner="M90.1")
    assert "PASS M90.1" in ss("check", "M90.1", rc=0).out
    unit = ss.learner / "python/tinyllm/demo/scale.py"
    unit.write_text(
        unit.read_text().replace(
            "return [k * x for x in xs]", "return [k * x * 1.5 for x in xs]"
        )
    )
    out = ss("check", "M90.1", rc=1).out
    assert "M90.1/test_learns_a_scale loss" in out and "over 5 seeds" in out
