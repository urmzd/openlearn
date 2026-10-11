"""ethics.03 artifact checks: your model card, datasheet, and third-party
model card.

Run by `ol check ethics.03` (through ./check) from the root of your repo,
which `check` passes as OL_REPO. Annotated exemplars (DESIGN 5.12): every
test says why it exists and fails with a message that says what to change.

The files and their sections are the chapter's (responsible-ai/
03-model-and-data-cards/01-datasheet-and-model-card.md, section 4):

    docs/MODEL_CARD.md                       from contracts/templates/MODEL_CARD.md
    docs/DATASHEET.md                        from contracts/templates/DATASHEET.md
    docs/models/smollm2-135m-instruct.md     the third-party model, labelled as such

A check can tell that a card is complete, specific, and consistent with
your other records; it cannot tell that it is true. That part is yours.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

REPO = Path(os.environ.get("OL_REPO", "."))
COURSE = Path(os.environ.get("OL_COURSE_TREE", Path(__file__).resolve().parents[2]))
CARD = REPO / "docs" / "MODEL_CARD.md"
SHEET = REPO / "docs" / "DATASHEET.md"
THIRD = REPO / "docs" / "models" / "smollm2-135m-instruct.md"
ALLOWLIST = REPO / "docs" / "data" / "license-allowlist.toml"

CARD_SECTIONS = {  # heading -> minimum words of prose
    "Model details": 40,
    "Intended use": 25,
    "Evaluation": 30,
    "Bias, risks, and limitations": 40,
    "Data": 25,
}
SHEET_SECTIONS = {
    "Motivation": 15,
    "Composition": 30,
    "Collection": 20,
    "Preprocessing": 25,
    "Uses": 10,
    "Distribution and maintenance": 15,
}
THIRD_SECTIONS = {
    "Provenance": 25,
    "License": 5,
    "Intended use": 10,
    "Evaluation": 10,
    "Limitations": 15,
}
DETAILS = (
    "Developer",
    "Architecture",
    "Training",
    "Release",
    "Third-party components",
    "License",
)
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|<[a-z][^<>\n]*>")
INTERVAL = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)
SPDX = re.compile(
    r"\b(Apache-2\.0|MIT|BSD-[23]-Clause|CC0-1\.0|CC-BY(?:-SA)?-4\.0|ODC-By-1\.0|CDLA-(?:Sharing|Permissive)-[12]\.0|LicenseRef-[A-Za-z0-9.-]+)\b"
)
UPSTREAM = "HuggingFaceTB/SmolLM2-135M-Instruct"


def sections(text: str) -> dict[str, str]:
    """`## heading` -> body, outside code fences."""
    out: dict[str, str] = {}
    cur, buf, fence = None, [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^## \S", line):
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = line[3:].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def words(body: str) -> int:
    body = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", body))


def need_sections(path: Path, want: dict[str, int]) -> dict[str, str]:
    sec = sections(path.read_text(encoding="utf-8"))
    name = path.relative_to(REPO)
    missing = [h for h in want if h not in sec]
    assert not missing, (
        f"{name} has no `## {missing[0]}` section (needs: {', '.join(want)})"
    )
    thin = [
        f"{h} ({words(sec[h])} of {n} words)"
        for h, n in want.items()
        if words(sec[h]) < n
    ]
    assert not thin, f"{name}: sections too short to say anything: {', '.join(thin)}"
    return sec


def table(body: str) -> list[list[str]]:
    rows = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", s):
            rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def bullet(body: str, label: str) -> str:
    m = re.search(
        r"^\s*[-*]\s+\*\*"
        + re.escape(label)
        + r":?\*\*:?\s*(.+?)(?=^\s*[-*]\s+\*\*|\Z)",
        body,
        re.M | re.S,
    )
    return " ".join(m.group(1).split()) if m else ""


# -- the model card ------------------------------------------------------------------------


def test_model_card_sections():
    # WHY: Mitchell et al.'s card answers fixed questions: what the model
    #      is, what it is for and not for, how it was evaluated, what it gets
    #      wrong, and what it was trained on. The template's five sections,
    #      each with enough prose to say something, and a title naming the
    #      model id and version the release gate (dur.12) looks for.
    # KIND: unit
    text = CARD.read_text(encoding="utf-8")
    need_sections(CARD, CARD_SECTIONS)
    first = next((x for x in text.splitlines() if x.startswith("# ")), "")
    assert re.fullmatch(r"# Model card: [a-z0-9][a-z0-9._-]* v\d+(\.\d+)*", first), (
        f"MODEL_CARD.md's title must be `# Model card: <model_id> v<N>`, got {first!r}"
    )


def test_no_template_placeholders():
    # WHY: a card with `<you>` or TODO left in it was copied, not written. The
    #      release gate checks presence; this checks that every field of all
    #      three documents was filled in.
    # KIND: unit
    for path in (CARD, SHEET, THIRD):
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        for i, line in enumerate(text.splitlines(), 1):
            m = PLACEHOLDER.search(line)
            assert not m, (
                f"{path.relative_to(REPO)}:{i}: template placeholder {m.group(0)!r} left in"
            )


def test_model_details_fields():
    # WHY: the details a reader needs before anything else: who built it, the
    #      architecture, how it was trained, where the release lives, what in
    #      it is not yours, and its license as an SPDX identifier.
    # KIND: unit
    body = sections(CARD.read_text(encoding="utf-8"))["Model details"]
    missing = [d for d in DETAILS if not bullet(body, d)]
    assert not missing, (
        f"Model details needs a `- **{missing[0]}:** ...` bullet (all of: {', '.join(DETAILS)})"
    )
    lic = bullet(body, "License")
    assert SPDX.search(lic), (
        f"the License bullet must name an SPDX id (Apache-2.0, MIT, ...), got {lic!r}"
    )


def test_intended_and_out_of_scope_uses():
    # WHY: the out-of-scope list is the part of a card that prevents harm: it
    #      is what the usage policy (ethics.05) enforces at the gateway. Both
    #      lists must say something specific.
    # KIND: unit
    body = sections(CARD.read_text(encoding="utf-8"))["Intended use"]
    for label in ("Primary uses", "Out of scope"):
        b = bullet(body, label)
        assert words(b) >= 8, (
            f"Intended use needs a `- **{label}:**` bullet of at least 8 words, got {b!r}"
        )


def test_evaluation_table_has_intervals():
    # WHY: every reported number carries its uncertainty, and the card covers
    #      quality, safety, and bias, the three suites the release runs
    #      (ethics.04, L6.7). Each value is written `value (lo, hi)` with
    #      lo <= value <= hi, and each row says what threshold it was held to.
    # KIND: unit
    body = sections(CARD.read_text(encoding="utf-8"))["Evaluation"]
    rows = table(body)
    assert rows and [c.lower() for c in rows[0][:4]] == [
        "suite",
        "metric",
        "value (95% ci)",
        "release threshold",
    ], (
        "Evaluation needs the template's table: | Suite | Metric | Value (95% CI) | Release threshold |"
    )
    data = [r for r in rows[1:] if len(r) >= 4]
    assert len(data) >= 3, (
        f"Evaluation lists {len(data)} rows; report at least quality, safety, and bias"
    )
    suites = {r[0].strip("`").lower() for r in data}
    for s in ("safety", "bias"):
        assert s in suites, f"Evaluation has no `{s}` row (the ethics.04 suites)"
    assert any(re.search(r"\b(bpb|ppl|loss|perplexity)\b", r[1], re.I) for r in data), (
        "Evaluation has no quality row: bits per byte, perplexity, or loss"
    )
    for r in data:
        m = INTERVAL.search(r[2])
        assert m, (
            f"row {r[0]} / {r[1]}: write the value as `value (lo, hi)`, got {r[2]!r}"
        )
        v, lo, hi = (float(x) for x in m.groups())
        assert lo <= v <= hi, (
            f"row {r[0]} / {r[1]}: the interval ({lo}, {hi}) does not contain {v}"
        )
        assert words(r[3]) >= 1, (
            f"row {r[0]} / {r[1]}: say what threshold the release held it to"
        )


def test_limitations_cite_measurements():
    # WHY: "the model may be biased" is not a limitation statement. The bias
    #      section must cite at least one measured result with its interval and
    #      talk about both bias and safety.
    # KIND: unit
    body = sections(CARD.read_text(encoding="utf-8"))["Bias, risks, and limitations"]
    assert INTERVAL.search(body), (
        "Bias, risks, and limitations must cite a measured value with its interval: `x (lo, hi)`"
    )
    low = body.lower()
    assert "bias" in low or "gap" in low or "stereotype" in low, (
        "Bias, risks, and limitations says nothing about bias"
    )
    assert any(w in low for w in ("toxic", "refus", "safety", "harm")), (
        "Bias, risks, and limitations says nothing about safety"
    )


def test_card_links_the_datasheet():
    # WHY: the card's Data section points at the datasheet that documents the
    #      corpus, and names the licenses involved, so a reader can follow the
    #      model back to its sources.
    # KIND: unit
    body = sections(CARD.read_text(encoding="utf-8"))["Data"]
    assert re.search(r"\]\((?:\./)?DATASHEET\.md\)", body), (
        "the Data section must link [DATASHEET.md](DATASHEET.md)"
    )
    assert SPDX.search(body), (
        "the Data section must name the license of the training data (an SPDX id)"
    )


# -- the datasheet ------------------------------------------------------------------------------


def test_datasheet_sections():
    # WHY: Gebru et al.'s questions, in order: why the data exists, what is in
    #      it (including personal data), where it came from, what was done to
    #      it, what it may be used for, and who maintains it.
    # KIND: unit
    sec = need_sections(SHEET, SHEET_SECTIONS)
    first = next(
        (
            x
            for x in SHEET.read_text(encoding="utf-8").splitlines()
            if x.startswith("# ")
        ),
        "",
    )
    assert first.startswith("# Datasheet: "), (
        f"DATASHEET.md's title must be `# Datasheet: <dataset> <version>`, got {first!r}"
    )
    comp = sec["Composition"].lower()
    assert "personal" in comp or "pii" in comp, (
        "Composition must say what personal data the dataset holds (the PII scrub's findings)"
    )
    pre = sec["Preprocessing"].lower()
    assert "dedup" in pre or "duplicat" in pre, (
        "Preprocessing must say how duplicates were handled"
    )
    assert "contamina" in pre, (
        "Preprocessing must say how the eval sets were kept out (decontamination)"
    )


def test_datasheet_sources_are_allowed():
    # WHY: every source the datasheet lists carries a license your ethics.01
    #      allowlist permits for training. A source under a refused or unknown
    #      license is a release blocker (dur.12, data.08), so it is one here.
    #      The verdict is your data.08 permitted_uses over the allowlist as an
    #      {spdx: uses} mapping: the rule data.08's verify applies to the
    #      ledger rows the datasheet was generated from.
    # KIND: unit
    from corpus.ledger import permitted_uses

    assert ALLOWLIST.is_file(), (
        "docs/data/license-allowlist.toml is missing: finish ethics.01 first"
    )
    allow = tomllib.loads(ALLOWLIST.read_text(encoding="utf-8")).get("allow", [])
    policy = {e.get("spdx"): e.get("uses", []) for e in allow if e.get("spdx")}
    body = sections(SHEET.read_text(encoding="utf-8"))["Collection"]
    found = SPDX.findall(body)
    assert found, "Collection must give each source's license as an SPDX id"
    bad = sorted(
        lic for lic in set(found) if "train" not in (permitted_uses(lic, policy) or ())
    )
    assert not bad, (
        f"Collection lists {', '.join(bad)}, which your allowlist does not allow for train"
    )


# -- the third-party card ------------------------------------------------------------------------


def pinned_revision() -> str:
    rows = (COURSE / "fixtures" / "ASSETS.tsv").read_text(encoding="utf-8").splitlines()
    for line in rows:
        f = line.split("\t")
        if f[0] == "smollm2-135m-instruct" and len(f) >= 7:
            return f[6].split("@", 1)[1]
    raise AssertionError(
        "smollm2-135m-instruct is not pinned in the course's fixtures/ASSETS.tsv"
    )


def test_third_party_model_labelled():
    # WHY: a model you serve but did not train must say so, name its exact
    #      upstream (repository and the revision ol fetch pinned), keep its own
    #      license (Apache-2.0), and report only what you measured yourself.
    # KIND: unit
    sec = need_sections(THIRD, THIRD_SECTIONS)
    text = THIRD.read_text(encoding="utf-8")
    assert UPSTREAM in sec["Provenance"], (
        f"Provenance must name the upstream repository {UPSTREAM}"
    )
    rev = pinned_revision()
    assert rev in sec["Provenance"], (
        f"Provenance must give the pinned revision {rev} (course fixtures/ASSETS.tsv)"
    )
    assert "Apache-2.0" in sec["License"], (
        "the License section must say Apache-2.0, the upstream license"
    )
    low = text.lower()
    assert "third-party" in low or "did not train" in low, (
        "say plainly that this is a third-party model you did not train"
    )
