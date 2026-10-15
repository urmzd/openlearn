"""ethics.01 artifact checks: your data licensing policy and allowlist.

Run by `ol check ethics.01` (through ./check) from the root of your repo,
which `check` passes as OL_REPO. Annotated exemplars (DESIGN 5.12): every
test says why it exists and fails with a message that says what to change.

The formats are the chapter's (content/tracks/responsible-ai/01-data-licensing/
01-data-licensing-and-the-ledger.md, section 4):

    docs/data/LICENSE_POLICY.md          the policy, in prose, with fixed `##` sections
    docs/data/license-allowlist.toml     the machine-readable allowlist data.08 enforces

This is an engineering check of a written decision, not legal advice: it
checks that every decision is recorded, uses SPDX identifiers, and respects
the few rules the course fixes (section 2.4), not that a lawyer would agree.
"""

from __future__ import annotations

import datetime
import os
import re
import tomllib
from pathlib import Path

REPO = Path(os.environ.get("OL_REPO", "."))
POLICY = REPO / "docs" / "data" / "LICENSE_POLICY.md"
ALLOWLIST = REPO / "docs" / "data" / "license-allowlist.toml"

SECTIONS = {  # heading -> minimum words of prose
    "Scope": 20,
    "Allowed uses": 20,
    "Ledger": 25,
    "Allowlist": 15,
    "Obligations": 15,
    "Unknown and missing licenses": 15,
    "Revocation": 20,
    "Owner and review": 8,
}
LEDGER_FIELDS = ("source_id", "license_spdx", "retrieved_at", "sha256", "allowed_uses")
USES = {"train", "eval"}
OBLIGATIONS = {"attribution", "share-alike", "notice", "no-endorsement"}
# SPDX License List identifiers the course expects to meet in data and code
# (spdx.org/licenses). An identifier not listed here must be written as
# LicenseRef-<name>, which SPDX reserves for licenses outside its list.
KNOWN_SPDX = set(
    """
    0BSD AFL-3.0 AGPL-3.0-only AGPL-3.0-or-later Apache-2.0 Artistic-2.0 BSD-2-Clause
    BSD-3-Clause BSL-1.0 CC-BY-1.0 CC-BY-2.0 CC-BY-2.5 CC-BY-3.0 CC-BY-4.0 CC-BY-NC-2.0
    CC-BY-NC-3.0 CC-BY-NC-4.0 CC-BY-NC-ND-3.0 CC-BY-NC-ND-4.0 CC-BY-NC-SA-2.0
    CC-BY-NC-SA-3.0 CC-BY-NC-SA-4.0 CC-BY-ND-3.0 CC-BY-ND-4.0 CC-BY-SA-2.0 CC-BY-SA-2.5
    CC-BY-SA-3.0 CC-BY-SA-4.0 CC-PDDC CC0-1.0 CDLA-Permissive-1.0 CDLA-Permissive-2.0
    CDLA-Sharing-1.0 EPL-2.0 EUPL-1.2 GFDL-1.3-only GFDL-1.3-or-later GPL-2.0-only
    GPL-2.0-or-later GPL-3.0-only GPL-3.0-or-later ISC LGPL-2.1-only LGPL-2.1-or-later
    LGPL-3.0-only LGPL-3.0-or-later MIT MPL-2.0 ODbL-1.0 ODC-By-1.0 OFL-1.1 PDDL-1.0
    Unlicense Zlib NOASSERTION NONE
    """.split()
)
REF = re.compile(r"^LicenseRef-[A-Za-z0-9.-]+$")
# The sources the course itself feeds the corpus: the TinyStories corpus for
# C1 (CDLA-Sharing-1.0) and the course's own synthetic fixtures (Apache-2.0).
COURSE_SOURCES = {
    "CDLA-Sharing-1.0": "TinyStories (C1)",
    "Apache-2.0": "the course fixtures",
}
# The review cases of section 4: every one must be decided, allowed or refused.
REVIEW = (
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-ND-4.0",
    "ODC-By-1.0",
    "MIT",
    "NOASSERTION",
)
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME|XXX)\b|<[a-z][^<>\n]*>")


# -- reading ------------------------------------------------------------------


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


def allowlist() -> dict:
    try:
        return tomllib.loads(ALLOWLIST.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise AssertionError(
            f"{ALLOWLIST.relative_to(REPO)} is not valid TOML: {e}"
        ) from None


def entries(kind: str) -> list[dict]:
    v = allowlist().get(kind, [])
    assert isinstance(v, list), f"`{kind}` must be an array of tables ([[{kind}]])"
    return v


def allowed() -> dict[str, dict]:
    return {e.get("spdx"): e for e in entries("allow")}


def refused() -> dict[str, dict]:
    return {e.get("spdx"): e for e in entries("refuse")}


# -- the policy document ----------------------------------------------------------


def test_policy_has_every_section():
    # WHY: a policy someone can act on answers fixed questions: what it
    #      covers, what uses it allows, what the ledger records, what is
    #      allowed, what each license obliges you to do, what happens to an
    #      unknown license, what happens when a license is withdrawn, and
    #      who owns the decision. A missing or one-line section is a
    #      question nobody answered.
    # KIND: unit
    text = POLICY.read_text(encoding="utf-8")
    sec = sections(text)
    missing = [h for h in SECTIONS if h not in sec]
    assert not missing, (
        f"LICENSE_POLICY.md has no `## {missing[0]}` section (needs: {', '.join(SECTIONS)})"
    )
    thin = [
        f"{h} ({words(sec[h])} of {n} words)"
        for h, n in SECTIONS.items()
        if words(sec[h]) < n
    ]
    assert not thin, f"sections too short to say anything: {', '.join(thin)}"


def test_policy_has_no_placeholders():
    # WHY: a template with "<your name>" or TODO left in it records no
    #      decision; it reads as if it did.
    # KIND: unit
    text = re.sub(r"`[^`\n]*`", "", POLICY.read_text(encoding="utf-8"))
    text = re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)
    hits = [m.group(0) for m in PLACEHOLDER.finditer(text)]
    assert not hits, f"LICENSE_POLICY.md still has template placeholders: {hits[:5]}"


def test_ledger_section_names_the_fields():
    # WHY: the ledger is how a license decision becomes checkable: the
    #      policy must say which recorded fields carry it (DESIGN 2.9,
    #      formats/ledger.schema.json), so a reader knows where to look.
    # KIND: unit
    body = sections(POLICY.read_text(encoding="utf-8")).get("Ledger", "")
    missing = [f for f in LEDGER_FIELDS if f not in body]
    assert not missing, (
        f"the Ledger section never names {missing}: say what each records"
    )


# -- the allowlist --------------------------------------------------------------------


def test_allowlist_has_an_owner_and_a_review_date():
    # WHY: a list nobody owns is never updated; a review date says how stale
    #      it is. `reviewed` is a TOML date (2026-10-09, no quotes), not
    #      later than today.
    # KIND: unit
    doc = allowlist()
    assert doc.get("version") == 1, "set `version = 1`"
    owner = doc.get("owner")
    assert isinstance(owner, str) and owner.strip(), (
        "set `owner` to the person or team who decides"
    )
    rev = doc.get("reviewed")
    assert isinstance(rev, datetime.date) and not isinstance(rev, datetime.datetime), (
        "`reviewed` must be a TOML date such as 2026-10-09 (no quotes)"
    )
    assert rev <= datetime.date.today(), "`reviewed` is in the future"
    extra = sorted(set(doc) - {"version", "owner", "reviewed", "allow", "refuse"})
    assert not extra, f"unknown top-level keys {extra}"


def test_every_entry_is_complete():
    # WHY: each decision carries its reason. An allow entry lists the uses
    #      (train, eval) and the obligations it brings; a refuse entry says
    #      why. Unknown keys are typos that data.08 would silently ignore.
    # KIND: unit
    for e in entries("allow"):
        assert set(e) <= {"spdx", "uses", "obligations", "why"}, (
            f"allow {e.get('spdx')}: unknown keys {sorted(set(e) - {'spdx', 'uses', 'obligations', 'why'})}"
        )
        uses = e.get("uses")
        assert isinstance(uses, list) and uses and set(uses) <= USES, (
            f"allow {e.get('spdx')}: `uses` must be a non-empty subset of {sorted(USES)}"
        )
        obl = e.get("obligations")
        assert isinstance(obl, list) and set(obl) <= OBLIGATIONS, (
            f"allow {e.get('spdx')}: `obligations` must be a list drawn from {sorted(OBLIGATIONS)}"
        )
        assert words(str(e.get("why", ""))) >= 6, (
            f"allow {e.get('spdx')}: `why` needs a sentence"
        )
    for e in entries("refuse"):
        assert set(e) <= {"spdx", "why"}, f"refuse {e.get('spdx')}: unknown keys"
        assert words(str(e.get("why", ""))) >= 6, (
            f"refuse {e.get('spdx')}: `why` needs a sentence"
        )


def test_every_id_is_an_spdx_identifier():
    # WHY: the ledger, the allowlist, and the datasheet must agree on names.
    #      "CC BY 4.0", "Apache 2.0", and "cc-by-4.0" are not the SPDX
    #      identifiers CC-BY-4.0 and Apache-2.0, and an exact-match check
    #      in data.08 would refuse them. A license SPDX does not list is
    #      written LicenseRef-<name>.
    # KIND: boundary
    for e in entries("allow") + entries("refuse"):
        s = e.get("spdx")
        assert isinstance(s, str), f"an entry has no `spdx`: {e}"
        assert s in KNOWN_SPDX or REF.match(s), (
            f"{s!r} is not an SPDX identifier the course knows; check the exact spelling at "
            "spdx.org/licenses, or write LicenseRef-<name> for a license SPDX does not list"
        )


def test_no_license_is_both_allowed_and_refused():
    # WHY: one license, one decision. Listing an id twice, or in both
    #      tables, leaves the answer to whichever line data.08 reads last.
    # KIND: boundary
    a = [e.get("spdx") for e in entries("allow")]
    r = [e.get("spdx") for e in entries("refuse")]
    dup = sorted({x for x in a + r if (a + r).count(x) > 1})
    assert not dup, f"listed more than once: {dup}"


def test_the_course_sources_are_allowed_for_training():
    # WHY: your corpus is built from these sources (TinyStories in C1, the
    #      course fixtures from Pass 3). If your own policy refuses them,
    #      data.08 refuses your corpus and dur.12 refuses your model.
    # KIND: unit
    a = allowed()
    for spdx, what in COURSE_SOURCES.items():
        assert spdx in a and "train" in a[spdx].get("uses", []), (
            f"{spdx} ({what}) must be allowed for train; if you decide otherwise, the course corpus cannot be used"
        )


def test_every_review_case_is_decided():
    # WHY: the section 4 review: seven licenses you will meet on dataset
    #      cards. Each one is allowed (with uses and obligations) or refused
    #      (with a reason); "not decided yet" is how unlicensed data slips in.
    # KIND: unit
    decided = set(allowed()) | set(refused())
    missing = [s for s in REVIEW if s not in decided]
    assert not missing, (
        f"no decision for {missing}: add each to [[allow]] or [[refuse]]"
    )


def test_non_commercial_and_no_derivatives_are_never_trained_on():
    # WHY: the course rule of section 2.4. Your model's weights are released
    #      (dur.12) under a license you choose; an NC term forbids commercial
    #      use of what you build from the data and an ND term forbids
    #      adaptations, so neither may be in a training set whose model you
    #      release. Evaluation use is a separate decision.
    # KIND: boundary
    bad = [
        s
        for s, e in allowed().items()
        if re.search(r"-(NC|ND)\b|-NC-|-ND-", s or "") and "train" in e.get("uses", [])
    ]
    assert not bad, f"allowed for train despite NC or ND terms: {bad}"


def test_unknown_licenses_are_refused():
    # WHY: NOASSERTION ("nobody said") and NONE ("no license", so all rights
    #      reserved) are where provenance is missing; the default answer is
    #      no. They must be refused explicitly, so the policy says so.
    # KIND: boundary
    a, r = allowed(), refused()
    assert "NOASSERTION" in r and "NOASSERTION" not in a, (
        "refuse NOASSERTION (a source whose license nobody recorded)"
    )
    assert "NONE" not in a, "NONE means all rights reserved: it cannot be allowed"


def test_obligations_follow_the_license_family():
    # WHY: what a license makes you do travels with the data: the CC BY
    #      family and ODC-By are attribution licenses by name, and every
    #      ShareAlike license (CC *-SA-*, CDLA-Sharing) asks that shared
    #      derived data keep the same terms. The datasheet (ethics.03) lists
    #      them from here.
    # KIND: unit
    for s, e in allowed().items():
        obl = set(e.get("obligations", []))
        if re.match(r"^(CC-BY|ODC-By)", s or ""):
            assert "attribution" in obl, f"{s}: record the `attribution` obligation"
        if re.search(r"-SA-|^CDLA-Sharing", s or ""):
            assert "share-alike" in obl, f"{s}: record the `share-alike` obligation"
