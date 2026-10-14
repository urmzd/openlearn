# contracts/py/corpus/ledger.pyi (data.08): licensing ledger verification
# and the datasheet
# chapter: data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md
#
# The ledger is corpus/LEDGER.jsonl, one formats/ledger.schema.json object
# per line, append-only: data.01 appends a row per fetched source, and
# reconcile() appends corrected rows. The LAST row of a source_id is its
# current state.
#
# Licenses. ALLOWLIST maps an SPDX license id to the uses it permits
# ("train", "eval"); the policy is ethics.01's. An expression joins ids
# with OR (the licensee may pick: the union of the uses) or AND (both
# apply: the intersection), evaluated left to right with AND binding
# tighter; anything else (parentheses, WITH, an id outside ALLOWLIST) is
# unknown.
#
# Exit code. A failed verification is EX_DATAERR = 65: non-retryable under
# spec/subprocess-activity.md, so a CorpusBuild with a bad license fails
# once instead of retrying forever (the CLI exits with LedgerError.exit_code).
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

EX_DATAERR: int  # 65
ALLOWLIST: dict[str, frozenset[str]]

class LedgerError(Exception):
    exit_code: int  # EX_DATAERR
    problems: list[str]

    def __init__(self, problems: list[str]) -> None:
        """str(error) lists every problem, one per line."""

def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every row in file order. ValueError naming the line for a line that
    is not a JSON object (blank lines are skipped)."""

def latest(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """source_id -> its last row."""

def row_errors(row: Mapping[str, Any]) -> list[str]:
    """Violations of formats/ledger.schema.json by one row: a missing or
    extra key, a wrong type, a source_id or sha256 that does not match its
    pattern, a negative count, an allowed_uses or pii_policy value outside
    its enum. Optional modality, biometric, consent, attribution and
    prohibited_uses fields follow the multimodal ledger extension.
    [] for a valid row."""

def permitted_uses(
    license_spdx: str, allowlist: Mapping[str, Iterable[str]] = ...
) -> Optional[frozenset[str]]:
    """The uses an SPDX expression permits, or None when it is unknown."""

def verify(
    ledger: Path,
    corpus: Path,
    *,
    use: str = "train",
    allowlist: Mapping[str, Iterable[str]] = ...,
) -> list[str]:
    """Every problem, [] when the corpus at `corpus` (a data.06 output
    directory) may be used for `use`:
      - a ledger row that breaks the schema;
      - a source whose license is unknown, whose allowed_uses claims a use
        its license does not permit, or that has shard rows and is revoked
        or lacks `use`;
      - a shard row whose source_id has no ledger row, or whose
        license_spdx differs from its source's;
      - a source whose kept count differs from its rows in the shards."""

def check(
    ledger: Path,
    corpus: Path,
    *,
    use: str = "train",
    allowlist: Mapping[str, Iterable[str]] = ...,
) -> None:
    """verify(), raising LedgerError(problems) when there is any."""

def reconcile(ledger: Path, corpus: Path, *, filters_applied: Sequence[str] = ()) -> list[dict[str, Any]]:
    """For every source with a ledger row, append a copy of its latest row
    with kept = its rows in the shards, dropped = max(0, n_docs - kept), and
    filters_applied; return the appended rows (sorted by source_id). Rows
    are appended with one write, each json.dumps(row, ensure_ascii=False)
    plus "\\n"."""

def datasheet(ledger: Path, corpus: Path, tokens: Optional[Path] = None) -> str:
    """contracts/templates/DATASHEET.md with the counts filled in: the
    title "# Datasheet: <dataset> <version>" and the six "## " sections of
    the template in order. Composition gives n_docs, the train and val row
    counts, the languages, and the PII counts by type; Collection lists
    every source (source_id, url, license_spdx, retrieved_at, sha256);
    Preprocessing gives each filter count, exact and near dedup counts with
    jaccard_threshold, num_perm, bands, the decontaminated count with ngram,
    and, when `tokens` (a data.07 output directory) is given, the
    tokenizer_id and the token count of each split; Uses gives each
    source's allowed_uses. Motivation keeps the template's prompt for your
    own prose."""
