"""The PDF lists the concepts each section borrows, as the website does."""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemble_book import concept_lists, concepts_block, with_concepts
from course_catalog import ROOT

PAGE = "content/courses/ai-engineering/09-recurrent-networks/02-lstm.md"
CONCEPT = {
    "label": "Orthogonal matrix",
    "definition": "a square $Q$ with $Q^\\top Q = I$; see [QR](qr.md).",
    "home": "content/courses/mathematics/09-linear-algebra/03-qr.md",
    "homeTitle": "9.3 Orthogonality",
    "book": "mathematics",
    "bookTitle": "Mathematics",
}


def test_block_names_where_each_concept_is_taught():
    block = concepts_block(PAGE, [CONCEPT])
    assert block.startswith("> **Concepts from other sections.**")
    # Links in a borrowed definition are relative to another file: keep the text.
    assert (
        "> - **Orthogonal matrix**: a square $Q$ with $Q^\\top Q = I$; see QR. "
        "(See 9.3 Orthogonality, *Mathematics*.)" in block
    )
    same_book = dict(CONCEPT, book="ai-engineering")
    assert "(See 9.3 Orthogonality.)" in concepts_block(PAGE, [same_book])


def test_block_goes_after_the_title_and_opening_paragraph():
    text = "## 9.2 LSTM\n\nOpening line one.\nline two.\n\nMore."
    out = with_concepts(text, "> BOX")
    assert out.index("line two.") < out.index("> BOX") < out.index("More.")
    code_first = "## 9.2 LSTM\n\n```\ncode\n```"
    assert with_concepts(code_first, "> BOX").startswith("## 9.2 LSTM\n\n> BOX")


@pytest.mark.skipif(not shutil.which("node"), reason="needs node")
def test_pdf_and_website_share_one_matcher():
    lists = concept_lists(ROOT, "ai-engineering")
    assert PAGE in lists
    labels = {c["label"] for c in lists[PAGE]}
    assert "Orthogonal matrix" in labels
    assert all(c["home"] != PAGE for c in lists[PAGE])
