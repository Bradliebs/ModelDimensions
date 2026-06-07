"""Offline tests for the BM25 lexical index.

Pure-Python; uses only ``rank_bm25`` and the lexical_index wrapper.
No bank or encoder is loaded — the index is corpus-agnostic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.lexical_index import (
    DEFAULT_B,
    DEFAULT_K1,
    LexicalIndex,
    LexicalIndexStale,
    TOKENIZER_VERSION,
    tokenize,
)


# ---- tokenizer ----

def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The quick brown fox jumps over the lazy dog") == [
        "quick", "brown", "fox", "jumps", "lazy", "dog",
    ]


def test_tokenize_keeps_proper_nouns_and_digits():
    assert tokenize("Don Ellis scored The French Connection in 1971") == [
        "don", "ellis", "scored", "french", "connection", "1971",
    ]


def test_tokenize_drops_length_one_and_empty():
    assert tokenize("a I 9 of go") == ["go"]
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


# ---- build + query ----

@pytest.fixture
def tiny_corpus() -> tuple[list[int], list[str]]:
    """5-doc corpus that exercises the BM25 lexical signal: proper nouns
    that the dense encoder is known to fold (per exp27 Q2). One doc
    mentions Don Ellis and The French Connection, one is generic film
    noise, the rest are unrelated."""
    cell_ids = [100, 101, 102, 103, 104]
    texts = [
        "Don Ellis composed the music for The French Connection in 1971.",
        "The Godfather is a 1972 American crime film directed by Coppola.",
        "Zebras have black and white stripes and live in Africa.",
        "Paris is the capital of France and sits on the river Seine.",
        "Chess is a two-player strategy game played on 64 squares.",
    ]
    return cell_ids, texts


def test_build_and_topk_returns_keyword_doc_first(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    ids, scores = idx.topk("who composed The French Connection", k=3)
    assert ids[0] == 100
    assert scores[0] > 0.0
    # The Godfather doc shares "the" and "film" but no proper nouns; it
    # may or may not appear. Only the requirement is that Don Ellis wins.
    assert all(s >= 0.0 for s in scores)


def test_topk_drops_zero_score_docs(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    # Query with terms in only the zebra doc; the chess and Paris docs
    # must not be returned even if k is large.
    ids, scores = idx.topk("zebras stripes Africa", k=5)
    assert 102 in ids
    assert all(s > 0.0 for s in scores)


def test_empty_query_returns_empty(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    ids, scores = idx.topk("", k=3)
    assert ids.shape == (0,) and scores.shape == (0,)
    ids, scores = idx.topk("the a an of", k=3)
    assert ids.shape == (0,) and scores.shape == (0,)


def test_excluded_ids_are_masked(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    # Without exclusion: 100 wins.
    ids, _ = idx.topk("Don Ellis French Connection", k=3)
    assert ids[0] == 100
    # Mask 100 (simulating a post-build tombstone): 100 must not appear.
    ids, _ = idx.topk(
        "Don Ellis French Connection", k=3, excluded_ids={100}
    )
    assert 100 not in ids


def test_build_rejects_mismatched_lengths(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    with pytest.raises(ValueError, match="parallel"):
        idx.build_from_texts(cell_ids, texts[:-1])


def test_build_rejects_empty_corpus():
    idx = LexicalIndex()
    with pytest.raises(ValueError, match="empty"):
        idx.build_from_texts([], [])


def test_build_handles_all_empty_texts():
    # Cell ids present but no text — index should build (sentinel injected)
    # and return zero hits for any real query.
    idx = LexicalIndex()
    idx.build_from_texts([1, 2, 3], [None, "", ""])
    ids, scores = idx.topk("anything", k=2)
    assert ids.shape == (0,)


def test_topk_preserves_order_descending(tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    _, scores = idx.topk("composed music film Connection", k=4)
    assert list(scores) == sorted(scores, reverse=True)


# ---- persistence ----

def test_save_and_load_roundtrip(tmp_path: Path, tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts, source="test")
    idx.save(tmp_path / "lex")

    manifest = json.loads((tmp_path / "lex" / "manifest.json").read_text())
    assert manifest["n_docs"] == 5
    assert manifest["tokenizer_version"] == TOKENIZER_VERSION
    assert manifest["k1"] == DEFAULT_K1
    assert manifest["b"] == DEFAULT_B
    assert manifest["source"] == "test"

    reloaded = LexicalIndex.load(tmp_path / "lex")
    ids_a, scores_a = idx.topk("Don Ellis French Connection", k=3)
    ids_b, scores_b = reloaded.topk("Don Ellis French Connection", k=3)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(scores_a, scores_b, rtol=1e-6)


def test_load_rejects_stale_corpus_hash(tmp_path: Path, tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    idx.save(tmp_path / "lex")
    with pytest.raises(LexicalIndexStale, match="corpus hash"):
        LexicalIndex.load(
            tmp_path / "lex", expected_corpus_hash="0" * 64,
        )


def test_load_rejects_stale_tokenizer_version(tmp_path: Path, tiny_corpus):
    cell_ids, texts = tiny_corpus
    idx = LexicalIndex()
    idx.build_from_texts(cell_ids, texts)
    idx.save(tmp_path / "lex")
    # Mutate the manifest to simulate a tokenizer-version drift.
    mpath = tmp_path / "lex" / "manifest.json"
    m = json.loads(mpath.read_text())
    m["tokenizer_version"] = "v0.fake"
    mpath.write_text(json.dumps(m))
    with pytest.raises(LexicalIndexStale, match="tokenizer version"):
        LexicalIndex.load(tmp_path / "lex")
