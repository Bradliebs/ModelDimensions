"""Data sources for embedding experiments.

We make each source EXPLICIT and named, with no silent fallbacks between
sources. If you ask for wikitext and it isn't available, you get an error.

Available sources:
  - "wikitext"  : Wikipedia paragraphs via Salesforce/wikitext mirror
  - "ag_news"   : News articles (Parquet-backed; has syndication duplicates)
  - "imdb"      : Movie reviews (Parquet-backed; longer single-source texts)
  - "synthetic" : Templated sentences (no real semantics; offline fallback)

For audit experiments that need to track item provenance (which article a
paragraph came from), use `load_corpus_grouped(...)` instead of
`load_corpus(...)`. This returns texts with their source-article IDs so
splits can be made at the article level rather than the paragraph level.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Tuple


# --------- Public API ---------

def load_corpus(name: str, n: int = 2000, seed: int = 0,
                min_chars: int = 40, max_chars: int = 512) -> List[str]:
    """Load `n` texts from a named corpus. Raises if the source isn't available."""
    if name == "wikitext":
        return _load_wikitext(n, seed, min_chars, max_chars)
    elif name == "ag_news":
        return _load_ag_news(n, seed, max_chars)
    elif name == "imdb":
        return _load_imdb(n, seed, min_chars, max_chars)
    elif name == "synthetic":
        return _synthetic_texts(n, seed)
    else:
        raise ValueError(
            f"Unknown corpus: {name!r}. "
            "Use 'wikitext', 'ag_news', 'imdb', or 'synthetic'."
        )


def load_corpus_split(name: str, n_train: int, n_test: int,
                       seed: int = 0, **kwargs) -> Tuple[List[str], List[str]]:
    """Load a corpus and split into disjoint train/test sets at paragraph level."""
    total = n_train + n_test
    pool = load_corpus(name, n=total + 200, seed=seed, **kwargs)

    seen = set()
    unique = []
    for t in pool:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if len(unique) < total:
        raise RuntimeError(
            f"Corpus {name!r} produced only {len(unique)} unique items, "
            f"need {total}. Try a larger n or different corpus."
        )

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[:n_train], unique[n_train:n_train + n_test]


# --------- Article-aware loading for the leak audit ---------

@dataclass
class GroupedItem:
    """A text paragraph with provenance metadata."""
    text: str
    group_id: str   # e.g. article title for wikitext
    group_idx: int  # integer alias for group_id (stable for one load)


def load_corpus_grouped(name: str, n: int = 2000, seed: int = 0,
                         min_chars: int = 40, max_chars: int = 512
                         ) -> List[GroupedItem]:
    """Load texts WITH their group-of-origin label.

    Currently supported for: wikitext (groups = article titles).
    For other corpora this would need source-specific implementation.
    """
    if name == "wikitext":
        return _load_wikitext_grouped(n, seed, min_chars, max_chars)
    else:
        raise NotImplementedError(
            f"Grouped loading not implemented for {name!r}. "
            "Currently only wikitext supports article-level provenance."
        )


def article_level_split(items: List[GroupedItem], n_train: int, n_test: int,
                         seed: int = 0) -> Tuple[List[str], List[str], dict]:
    """Split paragraphs into train/test sets such that no article appears
    in both sides.

    Returns (train_texts, test_texts, stats) where stats includes:
      - n_train_articles, n_test_articles
      - n_train_paragraphs, n_test_paragraphs
      - article_overlap (should be 0)

    Raises if we can't find a partition meeting the size constraints.
    """
    # Group paragraphs by article
    by_article: dict[str, list[str]] = {}
    for it in items:
        by_article.setdefault(it.group_id, []).append(it.text)

    article_ids = sorted(by_article.keys())
    rng = random.Random(seed)
    rng.shuffle(article_ids)

    # Greedy fill: walk shuffled articles, assign each one entirely to
    # train (until n_train) or test (until n_test).
    train_texts: List[str] = []
    test_texts: List[str] = []
    train_articles: set[str] = set()
    test_articles: set[str] = set()

    for aid in article_ids:
        paragraphs = by_article[aid]
        if len(train_texts) < n_train:
            take = min(len(paragraphs), n_train - len(train_texts))
            train_texts.extend(paragraphs[:take])
            train_articles.add(aid)
        elif len(test_texts) < n_test:
            take = min(len(paragraphs), n_test - len(test_texts))
            test_texts.extend(paragraphs[:take])
            test_articles.add(aid)
        if len(train_texts) >= n_train and len(test_texts) >= n_test:
            break

    if len(train_texts) < n_train or len(test_texts) < n_test:
        raise RuntimeError(
            f"Could not gather enough paragraphs from disjoint articles: "
            f"got train={len(train_texts)}/{n_train}, "
            f"test={len(test_texts)}/{n_test}. "
            f"Total articles available: {len(article_ids)}. "
            "Try increasing the source pool size."
        )

    overlap = train_articles & test_articles
    assert len(overlap) == 0, f"Article overlap detected: {overlap}"

    stats = {
        "n_train_articles": len(train_articles),
        "n_test_articles": len(test_articles),
        "n_train_paragraphs": len(train_texts),
        "n_test_paragraphs": len(test_texts),
        "article_overlap": 0,
        "total_articles_in_pool": len(article_ids),
    }
    return train_texts[:n_train], test_texts[:n_test], stats


# --------- Source implementations ---------

_WIKITEXT_CANDIDATES = [
    ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
]

# Wikitext header line regex: " = Article Title = " at level 1 only (one space-bounded =)
# Level 2+ are subsections "== Section ==" which we treat as still inside the article.
_ARTICLE_HEADER_RE = re.compile(r"^\s*=\s[^=].*?\s=\s*$")


def _load_wikitext(n: int, seed: int, min_chars: int, max_chars: int
                   ) -> List[str]:
    """Standard paragraph-level wikitext loader (no provenance)."""
    from datasets import load_dataset
    last_err = None
    for repo, config in _WIKITEXT_CANDIDATES:
        try:
            ds = load_dataset(repo, config, split="train", streaming=True)
            texts: List[str] = []
            for row in ds:
                t = row["text"].strip()
                if len(t) > min_chars and not t.startswith("="):
                    texts.append(t[:max_chars])
                    if len(texts) >= max(n * 2, n + 500):
                        break
            if len(texts) < n:
                last_err = RuntimeError(
                    f"{repo}/{config} yielded only {len(texts)} usable "
                    f"texts (need {n})."
                )
                continue
            random.Random(seed).shuffle(texts)
            print(f"[data] wikitext loaded from {repo}/{config} "
                  f"({len(texts)} usable texts)")
            return texts[:n]
        except Exception as e:
            last_err = e
            print(f"[data] {repo}/{config} failed: {type(e).__name__}: {e}")
            continue
    raise RuntimeError(
        f"All wikitext mirrors failed. Last error: {last_err}. "
        "If you're on datasets >= 4.5.0, legacy script-based loaders "
        "are unsupported; only Parquet-backed mirrors work."
    )


def _load_wikitext_grouped(n: int, seed: int, min_chars: int, max_chars: int
                            ) -> List[GroupedItem]:
    """Article-aware wikitext loader.

    Walks the stream tracking the current article (from " = Title = " headers,
    level-1 only). Returns paragraphs tagged with their article-of-origin.

    Pulls enough raw rows to yield ~3x the requested n, so the downstream
    article-level split has room to satisfy size constraints.
    """
    from datasets import load_dataset
    last_err = None
    target_rows = max(n * 4, n + 2000)

    for repo, config in _WIKITEXT_CANDIDATES:
        try:
            ds = load_dataset(repo, config, split="train", streaming=True)
            items: List[GroupedItem] = []
            current_article = "<no_article>"
            article_idx_map: dict[str, int] = {}

            for row in ds:
                raw = row["text"]
                stripped = raw.strip()

                # Detect a top-level article header: " = Title = "
                if _ARTICLE_HEADER_RE.match(raw):
                    # Strip the leading/trailing = and whitespace
                    title = stripped.strip("= ").strip()
                    if title:
                        current_article = title
                        if title not in article_idx_map:
                            article_idx_map[title] = len(article_idx_map)
                    continue

                # Skip blank lines, subsection headers, other = lines
                if not stripped or stripped.startswith("="):
                    continue

                if len(stripped) > min_chars:
                    items.append(GroupedItem(
                        text=stripped[:max_chars],
                        group_id=current_article,
                        group_idx=article_idx_map.get(current_article, -1),
                    ))
                    if len(items) >= target_rows:
                        break

            if len(items) < n:
                last_err = RuntimeError(
                    f"{repo}/{config} grouped: yielded only {len(items)} "
                    f"paragraphs (need {n})."
                )
                continue

            # Drop the synthetic <no_article> bucket if it appeared
            items = [it for it in items if it.group_id != "<no_article>"]
            if len(items) < n:
                last_err = RuntimeError(
                    f"After dropping unattributed paragraphs: only {len(items)} left."
                )
                continue

            # Shuffle within reproducibility envelope
            random.Random(seed).shuffle(items)
            n_articles = len({it.group_id for it in items})
            print(f"[data] wikitext (grouped) loaded from {repo}/{config}: "
                  f"{len(items)} paragraphs across {n_articles} articles")
            return items[:n]
        except Exception as e:
            last_err = e
            print(f"[data] {repo}/{config} grouped failed: "
                  f"{type(e).__name__}: {e}")
            continue

    raise RuntimeError(
        f"All wikitext mirrors failed for grouped load. Last error: {last_err}."
    )


def _load_ag_news(n: int, seed: int, max_chars: int) -> List[str]:
    from datasets import load_dataset
    try:
        ds = load_dataset("ag_news", split="train")
    except Exception as e:
        print(f"[data] ag_news direct failed ({e}), trying fancyzhx/ag_news...")
        ds = load_dataset("fancyzhx/ag_news", split="train")
    available = min(n * 2, len(ds))
    texts = [row["text"][:max_chars] for row in ds.select(range(available))]
    if len(texts) < n:
        raise RuntimeError(
            f"ag_news yielded only {len(texts)} usable texts, need {n}."
        )
    random.Random(seed).shuffle(texts)
    return texts[:n]


def _load_imdb(n: int, seed: int, min_chars: int, max_chars: int) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("imdb", split="train", streaming=True)
    texts: List[str] = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > min_chars:
            texts.append(t[:max_chars])
            if len(texts) >= max(n * 2, n + 500):
                break
    if len(texts) < n:
        raise RuntimeError(
            f"imdb yielded only {len(texts)} usable texts, need {n}."
        )
    random.Random(seed).shuffle(texts)
    return texts[:n]


def _synthetic_texts(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    subjects = ["The cat", "A scientist", "The old man", "My neighbour",
                "The painter", "A traveller", "The detective", "A child",
                "The musician", "An engineer", "The librarian", "A pilot"]
    verbs = ["discovered", "questioned", "abandoned", "rebuilt", "celebrated",
             "translated", "calibrated", "memorized", "challenged", "designed"]
    objects = ["a hidden manuscript", "the broken mechanism",
               "an unfamiliar idea", "the abandoned garden",
               "a forgotten recipe", "the ancient instrument",
               "an impossible theorem", "the missing key", "a strange melody"]
    contexts = ["in winter", "before dawn", "under pressure",
                "on the third attempt", "with great care", "without warning",
                "for the first time"]
    out = []
    for _ in range(n):
        out.append(f"{rng.choice(subjects)} {rng.choice(verbs)} "
                   f"{rng.choice(objects)} {rng.choice(contexts)}.")
    return out


# --------- Legacy compat (exp01 uses this) ---------

def diverse_sample_texts(n: int = 2000, seed: int = 0) -> List[str]:
    """Legacy fallback-chain loader, kept so exp01 still runs unchanged."""
    for name in ("wikitext", "ag_news"):
        try:
            return load_corpus(name, n=n, seed=seed)
        except Exception as e:
            print(f"[data] {name} unavailable ({type(e).__name__}), trying next...")
    print("[data] All real corpora unavailable, using synthetic fallback.")
    return _synthetic_texts(n=n, seed=seed)
