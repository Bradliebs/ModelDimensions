"""V1 answer pipeline: question in, grounded answer or honest silence out.

Wires together the four pieces validated in Steps 1-2.5:

    question
      → EncoderSingleton.encode_one  (MiniLM, query-side prefix if any)
      → StreamingBank.whiten + topk  (k retrieved cells, descending activation)
      → v1_silence_gate.decide       (fire iff top1-top2 >= 0.05)
      → Phi-3 generate               (only if gate fires)
      → v1_answer_verifier.verify    (token coverage >= 0.50 against cells)
      → grounded answer + citations, OR honest-silence string

The generator is injectable so tests can avoid loading 2.5 GB of weights:
pass ``generator=<callable str -> str>`` to the constructor and the
default Phi-3 path is skipped entirely.

This module is *the* V1 contract. The CLI in ``scripts/ask.py`` is a
thin shell over it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from src.agent import v1_answer_verifier, v1_silence_gate
from src.agent.reranker import CrossEncoderReranker
from src.agent.streaming_bank import StreamingBank
from src.cc_service.encoder import EncoderSingleton


# Honest-silence strings. These are not error messages; they are the
# system's valid answer for "this is outside the library".
SILENCE_NO_MATCH = "I have no matching memory for that."
SILENCE_DRIFT = "I drafted an answer but could not ground it in memory."

# Number of "closest topics" to surface alongside a silence response. Three
# is enough to signal "the bank knows about X, Y, Z but not your question"
# without turning the silence path into a covert retrieval channel.
DEFAULT_CLOSEST_TOPICS_K: int = 3
# Truncation length for the source-text fallback when a cell's label is NULL.
_TOPIC_SNIPPET_LEN: int = 60


PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def _build_phi3_generator(
    model_name: str, use_4bit: bool
) -> tuple[Callable[[str], str], Callable[[str, Sequence[dict]], str]]:
    """Construct (generate, build_prompt) callables sharing a tokenizer.

    Pre-inits CUDA via ``mem_get_info`` to avoid the Windows lazy-init
    access violation that bites when bitsandbytes initialises CUDA after
    transformers does.
    """
    import torch

    if torch.cuda.is_available():
        _ = torch.cuda.mem_get_info()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": False,
        "attn_implementation": "eager",
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    def _generate(prompt: str, max_new_tokens: int = 200) -> str:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return gen.strip()

    def _build_prompt(question: str, cells: Sequence[dict]) -> str:
        facts_block = "\n".join(
            f"[{c['cell_id']}] {c['text']}" for c in cells if c.get("text")
        )
        system = (
            "You answer questions strictly from the FACTS provided. "
            "Use only what is in the FACTS; do not add outside information. "
            "Cite each fact you use with its bracketed id, e.g. [123]. "
            "Keep answers to one or two sentences."
        )
        user = (
            f"FACTS:\n{facts_block}\n\nQUESTION: {question}\n\n"
            f"Answer using only the FACTS above, with citations."
        )
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return _generate, _build_prompt


def _default_plain_prompt(question: str, cells: Sequence[dict]) -> str:
    """Fallback prompt-builder for injected generators (tests).

    Plain text so callers don't need a tokenizer.
    """
    facts_block = "\n".join(
        f"[{c['cell_id']}] {c['text']}" for c in cells if c.get("text")
    )
    return (
        f"FACTS:\n{facts_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the FACTS above, with citations."
    )


@dataclass
class PipelineResult:
    question: str
    answer: str
    silence: bool
    silence_reason: str
    citations: List[int]
    retrieval: dict
    gate: dict
    verification: Optional[dict]
    timings: dict
    # Top-N "closest topics" surfaced alongside silence (and informationally
    # on grounded answers). Each entry: {"topic": str, "activation": float}.
    # Topic is `cells.label` when set, else a truncated source-text snippet.
    closest_topics: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "silence": self.silence,
            "silence_reason": self.silence_reason,
            "citations": list(self.citations),
            "retrieval": self.retrieval,
            "gate": self.gate,
            "verification": self.verification,
            "timings": self.timings,
            "closest_topics": list(self.closest_topics),
        }


class AnswerPipeline:
    """End-to-end pipeline; instantiate once, call :meth:`ask` per question."""

    def __init__(
        self,
        bank_path: str | Path,
        *,
        top_k: int = 10,
        margin_threshold: float = v1_silence_gate.DEFAULT_MARGIN_THRESHOLD,
        min_coverage: float = v1_answer_verifier.MIN_COVERAGE,
        closest_topics_k: int = DEFAULT_CLOSEST_TOPICS_K,
        generator: Optional[Callable[[str], str]] = None,
        generator_model: str = PHI3_MODEL,
        use_4bit: bool = True,
        encoder: Optional[EncoderSingleton] = None,
        bank: Optional[StreamingBank] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_margin_threshold: Optional[float] = None,
    ) -> None:
        self.top_k = int(top_k)
        self.margin_threshold = float(margin_threshold)
        self.min_coverage = float(min_coverage)
        self.closest_topics_k = int(closest_topics_k)
        self.reranker = reranker
        # Reranker score scale differs from cosine activation scale, so the
        # threshold has to be supplied explicitly when reranker is on. No
        # safe default exists across cross-encoders.
        if reranker is not None and rerank_margin_threshold is None:
            raise ValueError(
                "rerank_margin_threshold must be set when reranker is provided"
            )
        self.rerank_margin_threshold = (
            float(rerank_margin_threshold)
            if rerank_margin_threshold is not None else None
        )

        t0 = time.time()
        self.bank = bank if bank is not None else StreamingBank(bank_path)
        self.bank_load_seconds = time.time() - t0

        encoder_name = self.bank.encoder_model or "all-MiniLM-L6-v2"
        self.encoder = encoder if encoder is not None else EncoderSingleton(
            model_name=encoder_name
        )
        if self.encoder.model_name != encoder_name:
            raise ValueError(
                f"encoder mismatch: bank built with {encoder_name!r}, "
                f"caller supplied {self.encoder.model_name!r}"
            )

        if generator is not None:
            self._generate = generator
            self._build_prompt = _default_plain_prompt
            self._generator_model_name = "<injected>"
        else:
            self._generate, self._build_prompt = _build_phi3_generator(
                generator_model, use_4bit
            )
            self._generator_model_name = generator_model

    # ---- public API ----

    def _build_closest_topics(self, retrieval: dict) -> list[dict]:
        """Top-K topic snippets for progressive disclosure on silence.

        Topic is the cell's label when set; falls back to a truncated
        source-text snippet so a NULL-label cell still names what it is
        about. Returns an empty list if the bank doesn't expose
        ``fetch_labels`` (e.g. an injected mock).
        """
        if self.closest_topics_k <= 0:
            return []
        ids = retrieval["top_k_cell_ids"][: self.closest_topics_k]
        acts = retrieval["activations"][: self.closest_topics_k]
        if not ids:
            return []
        try:
            labels = self.bank.fetch_labels(ids)
        except AttributeError:
            labels = [None] * len(ids)
        try:
            texts = self.bank.fetch_source_texts(ids)
        except AttributeError:
            texts = [None] * len(ids)
        out: list[dict] = []
        for label, text, act in zip(labels, texts, acts):
            topic: str
            if label and str(label).strip():
                topic = str(label).strip()
            elif text:
                snippet = " ".join(str(text).split())[:_TOPIC_SNIPPET_LEN]
                topic = snippet + ("…" if len(str(text)) > _TOPIC_SNIPPET_LEN
                                   else "")
            else:
                continue  # nothing useful to surface for this cell
            out.append({"topic": topic, "activation": float(act)})
        return out

    def ask(self, question: str) -> PipelineResult:
        timings: dict = {}

        t0 = time.time()
        raw = self.encoder.encode_one(question, is_query=True)
        whitened = self.bank.whiten(raw)
        timings["encode"] = time.time() - t0

        t0 = time.time()
        topk = self.bank.topk(whitened, k=self.top_k)
        timings["retrieve"] = time.time() - t0

        activations = topk["activations"]
        retrieval = {
            "top_k_cell_ids": [int(c) for c in topk["cell_ids"]],
            "activations": [float(a) for a in activations],
            "thetas": [float(t) for t in topk["thetas"]],
        }

        # Reranker path: fetch texts now so the cross-encoder can score
        # (query, text) pairs, then reorder cells by rerank score and gate
        # on the rerank margin instead of the cosine margin.
        reranked_cell_texts: Optional[list[str]] = None
        if self.reranker is not None:
            t0 = time.time()
            pre_texts = self.bank.fetch_source_texts(
                retrieval["top_k_cell_ids"]
            )
            timings["fetch_sources"] = time.time() - t0

            t0 = time.time()
            text_strs = [t or "" for t in pre_texts]
            rerank_scores = self.reranker.score(question, text_strs)
            # Cells with no source text are pushed below the rest by giving
            # them -inf; they cannot ground an answer in any case.
            for i, t in enumerate(pre_texts):
                if not t:
                    rerank_scores[i] = -float("inf")
            order = np.argsort(-rerank_scores)
            retrieval["top_k_cell_ids"] = [
                retrieval["top_k_cell_ids"][i] for i in order
            ]
            retrieval["activations"] = [
                retrieval["activations"][i] for i in order
            ]
            retrieval["thetas"] = [retrieval["thetas"][i] for i in order]
            retrieval["rerank_scores"] = [float(rerank_scores[i]) for i in order]
            reranked_cell_texts = [text_strs[i] for i in order]
            timings["rerank"] = time.time() - t0

            gate_signal = retrieval["rerank_scores"]
            gate_threshold = self.rerank_margin_threshold
        else:
            gate_signal = activations.tolist()
            gate_threshold = self.margin_threshold

        gate = v1_silence_gate.decide(
            gate_signal, margin_threshold=gate_threshold
        )
        gate_dict = gate.as_dict()
        gate_dict["signal_source"] = (
            "rerank" if self.reranker is not None else "encoder"
        )

        t0 = time.time()
        closest_topics = self._build_closest_topics(retrieval)
        timings["closest_topics"] = time.time() - t0

        if not gate.fire:
            return PipelineResult(
                question=question,
                answer=SILENCE_NO_MATCH,
                silence=True,
                silence_reason=f"gate: {gate.reason}",
                citations=[],
                retrieval=retrieval,
                gate=gate_dict,
                verification=None,
                timings=timings,
                closest_topics=closest_topics,
            )

        # Gate fired: fetch source texts only for the cells we'll cite
        # (reranker path already has them in hand and in the right order).
        if reranked_cell_texts is not None:
            texts = reranked_cell_texts
        else:
            t0 = time.time()
            texts = self.bank.fetch_source_texts(retrieval["top_k_cell_ids"])
            timings["fetch_sources"] = time.time() - t0
        cells = [
            {"cell_id": cid, "text": txt or ""}
            for cid, txt in zip(retrieval["top_k_cell_ids"], texts)
            if txt
        ]

        if not cells:
            return PipelineResult(
                question=question,
                answer=SILENCE_NO_MATCH,
                silence=True,
                silence_reason="gate fired but no source text on disk for top cells",
                citations=[],
                retrieval=retrieval,
                gate=gate_dict,
                verification=None,
                timings=timings,
                closest_topics=closest_topics,
            )

        t0 = time.time()
        prompt = self._build_prompt(question, cells)
        timings["build_prompt"] = time.time() - t0

        t0 = time.time()
        raw_answer = self._generate(prompt)
        timings["generate"] = time.time() - t0

        t0 = time.time()
        cell_texts = [c["text"] for c in cells]
        verdict = v1_answer_verifier.verify(
            raw_answer, cell_texts, question=question,
            min_coverage=self.min_coverage,
            cell_ids=[c["cell_id"] for c in cells],
            strict_nonsense=True,
        )
        timings["verify"] = time.time() - t0

        if not verdict.grounded:
            if verdict.reason.startswith("query incoherent") or \
               verdict.reason.startswith("query-evidence mismatch"):
                reason = f"verify: {verdict.reason}"
            elif verdict.coverage < verdict.threshold and verdict.uncited_numerics:
                reason = (
                    f"verify: coverage={verdict.coverage:.2f} "
                    f"< {verdict.threshold:.2f} AND "
                    f"uncited numerics={verdict.uncited_numerics[:3]}"
                )
            elif verdict.coverage < verdict.threshold:
                reason = (
                    f"verify: coverage={verdict.coverage:.2f} "
                    f"< {verdict.threshold:.2f}; "
                    f"uncovered={verdict.uncovered_tokens[:5]}"
                )
            elif verdict.unanchored_proper_nouns:
                reason = (
                    f"verify: answer entity not colocated with question "
                    f"anchor: {verdict.unanchored_proper_nouns[:3]}"
                )
            else:
                reason = (
                    f"verify: uncited numerics in answer "
                    f"{verdict.uncited_numerics[:3]}"
                )
            return PipelineResult(
                question=question,
                answer=SILENCE_DRIFT,
                silence=True,
                silence_reason=reason,
                citations=[c["cell_id"] for c in cells],
                retrieval=retrieval,
                gate=gate_dict,
                verification=verdict.as_dict(),
                timings=timings,
                closest_topics=closest_topics,
            )

        return PipelineResult(
            question=question,
            answer=raw_answer,
            silence=False,
            silence_reason="",
            citations=[c["cell_id"] for c in cells],
            retrieval=retrieval,
            gate=gate_dict,
            verification=verdict.as_dict(),
            timings=timings,
            closest_topics=closest_topics,
        )

    def close(self) -> None:
        self.bank.close()


__all__ = [
    "AnswerPipeline",
    "PipelineResult",
    "SILENCE_NO_MATCH",
    "SILENCE_DRIFT",
    "PHI3_MODEL",
    "DEFAULT_CLOSEST_TOPICS_K",
]
