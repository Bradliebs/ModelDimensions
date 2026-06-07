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
from src.agent.hybrid_retriever import HybridRetriever
from src.agent.lexical_index import LexicalIndex
from src.agent.reranker import CrossEncoderReranker
from src.agent.streaming_bank import StreamingBank
from src.cc_service.encoder import EncoderSingleton


# Honest-silence strings. These are not error messages; they are the
# system's valid answer for "this is outside the library".
SILENCE_NO_MATCH = "I have no matching memory for that."
SILENCE_DRIFT = "I drafted an answer but could not ground it in memory."

# Direct-evidence rescue policy. Rescue may override a sub-threshold gate
# only when the verifier already accepts and a top-N retrieved cell
# directly contains both the primary answer entity and a question anchor
# AND that cell's cosine activation clears an absolute floor. Calibrated
# from results/v1_rescue_floor_sweep.json.
RESCUE_RANK_WINDOW: int = 3
RESCUE_ACTIVATION_FLOOR: float = 0.40

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
    # Direct-evidence rescue audit. None when no rescue path was reached
    # (gate fired, or pre-filter rejected). Populated when the rescue gate
    # was evaluated (whether it granted or denied).
    rescue: Optional[dict] = None

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
            "rescue": self.rescue,
        }


def _decide_rescue(
    *,
    retrieval: dict,
    cells: list[dict],
    verdict: "v1_answer_verifier.VerificationDecision",
    gate_passed: bool,
    margin_observed: float,
    margin_threshold: float,
    rank_window: int = RESCUE_RANK_WINDOW,
    activation_floor: float = RESCUE_ACTIVATION_FLOOR,
) -> tuple[str, dict]:
    """Direct-evidence rescue gate.

    Only callable when the verifier already accepted (``verdict.grounded``).
    Returns ``(outcome, audit)`` where outcome is one of:

      ``"rescue"``                       — grant rescue
      ``"reject_no_primary"``            — verifier accepted but log has
                                            no novel non-skip entity
      ``"reject_no_supporting_cell"``    — no retrieved cell colocates the
                                            primary entity with any anchor
      ``"reject_outside_window"``        — supporting cell exists but is
                                            ranked at or beyond rank_window
      ``"reject_below_floor"``           — supporting cell is in window but
                                            its activation < floor
    """
    log = list(verdict.stage_e_log or [])
    novel = [
        rec for rec in log
        if rec.get("decision") != "skip" and rec.get("answer_entity")
    ]
    base_audit = {
        "rescue_attempted": True,
        "rescue_type": "direct_evidence",
        "normal_gate_passed": bool(gate_passed),
        "margin": float(margin_observed),
        "margin_threshold": float(margin_threshold),
        "activation_floor": float(activation_floor),
        "rank_window": int(rank_window),
        "stage_e_v2_passed": bool(verdict.grounded),
    }
    if not novel:
        audit = dict(base_audit)
        audit["decision"] = "silence"
        audit["rescue_rejection_reason"] = "no_primary_entity"
        return ("reject_no_primary", audit)

    primary_entity = novel[0]["answer_entity"]
    anchors = list(novel[0].get("question_anchors") or [])
    primary_norm = v1_answer_verifier._normalise_for_anchor(primary_entity)
    # Normalise anchors so the helper is robust to callers that pass raw
    # casings; in the production path the verifier emits anchors already
    # lowercased, so this is a no-op there.
    anchors_norm = [
        v1_answer_verifier._normalise_for_anchor(a) for a in anchors
    ]
    anchors_norm = [a for a in anchors_norm if a]

    text_by_id: dict[int, str] = {}
    for c in cells:
        cid = c.get("cell_id")
        text = c.get("text")
        if cid is None or not text:
            continue
        text_by_id[int(cid)] = text

    cell_ids = list(retrieval.get("top_k_cell_ids", []))
    activations = list(retrieval.get("activations", []))

    supporting: dict | None = None
    for rank, (cid, act) in enumerate(zip(cell_ids, activations)):
        text = text_by_id.get(int(cid))
        if not text:
            continue
        norm_text = v1_answer_verifier._normalise_for_anchor(text)
        if not primary_norm or primary_norm not in norm_text:
            continue
        anchors_present = [
            anchors[i] for i, a in enumerate(anchors_norm) if a in norm_text
        ]
        if not anchors_present:
            continue
        supporting = {
            "rank": rank,
            "cell_id": int(cid),
            "activation": float(act),
            "anchors_in_supporting_cell": anchors_present,
        }
        break

    if supporting is None:
        audit = dict(base_audit)
        audit["decision"] = "silence"
        audit["rescue_rejection_reason"] = "no_supporting_cell"
        audit["answer_entity"] = primary_entity
        audit["question_anchors"] = anchors
        return ("reject_no_supporting_cell", audit)

    audit = dict(base_audit)
    audit["answer_entity"] = primary_entity
    audit["question_anchors"] = anchors
    audit["supporting_cell_rank"] = supporting["rank"]
    audit["supporting_cell_id"] = supporting["cell_id"]
    audit["supporting_cell_activation"] = supporting["activation"]
    audit["anchors_in_supporting_cell"] = supporting["anchors_in_supporting_cell"]
    audit["entity_anchor_colocated"] = True
    audit["support_cell_cited"] = True

    if supporting["rank"] >= rank_window:
        audit["decision"] = "silence"
        audit["rescue_rejection_reason"] = "supporting_cell_outside_rank_window"
        return ("reject_outside_window", audit)
    if supporting["activation"] < activation_floor:
        audit["decision"] = "silence"
        audit["rescue_rejection_reason"] = "supporting_cell_below_floor"
        return ("reject_below_floor", audit)

    audit["decision"] = "answer_rescued"
    return ("rescue", audit)


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
        lexical_index: Optional[LexicalIndex] = None,
        lexical_k: int = 200,
    ) -> None:
        self.top_k = int(top_k)
        self.margin_threshold = float(margin_threshold)
        self.min_coverage = float(min_coverage)
        self.closest_topics_k = int(closest_topics_k)
        self.reranker = reranker
        # Hybrid retrieval is opt-in. When absent, behaviour is byte-
        # identical to V1 single-stage cosine retrieval (the
        # v1-grounded-answer-pipeline release tag).
        self.lexical_index = lexical_index
        self.lexical_k = int(lexical_k)
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

        # Construct the hybrid retriever once. None when the caller did
        # not supply a lexical index, in which case ``ask`` uses the
        # bank's ``topk`` directly.
        self._hybrid = (
            HybridRetriever(self.lexical_index, self.bank)
            if self.lexical_index is not None else None
        )

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

    def ask(
        self,
        question: str,
        *,
        on_phase: Optional[Callable[[str], None]] = None,
    ) -> PipelineResult:
        timings: dict = {}

        def _emit(phase: str) -> None:
            # Progress hook for streaming UIs. Never affects the result; a
            # failing callback must not break answering.
            if on_phase is None:
                return
            try:
                on_phase(phase)
            except Exception:
                pass

        _emit("Searching memory\u2026")

        t0 = time.time()
        raw = self.encoder.encode_one(question, is_query=True)
        whitened = self.bank.whiten(raw)
        timings["encode"] = time.time() - t0

        t0 = time.time()
        if self._hybrid is not None:
            topk = self._hybrid.topk(
                question, whitened,
                k_lexical=self.lexical_k, k_final=self.top_k,
            )
        else:
            topk = self.bank.topk(whitened, k=self.top_k)
        timings["retrieve"] = time.time() - t0

        activations = topk["activations"]
        retrieval = {
            "top_k_cell_ids": [int(c) for c in topk["cell_ids"]],
            "activations": [float(a) for a in activations],
            "thetas": [float(t) for t in topk["thetas"]],
        }
        # Carry the lexical-stage diagnostics through to PipelineResult
        # when the hybrid path produced them. Pure-cosine retrieval
        # omits these keys, preserving the V1 retrieval dict shape.
        if "lexical_scores" in topk:
            retrieval["lexical_scores"] = [
                float(s) for s in topk["lexical_scores"]
            ]
            retrieval["lexical_ranks"] = list(topk["lexical_ranks"])
            retrieval["retrieval_stage"] = topk.get("stage", "hybrid")

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

        gate_passed = bool(gate.fire)

        # Below-gate pre-filter: skip the generator entirely when no cell
        # in the rescue rank window clears the activation floor. This
        # preserves the original silence behaviour for queries with no
        # plausible support, and avoids paying the generation cost on
        # them. Rescue is only possible when at least one such cell
        # exists.
        if not gate_passed:
            top_window_acts = retrieval["activations"][:RESCUE_RANK_WINDOW]
            if not any(a >= RESCUE_ACTIVATION_FLOOR for a in top_window_acts):
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
                    rescue=None,
                )

        # Fetch source texts. The reranker path already has them in hand
        # and in the right order; otherwise pull from disk for the top_k
        # we will cite.
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
            reason = (
                "gate fired but no source text on disk for top cells"
                if gate_passed
                else f"gate: {gate.reason}; no source text available"
            )
            return PipelineResult(
                question=question,
                answer=SILENCE_NO_MATCH,
                silence=True,
                silence_reason=reason,
                citations=[],
                retrieval=retrieval,
                gate=gate_dict,
                verification=None,
                timings=timings,
                closest_topics=closest_topics,
                rescue=None,
            )

        t0 = time.time()
        prompt = self._build_prompt(question, cells)
        timings["build_prompt"] = time.time() - t0

        _emit("Drafting an answer\u2026")
        t0 = time.time()
        raw_answer = self._generate(prompt)
        timings["generate"] = time.time() - t0

        _emit("Checking the answer is grounded\u2026")
        t0 = time.time()
        cell_texts = [c["text"] for c in cells]
        verdict = v1_answer_verifier.verify(
            raw_answer, cell_texts, question=question,
            min_coverage=self.min_coverage,
            cell_ids=[c["cell_id"] for c in cells],
            strict_nonsense=True,
        )
        timings["verify"] = time.time() - t0

        # Gate-passed branch: existing two-outcome contract.
        if gate_passed:
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
                    rescue=None,
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
                rescue=None,
            )

        # Below-gate branch: rescue may grant when the verifier accepts and
        # a top-RESCUE_RANK_WINDOW cell directly supports the answer.
        if not verdict.grounded:
            return PipelineResult(
                question=question,
                answer=SILENCE_NO_MATCH,
                silence=True,
                silence_reason=f"gate: {gate.reason}; verify: {verdict.reason}",
                citations=[c["cell_id"] for c in cells],
                retrieval=retrieval,
                gate=gate_dict,
                verification=verdict.as_dict(),
                timings=timings,
                closest_topics=closest_topics,
                rescue=None,
            )

        t0 = time.time()
        rescue_outcome, rescue_audit = _decide_rescue(
            retrieval=retrieval,
            cells=cells,
            verdict=verdict,
            gate_passed=gate_passed,
            margin_observed=float(gate.margin),
            margin_threshold=float(gate_threshold),
        )
        timings["rescue"] = time.time() - t0

        if rescue_outcome == "rescue":
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
                rescue=rescue_audit,
            )

        return PipelineResult(
            question=question,
            answer=SILENCE_NO_MATCH,
            silence=True,
            silence_reason=(
                f"gate: {gate.reason}; "
                f"rescue: {rescue_audit['rescue_rejection_reason']}"
            ),
            citations=[c["cell_id"] for c in cells],
            retrieval=retrieval,
            gate=gate_dict,
            verification=verdict.as_dict(),
            timings=timings,
            closest_topics=closest_topics,
            rescue=rescue_audit,
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
