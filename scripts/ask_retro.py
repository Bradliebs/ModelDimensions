"""scripts/ask_retro.py -- Live retrieval-augmented generation smoke.

Wires the existing cc_service bank (~5.7M cells, all-MiniLM-L6-v2) to the
55M RetroGPT checkpoint via real-time HTTP retrieval, generates text for a
small prompt set both WITH and WITHOUT retrieval, and writes a transcript
artifact to reports/.

Builds on src/retro/generate_retro.py (which expects the flat-nanogpt
layout) -- this script is a repo-root entrypoint that:
  - imports through src.retro.model_retro
  - points at the real ckpt at H:\\MiniLM\\nanogpt\\out-retro-bank\\ckpt_best.pt
  - talks to a running cc_service at --service-url
  - emits a JSON + markdown transcript so the result is committable

HONEST CAVEAT (must read):
  The 55M RetroGPT was trained against its OWN cell_tokens.npy (built from a
  specific corpus at training time). At inference we feed it neighbors from
  the production cc_service bank. The MiniLM encoder identity matches
  (all-MiniLM-L6-v2, dim 384), so semantic retrieval is meaningful, but the
  source-text distribution differs from training. This is "does CCA produce
  coherent output when fed plausible but out-of-distribution neighbor token
  sequences at inference time?", NOT "is the trained pipeline working
  end-to-end."

Prerequisites:
  cc_service running at SERVICE_URL. From repo root:
      $env:CCMEM_DB_PATH = "H:\\MiniLM\\cc_service\\bank.db"
      $env:CCMEM_ENCODER = "all-MiniLM-L6-v2"
      .venv\\Scripts\\python.exe -m uvicorn src.cc_service.main:app `
          --host 127.0.0.1 --port 8765

Usage:
    python scripts/ask_retro.py \\
        --service-url http://127.0.0.1:8765 \\
        --out-json reports/live_generation_smoke.json \\
        --out-md reports/live_generation_smoke.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RETRO_DIR = ROOT / "src" / "retro"
for p in (str(ROOT), str(RETRO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tiktoken  # noqa: E402
import httpx  # noqa: E402

from src.retro.model_retro import RetroGPT  # noqa: E402


DEFAULT_CKPT = Path(r"H:\MiniLM\nanogpt\out-retro-bank\ckpt_best.pt")
DEFAULT_SERVICE = "http://127.0.0.1:8765"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = (
    torch.bfloat16
    if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)
SEED = 1337
EOT_TOKEN = 50256

PROMPTS = [
    "The theory of general relativity describes",
    "Machine learning is a branch of artificial intelligence that",
    "The French Revolution began in 1789 when",
    "In computer science, a hash table is",
    "The Great Wall of China was built",
]


def query_bank(client: httpx.Client, text: str, top_k: int = 2) -> list[dict]:
    r = client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


def retrieve_neighbors(
    client: httpx.Client,
    chunks_text: list[str],
    enc: tiktoken.Encoding,
    n_neighbors: int,
    neighbor_len: int,
) -> tuple[torch.Tensor, list[list[dict]]]:
    """Return (neighbor tensor shape (1,K,n_neighbors,neighbor_len), per-chunk hit metadata)."""
    K = len(chunks_text)
    nbrs = np.full((1, K, n_neighbors, neighbor_len), EOT_TOKEN, dtype=np.int64)
    hits_meta: list[list[dict]] = []
    for ci, chunk_text in enumerate(chunks_text):
        if not chunk_text.strip():
            hits_meta.append([])
            continue
        hits = query_bank(client, chunk_text, top_k=n_neighbors)
        chunk_meta = []
        for ni, hit in enumerate(hits[:n_neighbors]):
            source = hit.get("source_text") or hit.get("source") or ""
            toks = enc.encode(source, allowed_special={"<|endoftext|>"})
            toks = toks[:neighbor_len]
            if len(toks) < neighbor_len:
                toks = toks + [EOT_TOKEN] * (neighbor_len - len(toks))
            nbrs[0, ci, ni] = toks
            chunk_meta.append({
                "rank": ni,
                "activation": hit.get("activation"),
                "source_text_preview": (source[:120] + "...") if len(source) > 120 else source,
            })
        hits_meta.append(chunk_meta)
    return torch.from_numpy(nbrs).to(DEVICE), hits_meta


@torch.no_grad()
def generate(
    model: RetroGPT,
    enc: tiktoken.Encoding,
    client: httpx.Client | None,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    use_retrieval: bool,
) -> tuple[str, list[list[dict]]]:
    """Generate text. Re-retrieves neighbors only when crossing a chunk boundary."""
    model.eval()
    config = model.config
    chunk_size = config.chunk_size
    block_size = config.block_size
    K = block_size // chunk_size

    prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    if len(prompt_ids) > block_size:
        prompt_ids = prompt_ids[-block_size:]

    seq = list(prompt_ids)
    generated_ids: list[int] = []
    all_hits: list[list[dict]] = []
    cached_neighbors: torch.Tensor | None = None
    last_chunk_idx = -1

    ctx = (
        torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
        if DEVICE_TYPE == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    for _ in range(max_new_tokens):
        window = seq[-block_size:]
        if len(window) < block_size:
            window = [0] * (block_size - len(window)) + window
        idx = torch.tensor([window], dtype=torch.long, device=DEVICE)

        neighbors = None
        if use_retrieval:
            assert client is not None
            current_chunk = len(seq) // chunk_size
            if cached_neighbors is None or current_chunk != last_chunk_idx:
                chunks_text = [
                    enc.decode(window[ci * chunk_size : (ci + 1) * chunk_size])
                    for ci in range(K)
                ]
                cached_neighbors, hits_meta = retrieve_neighbors(
                    client, chunks_text, enc, config.n_neighbors, config.neighbor_len
                )
                last_chunk_idx = current_chunk
                # Record the retrieval that drove the FIRST chunk-boundary
                # crossing only (otherwise we get one record per chunk * len/chunk
                # which floods the report).
                if not all_hits:
                    all_hits.append({"chunks_text": chunks_text, "hits": hits_meta})
            neighbors = cached_neighbors

        with ctx:
            logits, _ = model(idx, neighbors=neighbors)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        seq.append(next_id)
        generated_ids.append(next_id)
        if next_id == enc.eot_token:
            break

    return enc.decode(generated_ids), all_hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--service-url", default=DEFAULT_SERVICE)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="override the default prompt set")
    args = parser.parse_args(argv)

    if not args.ckpt.exists():
        parser.error(f"checkpoint not found: {args.ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[ask_retro] device={DEVICE} dtype={DTYPE}")
    # Verify the service is up before loading the model.
    # First /query on a 5.7M-cell bank can take minutes (encoder load + index
    # warmup), so the per-request timeout is generous.
    client = httpx.Client(base_url=args.service_url, timeout=300.0)
    try:
        info_resp = client.get("/info")
        info_resp.raise_for_status()
        info = info_resp.json()
        print(
            f"[ask_retro] bank: {info.get('n_cells', '?'):,} cells, "
            f"dim={info.get('dim', '?')}, encoder={info.get('encoder_model', '?')}"
        )
    except Exception as e:
        print(f"[ask_retro] ERROR: cc_service not reachable at {args.service_url}: {e}",
              file=sys.stderr)
        return 2

    # Warmup query so the model-load + index-build cost is paid before we
    # start timing prompts.
    print("[ask_retro] warming up bank (first query may be slow)...")
    t0 = time.time()
    _ = query_bank(client, "warmup", top_k=2)
    print(f"[ask_retro] warmup done in {time.time() - t0:.1f}s")

    print(f"[ask_retro] loading checkpoint {args.ckpt}")
    ckpt = torch.load(str(args.ckpt), map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(
        f"[ask_retro] config: n_layer={config.n_layer} n_head={config.n_head} "
        f"n_embd={config.n_embd} block={config.block_size} chunk={config.chunk_size} "
        f"n_neighbors={config.n_neighbors} neighbor_len={config.neighbor_len}"
    )
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ask_retro] model params: {n_params/1e6:.2f}M")

    enc = tiktoken.get_encoding("gpt2")
    prompts = args.prompts if args.prompts else PROMPTS

    results: list[dict] = []
    t_total = time.time()
    for prompt in prompts:
        print(f"\n[ask_retro] prompt: {prompt}")

        torch.manual_seed(args.seed)
        t0 = time.time()
        no_text, _ = generate(
            model, enc, None, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
            use_retrieval=False,
        )
        dt_no = time.time() - t0

        torch.manual_seed(args.seed)
        t0 = time.time()
        ret_text, hits_meta = generate(
            model, enc, client, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k,
            use_retrieval=True,
        )
        dt_ret = time.time() - t0

        print(f"  WITHOUT retrieval ({dt_no:.1f}s): {no_text[:120]}...")
        print(f"  WITH    retrieval ({dt_ret:.1f}s): {ret_text[:120]}...")

        results.append({
            "prompt": prompt,
            "without_retrieval": {
                "continuation": no_text,
                "elapsed_sec": dt_no,
            },
            "with_retrieval": {
                "continuation": ret_text,
                "elapsed_sec": dt_ret,
                "first_chunk_retrieval": hits_meta,
            },
        })

    elapsed = time.time() - t_total
    print(f"\n[ask_retro] done in {elapsed:.1f}s ({len(results)} prompts)")

    # Quick sanity metric: any character-level overlap between retrieved
    # source-text previews and the generated continuation? This is a coarse
    # hallucination-vs-grounding signal (not a real verifier).
    for rec in results:
        gen = rec["with_retrieval"]["continuation"].lower()
        all_hits = rec["with_retrieval"]["first_chunk_retrieval"]
        sources = []
        for entry in all_hits:
            for chunk_hits in entry.get("hits", []):
                for hit in chunk_hits:
                    src = (hit.get("source_text_preview") or "").lower()
                    if src:
                        sources.append(src)
        # token-style overlap on words >= 4 chars
        gen_words = {w for w in gen.split() if len(w) >= 4}
        src_words = set()
        for s in sources:
            src_words.update(w for w in s.split() if len(w) >= 4)
        overlap = gen_words & src_words
        rec["with_retrieval"]["coarse_overlap_word_count"] = len(overlap)
        rec["with_retrieval"]["coarse_overlap_words_sample"] = sorted(overlap)[:10]

    payload = {
        "results": results,
        "meta": {
            "experiment": "live_generation_smoke",
            "ckpt_path": str(args.ckpt),
            "ckpt_params_million": n_params / 1e6,
            "service_url": args.service_url,
            "bank_info": info,
            "encoder_match": info.get("encoder_model") == "all-MiniLM-L6-v2",
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "device": DEVICE,
            "dtype": str(DTYPE),
            "elapsed_sec_total": elapsed,
            "caveats": [
                "55M debug ckpt, not the 404M roadmap target.",
                "Inference bank is the 5.7M cc_service production bank, not the "
                "cell_tokens.npy the model trained against. Encoder identity matches "
                "(all-MiniLM-L6-v2, dim 384) so retrieval is semantically meaningful, "
                "but source-text distribution differs from training.",
                "first_chunk_retrieval shows only the chunks at the FIRST chunk-boundary "
                "crossing (one snapshot per prompt) to keep the report compact.",
                "coarse_overlap_word_count is a crude grounding proxy, not a verifier.",
            ],
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ask_retro] wrote {args.out_json}")

    # Markdown report
    md = [
        "# Live generation smoke -- RetroGPT 55M + cc_service bank",
        "",
        f"- Checkpoint: `{args.ckpt}` ({n_params/1e6:.2f}M params, "
        f"n_layer={config.n_layer}, n_embd={config.n_embd})",
        f"- Service: {args.service_url}",
        f"- Bank: {info.get('n_cells', '?'):,} cells, dim={info.get('dim', '?')}, "
        f"encoder={info.get('encoder_model', '?')}",
        f"- Device: {DEVICE} / {DTYPE}",
        f"- Max new tokens: {args.max_tokens}, temperature: {args.temperature}, "
        f"top_k: {args.top_k}, seed: {args.seed}",
        f"- Total elapsed: {elapsed:.1f}s for {len(results)} prompts",
        "",
        "## What this is (and is not)",
        "",
        "This is a functional smoke test that the wiring works end-to-end:",
        "RetroGPT 55M -> chunked queries to cc_service -> tokenized neighbors fed",
        "back into CCA layers at generation time. Per prompt we show both the",
        "WITHOUT-retrieval and WITH-retrieval continuations using the same seed.",
        "",
        "It is **not** a quality benchmark. The 55M model is small and was trained",
        "against a different bank than the inference-time cc_service bank. The",
        "encoder identity matches (all-MiniLM-L6-v2, dim 384), so retrieval is",
        "semantically meaningful, but source-text distribution differs from training.",
        "",
    ]
    for rec in results:
        md += [
            f"## Prompt: {rec['prompt']}",
            "",
            f"**WITHOUT retrieval** ({rec['without_retrieval']['elapsed_sec']:.1f}s):",
            "",
            f"> {rec['without_retrieval']['continuation']}",
            "",
            f"**WITH retrieval** ({rec['with_retrieval']['elapsed_sec']:.1f}s, "
            f"coarse_overlap={rec['with_retrieval']['coarse_overlap_word_count']} words):",
            "",
            f"> {rec['with_retrieval']['continuation']}",
            "",
        ]
        # show retrieved sources for the first chunk-boundary snapshot
        if rec["with_retrieval"]["first_chunk_retrieval"]:
            entry = rec["with_retrieval"]["first_chunk_retrieval"][0]
            md.append("First-chunk retrieval:")
            md.append("")
            for ci, chunk_hits in enumerate(entry.get("hits", [])):
                if not chunk_hits:
                    continue
                md.append(f"- chunk {ci}:")
                for hit in chunk_hits:
                    act = hit.get("activation")
                    act_str = f"{act:.3f}" if isinstance(act, (int, float)) else "?"
                    md.append(
                        f"  - act={act_str}: {hit.get('source_text_preview', '')}"
                    )
            md.append("")

    md += [
        "## Caveats",
        "",
    ] + [f"- {c}" for c in payload["meta"]["caveats"]]

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ask_retro] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
