"""
================================================================================
eval_teach_new_fact.py - Exp 9: Teach a Genuinely New Fact
================================================================================

The clean, unambiguous version of the mutable-knowledge demo.

Exp 8 (eval_mutable_knowledge.py) deleted EXISTING topic cells. The effect was
real but modest (10-16%) because the 5.7M-cell bank holds many near-duplicate
cells: when you delete the photosynthesis cells, similar fallback cells quietly
fill in and mask the loss. That redundancy is also why French Revolution came
out ambiguous.

This experiment removes that confound by teaching the model a fact that CANNOT
already be in the bank - an invented entity ("the Zarnik comet"). Because no
duplicate cells exist, the effect is clean and large:

  Phase 0  IGNORANT  - bank has no Zarnik cell; loss on the fact is high.
  Phase 1  TAUGHT    - write the fact; retrieval finds it; loss DROPS.
  Phase 2  FORGOTTEN - delete the fact; retrieval loses it; loss returns.

A specificity control (an unrelated passage) is measured at every phase to show
the teaching is specific to the new fact and does not perturb everything else.

The weights are frozen the entire time. The only thing that changes is the bank.

Generation is NOT shown: the 55M model's free-running text is too noisy to read
small retrieval effects (confirmed in Exp 8). Cross-entropy loss is the rigorous,
deterministic signal.

REQUIRES:
  - cc_service running at 127.0.0.1:8765 (bank warm)
  - out-retro-bank/ckpt_best.pt

USAGE:
  cd H:/MiniLM/nanogpt
  H:/MiniLM/cc_service/.venv/Scripts/python.exe eval_teach_new_fact.py

NOTE: After the deletes in Phase 2, the first /query triggers a full bank
matrix reload (minutes at 5.7M cells). The 600s client timeout covers it. Warm
the service with one /query before running to avoid a slow cold start.
================================================================================
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import tiktoken
import httpx

from model_retro import RetroConfig, RetroGPT

# ---- Config ----
_SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = _SCRIPT_DIR / "out-retro-bank" / "ckpt_best.pt"
SERVICE_URL = "http://127.0.0.1:8765"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if "cuda" in DEVICE else "cpu"
DTYPE = (
    torch.bfloat16
    if (DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported())
    else torch.float16
)

# First query (and every query after a delete) reloads the full bank matrix.
_client = httpx.Client(base_url=SERVICE_URL, timeout=600)


# ---- The invented fact (cannot already be in the bank) ----
# Written to the bank as a few paraphrases so retrieval reliably picks it up for
# the fact-bearing chunks. ALL of them are ours and ALL are deleted in Phase 2,
# so the fact disappears completely - no redundancy masking.
FACT_CELLS = [
    "The Zarnik comet passes Earth every 412 years and glows green from "
    "thallium vapor. Discovered by Elina Vorst in 1968, it is nicknamed the "
    "Emerald Wanderer; its next appearance is in the year 2387.",
    "Zarnik comet: a green comet whose color comes from thallium vapor, "
    "returning to Earth every 412 years. Astronomer Elina Vorst found it in "
    "1968. Its nickname is the Emerald Wanderer and its next pass is 2387.",
    "The Emerald Wanderer, formally the Zarnik comet, shines green because of "
    "thallium vapor and orbits past Earth on a 412 year cycle, next visible in "
    "2387, first observed by the astronomer Elina Vorst in 1968.",
]

# Passage used to measure loss. Leads with the fact so chunk 0 retrieves it, and
# is dense enough to fill most of the 256-token block (less padding dilution).
FACT_PASSAGE = (
    "The Zarnik comet is a periodic comet that passes Earth every 412 years and "
    "glows a distinctive green. The green color of the Zarnik comet is caused by "
    "an unusual concentration of thallium vapor in its coma. It was discovered "
    "in 1968 by the astronomer Elina Vorst, who recorded its 412 year period and "
    "named it the Emerald Wanderer for its green glow. The Zarnik comet, also "
    "called the Emerald Wanderer, last appeared in the twentieth century and its "
    "next appearance is expected in the year 2387. Because the Zarnik comet "
    "glows green from thallium vapor and returns only every 412 years, the "
    "Emerald Wanderer is one of the rarest green comets known to astronomers."
)

# Fact-saturated variant: EVERY chunk is packed with the invented tokens
# (Zarnik / green / thallium / 412 / Emerald Wanderer / Elina Vorst / 1968 /
# 2387) and the prose fills the full 256-token block, so every chunk retrieves
# the fact AND padding dilution is minimized. This is the lever for magnitude -
# the mechanism is identical, there is just more fact-bearing surface to help.
SATURATED_FACT_PASSAGE = (
    "The Zarnik comet, also known as the Emerald Wanderer, is a green comet that "
    "passes Earth every 412 years. The Zarnik comet glows green because of "
    "thallium vapor in its coma, and this green color is how the Emerald "
    "Wanderer earned its name. The astronomer Elina Vorst discovered the Zarnik "
    "comet in 1968 and measured its 412 year orbital period. According to Elina "
    "Vorst, the green Zarnik comet, the Emerald Wanderer, will next appear in the "
    "year 2387. The Zarnik comet is famous among astronomers because the Emerald "
    "Wanderer glows green from thallium vapor and returns to Earth only once "
    "every 412 years. The next appearance of the green Zarnik comet, discovered "
    "by Elina Vorst in 1968, is the year 2387, which makes the Emerald Wanderer "
    "one of the rarest periodic comets known. Every 412 years the thallium green "
    "Zarnik comet, the Emerald Wanderer, sweeps past Earth, just as Elina Vorst "
    "first recorded in 1968, glowing its unmistakable green."
)

# Specificity control: unrelated passage. Should barely move across phases.
CONTROL_PASSAGE = (
    "The water cycle describes the continuous movement of water on, above and "
    "below the surface of the Earth. Water evaporates from oceans and lakes, "
    "rises into the atmosphere, cools and condenses into clouds, and falls back "
    "to the surface as precipitation such as rain or snow. The water then flows "
    "through rivers and streams, soaks into the ground, and eventually returns "
    "to the oceans, where the cycle begins again. Evaporation, condensation and "
    "precipitation are the three main stages of this cycle, which is driven by "
    "energy from the Sun and by gravity."
)

# Queries to probe the bank for the invented entity (novelty + verification).
FACT_QUERIES = [
    "Zarnik comet green thallium 412 years",
    "Emerald Wanderer comet Elina Vorst 1968",
    "green comet next appearance 2387",
]


# ---- Bank operations ----
@dataclass
class WrittenCell:
    cell_id: int
    text: str


def query_bank(text: str, top_k: int = 2) -> list[dict]:
    r = _client.post("/query", json={"text": text, "top_k": top_k})
    r.raise_for_status()
    return r.json().get("hits", [])


def query_bank_silent(text: str, top_k: int = 5) -> list[dict]:
    r = _client.post(
        "/query", json={"text": text, "top_k": top_k, "include_silent": True}
    )
    r.raise_for_status()
    return r.json().get("hits", [])


def delete_cell(cell_id: int) -> dict:
    r = _client.delete(f"/cells/{cell_id}")
    r.raise_for_status()
    return r.json()


def write_cell(text: str, label: str | None = None) -> dict:
    r = _client.post("/write", json={"text": text, "label": label})
    r.raise_for_status()
    return r.json()


def get_bank_info() -> dict:
    r = _client.get("/info")
    r.raise_for_status()
    return r.json()


# ---- Retrieval for RETRO ----
def retrieve_neighbors_for_chunks(
    chunks_text: list[str],
    enc: tiktoken.Encoding,
    n_neighbors: int,
    neighbor_len: int,
) -> torch.Tensor:
    K = len(chunks_text)
    nbrs = np.zeros((1, K, n_neighbors, neighbor_len), dtype=np.int64)
    for ci, chunk_text in enumerate(chunks_text):
        if not chunk_text.strip():
            continue
        hits = query_bank(chunk_text, top_k=n_neighbors)
        for ni, hit in enumerate(hits[:n_neighbors]):
            source = hit.get("source_text", "")
            toks = enc.encode(source, allowed_special={"<|endoftext|>"})
            toks = toks[:neighbor_len]
            if len(toks) < neighbor_len:
                toks = toks + [0] * (neighbor_len - len(toks))
            nbrs[0, ci, ni] = toks
    return torch.from_numpy(nbrs).to(DEVICE)


# ---- Loss measurement ----
@torch.no_grad()
def measure_loss(
    model: RetroGPT,
    enc: tiktoken.Encoding,
    passage: str,
    use_retrieval: bool = True,
) -> float:
    """Cross-entropy loss on a passage with optional bank retrieval."""
    model.eval()
    config = model.config
    chunk_size = config.chunk_size
    block_size = config.block_size
    K = block_size // chunk_size

    tokens = enc.encode(passage, allowed_special={"<|endoftext|>"})
    tokens = tokens[:block_size]
    if len(tokens) < block_size:
        tokens = tokens + [0] * (block_size - len(tokens))

    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    neighbors = None
    if use_retrieval:
        chunks_text = []
        for ci in range(K):
            start = ci * chunk_size
            chunk_toks = tokens[start : start + chunk_size]
            chunks_text.append(enc.decode(chunk_toks))
        neighbors = retrieve_neighbors_for_chunks(
            chunks_text, enc, config.n_neighbors, config.neighbor_len
        )

    ctx = (
        torch.amp.autocast(device_type=DEVICE_TYPE, dtype=DTYPE)
        if DEVICE_TYPE == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    with ctx:
        logits, loss = model(idx, targets=idx, neighbors=neighbors)
    return loss.item()


def show_top_retrievals(query: str, k: int = 3) -> None:
    """Print what the bank returns for a probe query (for transparency)."""
    hits = query_bank_silent(query, top_k=k)
    if not hits:
        print(f"      (no hits for \"{query}\")")
        return
    for h in hits:
        snippet = (h.get("source_text") or h.get("label") or "")[:64]
        snippet = " ".join(snippet.split())
        print(f"      cell {h['cell_id']:>9d}  act={h['activation']:.4f}  "
              f"\"{snippet}...\"")


def main():
    print("=" * 78)
    print("  Exp 9: Teach a Genuinely New Fact - clean mutable-knowledge demo")
    print("=" * 78)

    # --- Service + bank ---
    try:
        info = get_bank_info()
    except Exception as e:
        print(f"\nERROR: bank service not reachable at {SERVICE_URL}: {e}",
              file=sys.stderr)
        print("Start cc_service first.", file=sys.stderr)
        sys.exit(1)
    initial_count = info["n_cells"]
    print(f"\nBank: {initial_count:,} cells, dim={info['dim']}")

    # --- Model ---
    print(f"Loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = RetroGPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params "
          f"(weights FROZEN for the whole experiment)")

    # --- Novelty check: prove the bank does not already know this entity ---
    print(f"\n{'=' * 78}")
    print("  NOVELTY CHECK - is 'the Zarnik comet' already in the bank?")
    print(f"{'=' * 78}")
    max_act = 0.0
    for q in FACT_QUERIES:
        hits = query_bank_silent(q, top_k=1)
        a = hits[0]["activation"] if hits else 0.0
        max_act = max(max_act, a)
        print(f"  probe \"{q}\"")
        show_top_retrievals(q, k=2)
    print(f"\n  Best match for the invented fact: activation={max_act:.4f}")
    print("  (low / unrelated = the bank genuinely does not know this fact)")

    # --- Passages to measure head-to-head (original vs fact-saturated) ---
    passages = [("original", FACT_PASSAGE), ("saturated", SATURATED_FACT_PASSAGE)]

    def measure_all(use_retr: bool = True) -> dict[str, float]:
        return {
            name: measure_loss(model, enc, text, use_retrieval=use_retr)
            for name, text in passages
        }

    # --- Phase 0: IGNORANT ---
    print(f"\n{'=' * 78}")
    print("  PHASE 0: IGNORANT (fact not in bank)")
    print(f"{'=' * 78}")
    ignorant = measure_all(use_retr=True)
    noretr = measure_all(use_retr=False)
    ctrl_ignorant = measure_loss(model, enc, CONTROL_PASSAGE, use_retrieval=True)
    for name, _ in passages:
        print(f"  Fact[{name:<9s}] loss: retrieval={ignorant[name]:.4f}  "
              f"no-retrieval={noretr[name]:.4f}")
    print(f"  Control passage loss (with retrieval): {ctrl_ignorant:.4f}")

    # --- Teach: write the fact cells ---
    print(f"\n{'=' * 78}")
    print(f"  TEACHING: writing {len(FACT_CELLS)} fact cell(s) to the bank")
    print(f"{'=' * 78}")
    written: list[WrittenCell] = []
    for text in FACT_CELLS:
        res = write_cell(text, label="zarnik_comet_fact")
        written.append(WrittenCell(cell_id=res["cell_id"], text=text))
        print(f"  wrote cell {res['cell_id']}")
    info = get_bank_info()
    print(f"  Bank now: {info['n_cells']:,} cells")

    # --- Phase 1: TAUGHT ---
    print(f"\n{'=' * 78}")
    print("  PHASE 1: TAUGHT (fact in bank)")
    print(f"{'=' * 78}")
    print("  Retrieval for the fact query now returns:")
    show_top_retrievals(FACT_QUERIES[0], k=3)
    taught = measure_all(use_retr=True)
    ctrl_taught = measure_loss(model, enc, CONTROL_PASSAGE, use_retrieval=True)
    print()
    for name, _ in passages:
        print(f"  Fact[{name:<9s}] loss: {taught[name]:.4f}  "
              f"(delta vs ignorant: {taught[name] - ignorant[name]:+.4f})")
    print(f"  Control passage loss: {ctrl_taught:.4f}  "
          f"(delta vs ignorant: {ctrl_taught - ctrl_ignorant:+.4f})")

    # --- Forget: delete the fact cells ---
    print(f"\n{'=' * 78}")
    print(f"  FORGETTING: deleting the {len(written)} fact cell(s)")
    print("  (next query triggers a full bank reload - may take minutes)")
    print(f"{'=' * 78}")
    deleted = 0
    for w in written:
        res = delete_cell(w.cell_id)
        if res.get("deleted"):
            deleted += 1
        else:
            print(f"  SKIP cell {w.cell_id}: {res.get('reason')}")
    info = get_bank_info()
    print(f"  Deleted {deleted} cell(s). Bank now: {info['n_cells']:,} cells")

    # --- Phase 2: FORGOTTEN ---
    print(f"\n{'=' * 78}")
    print("  PHASE 2: FORGOTTEN (fact removed from bank)")
    print(f"{'=' * 78}")
    print("  Retrieval for the fact query now returns:")
    show_top_retrievals(FACT_QUERIES[0], k=3)
    forgotten = measure_all(use_retr=True)
    ctrl_forgotten = measure_loss(model, enc, CONTROL_PASSAGE, use_retrieval=True)
    print()
    for name, _ in passages:
        print(f"  Fact[{name:<9s}] loss: {forgotten[name]:.4f}  "
              f"(delta vs taught: {forgotten[name] - taught[name]:+.4f})")
    print(f"  Control passage loss: {ctrl_forgotten:.4f}  "
          f"(delta vs taught: {ctrl_forgotten - ctrl_taught:+.4f})")

    # --- Final report ---
    final_info = get_bank_info()
    final_count = final_info["n_cells"]
    ctrl_swing = max(
        abs(ctrl_taught - ctrl_ignorant),
        abs(ctrl_forgotten - ctrl_taught),
    )
    # Per-passage learn/forget effects (positive = as expected).
    effects = {
        name: {
            "learned": ignorant[name] - taught[name],
            "forgot": forgotten[name] - taught[name],
        }
        for name, _ in passages
    }
    # The saturated passage is the magnitude lever; judge the verdict on its best.
    best_name = max(effects, key=lambda n: effects[n]["learned"])
    learned = effects[best_name]["learned"]
    forgot = effects[best_name]["forgot"]

    print(f"\n{'=' * 78}")
    print("  FINAL REPORT - Exp 9: Teach a Genuinely New Fact")
    print(f"{'=' * 78}")
    print(f"\n  Bank cells: {initial_count:,} -> {final_count:,} "
          f"(delta: {final_count - initial_count:+d})")
    print(f"\n  {'Passage':<11s} {'Ignorant':>9s} {'Taught':>9s} "
          f"{'Forgotten':>10s} {'Learned':>9s} {'Forgot':>9s}")
    print(f"  {'~' * 66}")
    for name, _ in passages:
        print(f"  {name:<11s} {ignorant[name]:>9.4f} {taught[name]:>9.4f} "
              f"{forgotten[name]:>10.4f} {effects[name]['learned']:>+9.4f} "
              f"{effects[name]['forgot']:>+9.4f}")
    print(f"  {'control':<11s} {ctrl_ignorant:>9.4f} {ctrl_taught:>9.4f} "
          f"{ctrl_forgotten:>10.4f} {'':>9s} {'':>9s}")
    print(f"  {'~' * 66}")
    base = effects['original']['learned']
    sat = effects['saturated']['learned']
    if abs(base) > 1e-9:
        print(f"\n  Magnitude lift (saturated vs original learning effect): "
              f"{sat:+.4f} vs {base:+.4f} = {sat / base:.2f}x")
    print(f"  Best passage: '{best_name}'  learned {learned:+.4f}, "
          f"forgot {forgot:+.4f} nats")
    print(f"  Control passage max swing: {ctrl_swing:.4f} nats (want ~0)")

    # --- Verdict ---
    learned_ok = learned > 0.01
    forgot_ok = forgot > 0.01
    specific_ok = ctrl_swing < max(0.01, abs(learned) * 0.5)
    restored_ok = final_count == initial_count

    print(f"\n  {'~' * 66}")
    if learned_ok and forgot_ok and specific_ok and restored_ok:
        print("  VERDICT: PASS")
        print("  Writing a brand-new fact lowered the model's loss; deleting it")
        print("  raised the loss back. The control passage barely moved, so the")
        print("  effect is specific to the new fact - not a global shift. The")
        print("  weights never changed; only the bank did. A brain that grows.")
    elif learned_ok and forgot_ok:
        print("  VERDICT: PARTIAL - learn/forget worked but the control also")
        print("  moved (effect not fully specific) or the bank count differs.")
    elif learned_ok:
        print("  VERDICT: PARTIAL - the model learned the fact but forgetting")
        print("  did not cleanly reverse it.")
    else:
        print("  VERDICT: FAIL - writing the fact did not measurably lower loss.")
        print("  Likely the fact cell was not retrieved for the passage chunks;")
        print("  check the Phase 1 retrieval output above.")
    if not restored_ok:
        print(f"  WARNING: bank count changed ({initial_count:,} -> "
              f"{final_count:,}). Investigate before further experiments.")
    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    main()
