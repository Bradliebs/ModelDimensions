import json
from pathlib import Path

p = Path("results/v1_pipeline_multihop_eval.json")
d = json.loads(p.read_text(encoding="utf-8"))
for r in d["records"]:
    print("=" * 78)
    print(f"#{r['n']}  {r['query']}")
    print(f"hops: {r['hops']}")
    print(f"margin: {r['gate_margin']:+.4f}")
    print(f"answer: {r['answer'][:200]}")
    print("--- top3 cells ---")
    for i, c in enumerate(r["top_cells"]):
        print(f"  [{i+1}] act={c['activation']:.4f}  cell={c['cell_id']}")
        print(f"       {c['text_snippet'][:250]}")
    print()
