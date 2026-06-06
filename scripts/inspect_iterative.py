import json
from pathlib import Path

p = Path("results/v1_pipeline_iterative_multihop_eval.json")
d = json.loads(p.read_text(encoding="utf-8"))
for r in d["records"]:
    print("=" * 80)
    print(f"#{r['n']}  {r['query']}")
    print(f"hops: {r['hops']}")
    print(f"expanded: {r['expanded_query']}")
    print(f"entities: {r['pass1']['extracted_entities']}")
    print(f"outcome: {r['outcome']}  kw_hit={r['kw_hit']}  margin={r['pass2']['gate_margin']:+.4f}")
    print(f"silence_reason: {r['silence_reason']}")
    print(f"answer: {r['answer'][:240]}")
    print("--- pass-2 top3 ---")
    for i, c in enumerate(r["pass2"]["top_cells"]):
        print(f"  [{i+1}] act={c['activation']:.4f}  cell={c['cell_id']}")
        print(f"       {c['text_snippet'][:260]}")
    print()
