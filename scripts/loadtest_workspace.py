"""Load-test harness for a running Bank Management Workspace.

This is a measurement client only. It does NOT start a server or load the
bank; it sends HTTP requests to a workspace you are already running, so the
8.75 GB in-memory bank is never loaded twice.

Usage (against a running localhost server):

    python scripts/loadtest_workspace.py --url http://127.0.0.1:8765 \
        --concurrency 4 --requests 40

It reports p50/p95/p99 latency, throughput, and error rate for /api/ask.
Pass --token if the server has WORKSPACE_API_TOKEN set with read-auth on.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "Summarise the grounded answer pipeline.",
    "What does the silence gate do?",
    "How does the answer verifier decide coverage?",
]


def _ask_once(url: str, question: str, token: str | None, timeout: float) -> tuple[float, int, bool]:
    body = json.dumps({"question": question}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{url.rstrip('/')}/api/ask", data=body, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - start
            silence = bool(payload.get("silence", False))
            return elapsed, resp.status, silence
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - start, exc.code, False
    except Exception:
        return time.perf_counter() - start, 0, False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def run(url: str, total: int, concurrency: int, token: str | None, timeout: float) -> dict:
    questions = DEFAULT_QUESTIONS
    latencies: list[float] = []
    statuses: list[int] = []
    silences = 0
    lock = threading.Lock()
    wall_start = time.perf_counter()

    def worker(i: int) -> None:
        nonlocal silences
        elapsed, code, silence = _ask_once(url, questions[i % len(questions)], token, timeout)
        with lock:
            latencies.append(elapsed)
            statuses.append(code)
            if silence:
                silences += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(total)]
        for f in as_completed(futures):
            f.result()

    wall = time.perf_counter() - wall_start
    ok = sum(1 for c in statuses if c == 200)
    return {
        "url": url,
        "requests": total,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 3),
        "throughput_rps": round(total / wall, 3) if wall > 0 else 0.0,
        "ok": ok,
        "errors": total - ok,
        "error_rate": round((total - ok) / total, 4) if total else 0.0,
        "silenced": silences,
        "latency_p50_s": round(_percentile(latencies, 50), 3),
        "latency_p95_s": round(_percentile(latencies, 95), 3),
        "latency_p99_s": round(_percentile(latencies, 99), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
        "status_codes": dict(sorted({c: statuses.count(c) for c in set(statuses)}.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load-test a running Bank Management Workspace.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--token", default=None, help="Bearer token if the server requires read auth.")
    parser.add_argument("--output", default=None, help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    report = run(args.url, args.requests, args.concurrency, args.token, args.timeout)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
