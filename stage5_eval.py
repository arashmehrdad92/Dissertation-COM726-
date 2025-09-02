#!/usr/bin/env python3
"""
Stage 5 a€” True P@K evaluator for the dashboard (no pandas)

Reads a labels CSV (job_id,resume_id,labelaˆˆ{0,1}) and one or more
recommendation CSVs (Node2Vec/TF-IDF/BM25). Computes true P@K and writes:

    stage2_out/overall_metrics.csv
      source,p_at_10
      node2vec,0.1234
      tfidf,0.5678
      bm25,0.4321

Tolerant to UTF-8 BOM and Excel 'sep='; falls back to comma; no pandas.
"""

from __future__ import annotations
import argparse, os, csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ---------- Robust CSV reader (BOM + 'sep=' + sniff) ----------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(65536)
        f.seek(0)
        first = f.readline()
        sep_hint: Optional[str] = None
        if first.lower().startswith("sep="):
            raw = first.strip()[4:]
            sep_hint = "\t" if raw in ("\\t", "\t") else (raw[:1] if raw else None)
        else:
            f.seek(0)
        if sep_hint:
            delim = sep_hint
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                delim = dialect.delimiter
            except Exception:
                delim = ","
        rdr = csv.DictReader(f, delimiter=delim)
        for r in rdr:
            rows.append({(k or ""): ("" if v is None else str(v)) for k, v in r.items()})
    return rows

# ---------- P@K ----------
def precision_at_k(recs_path: str,
                   labels: Dict[Tuple[str,str], int],
                   k: int,
                   job_id_col: str = "job_id",
                   resume_id_col: str = "resume_id",
                   rank_col: str = "rank",
                   score_col: str = "score") -> Optional[float]:
    rows = read_csv_rows(recs_path)
    if not rows:
        return None
    # group by job; order by rank asc then score desc
    by_job: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    for r in rows:
        j = str(r.get(job_id_col, "")).strip()
        rid = str(r.get(resume_id_col, "")).strip()
        if not j or not rid:
            continue
        try:
            rk = int(r.get(rank_col, "") or 10**9)
        except Exception:
            rk = 10**9
        try:
            sc = float(r.get(score_col, "") or 0.0)
        except Exception:
            sc = 0.0
        by_job[j].append((rk, -sc, rid))
    if not by_job:
        return None
    s = 0
    n = 0
    for j, lst in by_job.items():
        lst.sort()
        top = lst[:k]
        s += sum(labels.get((j, rid), 0) for _,__,rid in top)
        n += k
    return (s / n) if n else None

def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 a€” compute true P@K and write overall_metrics.csv")
    ap.add_argument("--labels", required=True, help="Path to labels CSV (job_id,resume_id,label)")
    ap.add_argument("--node",   default="stage2_out/job_to_resumes_top10.csv", help="Node2Vec recs CSV")
    ap.add_argument("--tfidf",  default="stage2_out/job_to_resumes_top10_tfidf.csv", help="TF-IDF recs CSV")
    ap.add_argument("--bm25",   default="stage2_out/job_to_resumes_top10_bm25.csv", help="BM25 recs CSV")
    ap.add_argument("--out",    default="stage2_out/overall_metrics.csv", help="Output CSV for the dashboard")
    ap.add_argument("--k", type=int, default=10, help="K for precision@K (default 10)")
    # Optional column overrides
    ap.add_argument("--job_id_col", default="job_id")
    ap.add_argument("--resume_id_col", default="resume_id")
    ap.add_argument("--rank_col", default="rank")
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--label_col", default="label")
    args = ap.parse_args()

    # Load labels
    lab_rows = read_csv_rows(args.labels)
    if not lab_rows:
        print(f"ERROR: labels file empty or not found: {args.labels}")
        return 2
    labels: Dict[Tuple[str,str], int] = {}
    for r in lab_rows:
        j = str(r.get(args.job_id_col, "")).strip()
        rid = str(r.get(args.resume_id_col, "")).strip()
        try:
            lab = int(float(r.get(args.label_col, "0") or 0))
        except Exception:
            lab = 0
        if j and rid:
            labels[(j, rid)] = 1 if lab >= 1 else 0

    # Compute P@K for each model if file exists
    results: List[Tuple[str, Optional[float]]] = []
    for name, path in (("node2vec", args.node), ("tfidf", args.tfidf), ("bm25", args.bm25)):
        val = precision_at_k(path, labels, args.k, args.job_id_col, args.resume_id_col, args.rank_col, args.score_col) \
              if os.path.isfile(path) else None
        results.append((name, val))

    # Write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "p_at_10"])
        for name, val in results:
            if val is not None:
                w.writerow([name, f"{val:.4f}"])

    # Console summary
    print("P@%d results:" % args.k)
    for name, val in results:
        print(f"  {name:8s} : {'NA' if val is None else f'{val:.4f}'}")
    print(f"OK: wrote {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

