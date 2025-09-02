#!/usr/bin/env python3
"""
GraphMatch-AI a€“ Stage 2: Graph Construction & Embedding (with Error Handling)
================================================================================
Stage 2 loads sanitized data, builds a graph of resumes/jobs/skills, computes Node2Vec embeddings,
and generates top-K recommendations. Includes progress bars, timing, and robust error handling.
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
# pandas is deliberately avoided in this stage.  All CSV parsing is done via
# DictReader or custom shims to minimise dependencies and improve robustness.
from tqdm import tqdm
# Lazy import networkx.  When only baselines are executed, the absence of
# networkx (e.g. in environments where it is not installed) should not
# prevent the script from running.  Node2Vec and graph construction code
# will attempt to use ``nx`` only when needed.
try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # type: ignore
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed
# BOM/delimiter safe CSV helpers 

def _load_job2res(path: Path) -> Dict[str, List[str]]:
    """Load joba†’resumes as ordered top-K lists (rank asc, then score desc)."""
    try:
        rows, header = read_csv_dicts(str(path))
        if not rows:
            return {}
        by_job: Dict[str, List[Tuple[int, float, str]]] = {}
        for r in rows:
            jid = str(r.get("job_id", "")).strip()
            rid = str(r.get("resume_id", "")).strip()
            if not jid or not rid:
                continue
            try:
                rk = int(r.get("rank", "1000000000"))
            except Exception:
                rk = 1000000000
            try:
                sc = float(r.get("score", "0.0"))
            except Exception:
                sc = 0.0
            by_job.setdefault(jid, []).append((rk, -sc, rid))
        out: Dict[str, List[str]] = {}
        for j, items in by_job.items():
            items.sort()
            out[j] = [rid for _, __, rid in items]
        return out
    except Exception:
        return {}

def _overlap_at_10(A: Dict[str, List[str]], B: Dict[str, List[str]]) -> Optional[float]:
    jobs = set(A.keys()) & set(B.keys())
    if not jobs:
        return None
    hits = sum(len(set(A[j][:10]) & set(B[j][:10])) for j in jobs)
    return hits / (10 * len(jobs)) if jobs else None

def _coverage_full_k(m: Dict[str, List[str]], k: int) -> Tuple[int, int, float]:
    if not m:
        return (0, 0, 0.0)
    full = sum(1 for v in m.values() if len(v) >= k)
    return (full, len(m), (full / max(1, len(m))))

def _load_labels(path: Path) -> Dict[tuple, int]:
    """Load ground-truth labels: (job_id, resume_id) -> 0/1."""
    lab: Dict[tuple, int] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            j = str(r.get("job_id", "")).strip()
            ri = str(r.get("resume_id", "")).strip()
            lv = str(r.get("label", "0")).strip()
            if j and ri:
                lab[(j, ri)] = 1 if lv in ("1","true","True") else 0
    return lab

def _eval_job_accuracy(job2res: Dict[str, List[str]], labels: Dict[tuple, int], k: int) -> Dict[str, float]:
    """Compute P@k, nDCG@k, MAP@k across jobs that have at least one labeled item in top-k."""
    jobs_scored = 0
    hit_sum = 0
    ndcg_sum = 0.0
    ap_sum = 0.0
    import math

    for j, recs in job2res.items():
        rels = [labels.get((j, r), None) for r in recs[:k]]
        if all(v is None for v in rels):
            continue
        jobs_scored += 1

        # P@k
        hit_sum += sum(1 for v in rels if v == 1)

        # nDCG@k
        dcg = 0.0
        pos = 0
        for i, v in enumerate(rels, start=1):
            if v == 1:
                pos += 1
                dcg += 1.0 / math.log2(i + 1)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, pos + 1))
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        # MAP@k
        seen = 0
        precs = []
        for i, v in enumerate(rels, start=1):
            if v == 1:
                seen += 1
                precs.append(seen / i)
        ap_sum += (sum(precs) / len(precs)) if precs else 0.0

    if jobs_scored == 0:
        return {"jobs_scored": 0, "p_at_k": None, "ndcg_at_k": None, "map_at_k": None}

    return {
        "jobs_scored": jobs_scored,
        "p_at_k": hit_sum / (k * jobs_scored),
        "ndcg_at_k": ndcg_sum / jobs_scored,
        "map_at_k": ap_sum / jobs_scored,
    }

def _norm(s: str) -> str:
    """Strip BOM + whitespace from header/arg names."""
    return (s or "").lstrip("\ufeff").strip()

def _detect_delim_and_skip(path: str):
    """Return (delimiter, skiprows_for_sep_line). Handles BOM and Excel 'sep='."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(65536)
    first = sample.splitlines()[0] if sample else ""
    skip = 0
    if first.lower().startswith("sep="):
        raw = first[4:].strip()
        if raw in ("\\t", "\t"):
            return "\t", 1
        return (raw[:1] if raw else ","), 1
    # No explicit hint  sniff, then fall back to comma
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter, 0
    except Exception:
        return ",", 0

def read_csv_dicts(path: str):
    """
    Read a CSV as list[dict]: BOM-safe, Excel 'sep=' aware, delimiter sniffing.
    Header keys are normalized via _norm().
    """
    delim, skip = _detect_delim_and_skip(path)
    rows, header = [], None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # Skip 'sep=' line if present
        for _ in range(skip):
            f.readline()
        rdr = csv.reader(f, delimiter=delim)
        header = next(rdr, None)
        if header is None:
            return [], []
        header = [_norm(h) for h in header]
        for row in rdr:
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            rows.append({header[i]: row[i] for i in range(len(header))})
    return rows, header

# Configure logging
tk = logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GLOBAL_ARGS: argparse.Namespace | None = None

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for StageA 2.  In addition to the standard
    inputs (resumes, jobs, skill dictionaries, etc.) this function exposes a
    checkpoint directory and tuning parameters for concurrency.  If provided,
    intermediate objects (skill maps, embeddings) will be written to the
    checkpoint directory and reused on subsequent runs.  The ``workers`` and
    ``chunks_per_worker`` parameters control how the resume/job processing is
    parallelised.  Defaults are tuned to be conservative: the default number of
    workers is min(8, os.cpu_count()) and the default chunks per worker is 4.
    Additional arguments allow users to select which direction(s) of
    recommendations to produce and whether to generate TFa€‘IDF and/or BM25
    baseline rankings in lieu of computing node2vec embeddings.  A dedicated
    output directory can be specified to hold the various topa€‘K CSV files.
    """
    parser = argparse.ArgumentParser(
        description="Stage 2 a€“ Graph Construction & Embedding"
    )
    parser.add_argument("--resumes", required=True, help="Path to sanitized resumes CSV")
    parser.add_argument("--jobs", required=True, help="Path to sanitized jobs CSV")
    parser.add_argument("--skills_en", required=False, help="Path to ESCO skills CSV")
    parser.add_argument("--skills_txt", required=False, help="Path to O*NET skills TXT")
    parser.add_argument("--tech_txt", required=False, help="Path to technology skills TXT")
    parser.add_argument("--related", required=False, help="Path to related skills CSV")
    parser.add_argument("--resume_id_col", default="ID", help="Column name for resume IDs")
    parser.add_argument("--resume_text_col", default="Resume_str", help="Column name for resume text")
    parser.add_argument("--job_id_col", default="job_id", help="Column name for job IDs")
    parser.add_argument("--job_text_col", default="description", help="Column name for job text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations per entity")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=(
            "Number of worker processes to use for graph construction.  "
            "Defaults to min(8, available CPUs)."
        ),
    )
    parser.add_argument(
        "--chunks_per_worker",
        type=int,
        default=4,
        help=(
            "How many chunks to create per worker when parallelising resume/job "
            "processing.  Higher values improve load balance at the cost of "
            "slightly higher overhead.  Default 4 is conservative to reduce memory churn."
        ),
    )
    parser.add_argument(
        "--use_gpu_node2vec",
        action="store_true",
        help="Use PyTorch Geometric Node2Vec on the GPU if available",
    )
    parser.add_argument("--n2v_dim", type=int, default=128)
    parser.add_argument("--n2v_epochs", type=int, default=10)
    parser.add_argument("--n2v_walk_length", type=int, default=60)
    parser.add_argument("--n2v_context", type=int, default=20)
    parser.add_argument("--n2v_walks_per_node", type=int, default=20)
    parser.add_argument("--n2v_p", type=float, default=1.0)
    parser.add_argument("--n2v_q", type=float, default=0.5)
    parser.add_argument("--n2v_lr", type=float, default=0.01)
    parser.add_argument("--n2v_loader_bs", type=int, default=1024, help="Node2Vec loader batch size")
    parser.add_argument("--n2v_loader_workers", type=int, default=0, help="Node2Vec loader workers (Windows: often 0)")
    parser.add_argument("--n2v_early_stop_delta", type=float, default=0.005, help="Min rel. loss improvement to continue")
    parser.add_argument("--n2v_early_stop_patience", type=int, default=3, help="Epochs with < delta improvement before stop")

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help=(
            "Directory to save/load StageA 2 checkpoints.  If omitted, the output"
            " directory is used."
        ),
    )
    parser.add_argument(
        "--output",
        default="recommendations_stage2.csv",
        help=(
            "[Deprecated] Destination CSV file for recommendations.  "
            "If specified, will be used to derive the default output directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory to write topa€‘K CSV files.  If not provided, the parent of --output"
            " is used or the current working directory if --output is also omitted."
        ),
    )
    parser.add_argument(
        "--direction",
        default="both",
        choices=["resume_to_jobs", "job_to_resumes", "both"],
        help="Direction of ranking to output: resume_to_jobs, job_to_resumes, or both (default)",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help=(
            "Commaa€‘separated list of baseline methods to compute instead of node2vec. "
            "Supported baselines: tfidf, bm25.  If non-empty, node2vec embeddings will "
            "not be computed and only the specified baselines will run."
        ),
    )
    # Optional TFa€‘IDF GPU flags.  These flags allow the TFa€‘IDF baseline similarity
    # computation to be offloaded to a CUDAa€‘capable GPU via PyTorch.  When
    # ``--use_gpu_tfidf`` is enabled, the TFa€‘IDF vectors are still computed on the
    # CPU via scikita€‘learn, but similarity scoring is performed on the GPU in
    # batches.  ``--tfidf_gpu_chunk`` controls how many job postings are processed
    # per batch on the GPU to limit memory usage.  ``--tfidf_gpu_dtype`` selects
    # the floating point precision used on the GPU; half precision (float16)
    # reduces memory consumption at the cost of some numerical precision.
    parser.add_argument(
        "--use_gpu_tfidf",
        action="store_true",
        help=(
            "Enable GPU acceleration for the TFa€‘IDF baseline.  Requires PyTorch with CUDA "
            "support.  Falls back to the CPU path if CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--tfidf_gpu_chunk",
        type=int,
        default=2000,
        help=(
            "TFa€‘IDF GPU: number of job documents to process per batch when computing "
            "similarity on the GPU.  Smaller values reduce memory usage at the cost of "
            "more kernel launches.  Ignored when --use_gpu_tfidf is not set."
        ),
    )
    parser.add_argument(
        "--tfidf_gpu_dtype",
        default="float32",
        choices=["float32", "float16"],
        help=(
            "TFa€‘IDF GPU: floating point precision for GPU computations.  Options are "
            "float32 (single precision) or float16 (half precision).  Lower precision "
            "reduces memory footprint but may slightly affect similarity scores.  "
            "Ignored when --use_gpu_tfidf is not set."
        ),
    )
     # BM25 QoL flags
    parser.add_argument("--bm25_workers", type=int, default=4, help="BM25: parallel workers")
    parser.add_argument("--bm25_chunk",   type=int, default=500, help="BM25: tasks per batch")
    parser.add_argument("--bm25_ckpt",    type=str, default=None, help="BM25: checkpoint CSV path (.part)")
    parser.add_argument("--bm25_resume",  action="store_true", help="BM25: resume from checkpoint (skip done job_ids)")
    parser.add_argument(
    "--emit_overall_metrics",
    action="store_true",
    help="After writing outputs, compute quick eval (coverage + overlap@10) and write stage2_out/overall_metrics.csv"
)

    parser.add_argument(
        "--labels",
        default=None,
        help="Path to labels CSV with columns: job_id,resume_id,label (0/1). Enables true P@10/nDCG@10/MAP@10 in overall_metrics.",
    )

    return parser.parse_args()


def build_skill_dictionary(
    skills_en_path: Path,
    skills_txt_path: Path,
    tech_txt_path: Path,
    related_path: Path,
) -> Set[str]:
    """Load skill sets from multiple sources.  Uses tolerant CSV parsing and falls back to DictReader
    on failure.  Returns a set of lowercased skill strings.  Exits if no skills can be loaded."""
    skills: Set[str] = set()

    # Helper to load a skills file with candidate columns.  Returns list of strings.
    def _load_skills_file(path: Path, candidate_cols: List[str]) -> List[str]:
        """
        Load a skills CSV/TXT file using csv.DictReader only.  The function
        attempts to find the first matching column from ``candidate_cols``
        and returns lowera€‘cased values from that column.  If no candidate
        column is found, it falls back to the first column in the file.
        """
        vals: List[str] = []
        try:
            # For .txt files, O*NET and Technology Skills, enforce tab delimiter.
            delim = "\t" if str(path).lower().endswith(".txt") else None
            with open(path, "r", encoding="utf-8", newline="") as f:
                # Set delimiter only if specified; otherwise csv will default to comma.
                if delim:
                    reader = csv.DictReader(f, delimiter=delim)
                else:
                    reader = csv.DictReader(f)
                header_cols = reader.fieldnames or []
                col: Optional[str] = None
                # Choose first matching candidate column or fallback to first column
                for c in candidate_cols + (header_cols[:1] if header_cols else []):
                    if c in header_cols:
                        col = c
                        break
                if col is not None:
                    for row in reader:
                        try:
                            val = row.get(col)
                            if val:
                                vals.append(str(val).lower())
                        except Exception:
                            continue
        except Exception as ex:
            logger.error("Failed to load skills file %s: %s", path, ex)
        return [v.strip() for v in vals if v]

    # ESCO skills: preferredLabel column
    skills.update(_load_skills_file(skills_en_path, ["preferredLabel", "preferredlabel"]))
    # O*NET skills: Element Name column
    skills.update(_load_skills_file(skills_txt_path, ["Element Name", "element name"]))
    # Technology skills: Example column
    skills.update(_load_skills_file(tech_txt_path, ["Example", "example"]))
    # Related skills: always use DictReader; no pandas
    try:
        with open(related_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header_cols = reader.fieldnames or []
            cols = [c for c in header_cols if c.lower().startswith("related") or c.lower() in ("name", "skill", "skill1", "skill2")]
            for row in reader:
                for c in cols:
                    val = row.get(c)
                    if val:
                        skills.add(str(val).lower().strip())
    except Exception as ex:
        logger.error("Failed to load related skills file %s: %s", related_path, ex)

    if not skills:
        logger.error("No skills loaded; exiting.")
        sys.exit(1)
    return {s for s in skills if s}


def extract_skills(text: str, skills: Set[str]) -> List[str]:
    """
    Extract all skills present in the given text.  This routine attempts to use
    FlashText for higha€‘performance extraction when available.  When FlashText
    is not installed, it falls back to a compiled regular expression.  The
    pattern is rebuilt only when the input ``skills`` set changes.  To
    minimise false positives for very short tokens (C, R, Go) and tokens
    containing punctuation (e.g. C++, Node.js), word boundaries and looka€‘arounds
    are used where appropriate.  Hyphens, underscores and slashes in the text
    are normalised to spaces prior to matching, and multiple spaces are
    collapsed.  Returned skill names are lowera€‘cased.
    """
    try:
        if not text:
            return []
        # Normalise the text
        s = str(text).lower()
        # Convert common punctuation to space for matching
        s = re.sub(r"[-_/]", " ", s)
        s = re.sub(r"\s+", " ", s)
        # Attempt FlashText extraction if available
        if _flashtext_available:
            key = frozenset(skills)
            kp = _flash_cache.get(key)
            if kp is None:
                try:
                    kp = KeywordProcessor(case_sensitive=False)
                    # We add longer skills first to avoid shorter skills swallowing
                    for sk in sorted(skills, key=lambda x: (-len(x), x)):
                        kp.add_keyword(sk)
                    _flash_cache[key] = kp
                except Exception:
                    kp = None
                    _flash_cache[key] = None
            if kp is not None:
                try:
                    found = kp.extract_keywords(s)
                    # Dea€‘duplicate while preserving order
                    seen = set()
                    result = []
                    for f in found:
                        fl = f.lower()
                        if fl not in seen:
                            seen.add(fl)
                            result.append(fl)
                    return result
                except Exception:
                    # Fall back to regex
                    pass
        # Fallback: compiled regex
        # Build regex when skills set differs from previous
        if _regex_cache["skills_set"] != skills:
            pattern_parts: List[str] = []
            # Sort by length descending to match longest tokens first
            pattern_parts = []
            # Sort by length (desc) to match longest tokens first
            for sk in sorted(skills, key=lambda x: (-len(x), x)):
                escaped = re.escape(sk)

                # Very short tokens must stand alone
                if sk.lower() in {"c", "r", "go"}:
                    part = rf"(?<!\w){escaped}(?!\w)"
                else:
                    starts_non_alnum = not sk[:1].isalnum()
                    ends_non_alnum  = not sk[-1:].isalnum()

                    if ends_non_alnum and not starts_non_alnum:
                        # e.g., "C++", "C#", allow versions/suffix punctuation (C++17), but not letters
                        part = rf"(?<!\w){escaped}(?![A-Za-z_])"
                    elif starts_non_alnum and not ends_non_alnum:
                        # e.g., ".NET" a€” allow matches inside "ASP.NET", but block trailing letters
                        part = rf"{escaped}(?![A-Za-z_])"
                    elif starts_non_alnum and ends_non_alnum:
                        # rare: both ends non-alnum a€” enforce left boundary, block trailing letters
                        part = rf"(?<!\w){escaped}(?![A-Za-z_])"
                    else:
                        # normal words
                        part = rf"\b{escaped}\b"

                pattern_parts.append(part)

            try:
                combined = "|".join(pattern_parts) if pattern_parts else r"(?!x)x"  # safe empty regex
                _regex_cache["regex"] = re.compile(combined, flags=re.IGNORECASE)
                _regex_cache["skills_set"] = skills
            except Exception as e:
                logger.warning("Error compiling skill regex: %s", e)
                _regex_cache["regex"] = None
                _regex_cache["skills_set"] = None
        regex = _regex_cache.get("regex")
        if regex is None:
            # fallback naive scanning
            return [sk for sk in skills if sk in s]
        matches = regex.findall(s)
        if not matches:
            return []
        # Ensure unique list preserving order
        seen = set()
        result = []
        for m in matches:
            ml = m.lower()
            if ml not in seen:
                seen.add(ml)
                result.append(ml)
        return result
    except Exception as e:
        logger.warning("Error extracting skills: %s", e)
        return []


def _process_resume_chunk(
    args: Tuple[List[Dict[str, Any]], str, str, Set[str]]
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Worker function for processing a chunk of resumes.  It extracts skills
    from each resume and returns a dictionary of resume->skills and a list
    of edges (resume_id to skill).
    """
    records, rid_col, rtxt_col, skills = args
    local_resume_sk: Dict[str, List[str]] = {}
    edges: List[Tuple[str, str]] = []
    for row in records:
        rid = str(row[rid_col])
        sks = extract_skills(row.get(rtxt_col, ""), skills)
        local_resume_sk[rid] = sks
        for sk in sks:
            edges.append((f"resume_{rid}", f"skill_{sk}"))
    return local_resume_sk, edges


def _process_job_chunk(
    args: Tuple[List[Dict[str, Any]], str, str, Set[str]]
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Worker function for processing a chunk of jobs.  Similar to
    _process_resume_chunk but for job postings.
    """
    records, jid_col, jtxt_col, skills = args
    local_job_sk: Dict[str, List[str]] = {}
    edges: List[Tuple[str, str]] = []
    for row in records:
        jid = str(row[jid_col])
        sks = extract_skills(row.get(jtxt_col, ""), skills)
        local_job_sk[jid] = sks
        for sk in sks:
            edges.append((f"job_{jid}", f"skill_{sk}"))
    return local_job_sk, edges


def build_graph(
    resumes_df: Any,
    jobs_df: Any,
    skills: Set[str],
    rid_col: str,
    rtxt_col: str,
    jid_col: str,
    jtxt_col: str,
    workers: int = 1,
    chunks_per_worker: int = 4,
) -> Tuple[nx.Graph, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Construct the heterogeneous graph linking resumes, jobs and skills.  When
    ``workers`` is greater than one, this function parallelises the extraction
    of skills and edge generation using ``ProcessPoolExecutor``.  To improve
    load balancing and avoid idle cores near the end of processing, the
    ``chunks_per_worker`` parameter controls how many chunks of records are
    created per worker.  A higher value yields more, smaller chunks and
    typically results in better CPU utilisation at the cost of slightly
    increased scheduling overhead.  The resulting graph contains nodes of
    type "resume", "job", and "skill"; node_type attributes are assigned
    after all edges are added.  This function returns the graph and the
    pera€‘resume/job skill dictionaries.
    """
    total_start = time.time()
    num_resumes = len(resumes_df)
    num_jobs = len(jobs_df)
    # Prepare containers
    resume_sk: Dict[str, List[str]] = {}
    job_sk: Dict[str, List[str]] = {}
    edges: List[Tuple[str, str]] = []
    # Convert dataframes to list of records for multiprocessing
    resume_records = resumes_df.to_dict("records")
    job_records = jobs_df.to_dict("records")
    # Helper: split a list into a specified number of chunks
    def chunkify_n(lst: List[Dict[str, Any]], n_chunks: int) -> List[List[Dict[str, Any]]]:
        """Split ``lst`` into ``n_chunks`` nearly equal parts."""
        if n_chunks <= 1 or len(lst) <= 1:
            return [lst]
        chunk_size = ceil(len(lst) / n_chunks)
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    

    # Process resumes
    start = time.time()
    if workers and workers > 1 and num_resumes > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from collections import defaultdict
        n_chunks = max(1, min(len(resume_records), workers * chunks_per_worker))
        chunks = chunkify_n(resume_records, n_chunks)
        payloads = [(chunk, rid_col, rtxt_col, skills) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=workers) as executor, \
             tqdm(total=num_resumes, desc="Resumes", unit="rows",
                  dynamic_ncols=True, smoothing=0.1,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            fut_sizes = {}
            futures = []
            for chunk, pl in zip(chunks, payloads):
                fut = executor.submit(_process_resume_chunk, pl)
                futures.append(fut)
                fut_sizes[fut] = len(chunk)
            for fut in as_completed(futures):
                local_sk, local_edges = fut.result()
                resume_sk.update(local_sk)
                edges.extend(local_edges)
                pbar.update(fut_sizes.pop(fut, 0))
                # free local lists early
                del local_sk, local_edges
    else:
        for record in tqdm(resume_records, desc="Resumes", unit="rows",
                           dynamic_ncols=True, smoothing=0.1,
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            rid = str(record[rid_col])
            sks = extract_skills(record.get(rtxt_col, ""), skills)
            resume_sk[rid] = sks
            for sk in sks:
                edges.append((f"resume_{rid}", f"skill_{sk}"))
    dur = time.time() - start
    rate = num_resumes / dur if dur > 0 else 0
    logger.info("Processed %d resumes in %.2f sec (%.2f/sec)", num_resumes, dur, rate)

    # Process jobs
    start = time.time()
    if workers and workers > 1 and num_jobs > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        n_chunks = max(1, min(len(job_records), workers * chunks_per_worker))
        chunks = chunkify_n(job_records, n_chunks)
        payloads = [(chunk, jid_col, jtxt_col, skills) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=workers) as executor, \
             tqdm(total=num_jobs, desc="Jobs", unit="rows",
                  dynamic_ncols=True, smoothing=0.1,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            fut_sizes = {}
            futures = []
            for chunk, pl in zip(chunks, payloads):
                fut = executor.submit(_process_job_chunk, pl)
                futures.append(fut)
                fut_sizes[fut] = len(chunk)
            for fut in as_completed(futures):
                local_sk, local_edges = fut.result()
                job_sk.update(local_sk)
                edges.extend(local_edges)
                pbar.update(fut_sizes.pop(fut, 0))
                del local_sk, local_edges
    else:
        for record in tqdm(job_records, desc="Jobs", unit="rows",
                           dynamic_ncols=True, smoothing=0.1,
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            jid = str(record[jid_col])
            sks = extract_skills(record.get(jtxt_col, ""), skills)
            job_sk[jid] = sks
            for sk in sks:
                edges.append((f"job_{jid}", f"skill_{sk}"))
    dur_jobs = time.time() - start
    rate_jobs = num_jobs / dur_jobs if dur_jobs > 0 else 0
    logger.info("Processed %d jobs in %.2f sec (%.2f/sec)", num_jobs, dur_jobs, rate_jobs)

    # Build graph from edges
    G = nx.Graph()
    # Pre-add resume and job nodes so zero-skill rows still exist in the graph
    for rid in resumes_df[rid_col].astype(str):
        G.add_node(f"resume_{rid}", node_type="resume")
    for jid in jobs_df[jid_col].astype(str):
        G.add_node(f"job_{jid}", node_type="job")
    G.add_edges_from(edges)
    # Assign node types
    for rid in resume_sk.keys():
        G.nodes[f"resume_{rid}"]["node_type"] = "resume"
    for jid in job_sk.keys():
        G.nodes[f"job_{jid}"]["node_type"] = "job"
    # Skills appear in both resumes and jobs
    for _, sks in resume_sk.items():
        for sk in sks:
            G.nodes[f"skill_{sk}"]["node_type"] = "skill"
    for _, sks in job_sk.items():
        for sk in sks:
            G.nodes[f"skill_{sk}"]["node_type"] = "skill"
    logger.info(
        "Graph constructed with %d nodes and %d edges in %.2f sec",
        G.number_of_nodes(), G.number_of_edges(), time.time() - total_start
    )
    return G, resume_sk, job_sk


def compute_embeddings(
    
    G: nx.Graph,
    workers: int = 1,
    use_gpu: bool = False,
    args=None
) -> Tuple[Dict[str, np.ndarray], bool, bool]:
    """
    Compute lowa€‘dimensional embeddings for every node in the graph.  When
    ``use_gpu`` is True and a compatible CUDA device is available, a
    PyTorch Geometric implementation of Node2Vec is used; otherwise the
    gensim/Node2Vec CPU implementation is invoked.  Returns a tuple of
    (embeddings, gpu_used, gpu_unavailable).  The number of ``workers`` controls
    negative sampling and walk generation on CPU and the DataLoader worker count
    on GPU.
    """
    start = time.time()
    embeddings: Dict[str, np.ndarray] = {}
    gpu_used: bool = False
    gpu_unavailable: bool = False
    # Attempt GPU Node2Vec if requested
    if use_gpu:
        try:
            import torch  # type: ignore
            from torch_geometric.nn.models import Node2Vec as TGNode2Vec  # type: ignore
            # Check CUDA availability
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                raise RuntimeError("CUDA not available")
            # Set seeds for reproducibility
            import random as _random
            _random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            try:
                torch.cuda.manual_seed_all(42)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Build mapping from nodes to indices
            node_list = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            # Convert undirected graph edges to directed edges for Node2Vec
            src, dst = [], []
            for u, v in G.edges():
                ui, vi = node_to_idx[u], node_to_idx[v]
                src.extend([ui, vi])
                dst.extend([vi, ui])
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            # Initialise Node2Vec model (CLI-tunable, with safe fallbacks)
            # NOTE: This block does not require changing the function signature.
            # If 'args' exists, we read --n2v_* flags; otherwise we use the defaults below.
            _a = locals().get("args", None) or globals().get("GLOBAL_ARGS", None)
            def _g(name, default):
                try:
                    return getattr(_a, name)
                except Exception:
                    return default

            n2v_dim            = _g("n2v_dim",            64)
            n2v_walk_length    = _g("n2v_walk_length",    10)
            n2v_context        = _g("n2v_context",         5)
            n2v_walks_per_node = _g("n2v_walks_per_node", 20)
            n2v_p              = _g("n2v_p",             1.0)
            n2v_q              = _g("n2v_q",             1.0)
            n2v_lr             = _g("n2v_lr",           0.01)
            n2v_epochs         = _g("n2v_epochs",          5)

            model = TGNode2Vec(
                edge_index=edge_index,
                embedding_dim=n2v_dim,
                walk_length=n2v_walk_length,
                context_size=n2v_context,
                walks_per_node=n2v_walks_per_node,
                num_negative_samples=1,
                p=n2v_p,
                q=n2v_q,
                sparse=True,
            ).to(device)

            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=n2v_lr)

            # Train for multiple epochs (with a progress bar per epoch)
            from tqdm import tqdm  # already imported earlier in the file; harmless if repeated
            epochs = n2v_epochs
            for epoch in range(epochs):
                if epoch == 0:
                    _loss_hist = []
                epoch_start = time.time()
                model.train()
                total_loss: float = 0.0
                
                # num_workers=0 avoids Windows spawn issues
                bs = _g("n2v_batch", 128)
                w  = _g("n2v_loader_workers", 0)
                loader = model.loader(
                    batch_size=getattr(args, "n2v_loader_bs", 1024),
                    shuffle=True,
                    num_workers=getattr(args, "n2v_loader_workers", 0),
                    pin_memory=(device.type == "cuda"),
                    persistent_workers=(getattr(args, "n2v_loader_workers", 0) > 0),
                )

                for pos_rw, neg_rw in tqdm(
                    loader,
                    total=len(loader),
                    desc=f"GPU Node2Vec epoch {epoch+1}/{epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    smoothing=0.1,
                    leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ):
                    pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)
                
                _loss_hist.append(total_loss)
                logger.info(
                    "GPU Node2Vec epoch %d/%d completed; loss %.4f (%.2f sec)",
                    epoch + 1, epochs, total_loss, time.time() - epoch_start,
                )

                # --- Early stop: break if recent improvement is tiny ---
                if len(_loss_hist) >= max(2, getattr(args, "n2v_early_stop_patience", 3) + 1):
                    prev = _loss_hist[-(getattr(args, "n2v_early_stop_patience", 3) + 1)]
                    now  = _loss_hist[-1]
                    # relative improvement vs the value 'patience' epochs ago
                    rel_impr = (prev - now) / max(abs(prev), 1e-9)
                    if rel_impr < getattr(args, "n2v_early_stop_delta", 0.005):
                        logger.info("Early stop: relative improvement %.4f < %.4f for %d epochs",
                                    rel_impr,
                                    getattr(args, "n2v_early_stop_delta", 0.005),
                                    getattr(args, "n2v_early_stop_patience", 3))
                        break


            # Extract embeddings
            emb = model.embedding.weight.detach().cpu().numpy()
            embeddings = {node: emb[idx] for node, idx in node_to_idx.items()}
            
            logger.info("GPU embeddings computed in %.2f sec", time.time() - start)
            gpu_used = True
        except Exception as e:
            logger.warning("GPU Node2Vec failed (%s); will attempt CPU implementation", e)
            gpu_unavailable = True
            embeddings = {}
    # CPU fallback using gensim/node2vec
    if not embeddings:
        try:
            # Set seeds for reproducibility
            import random as _random
            _random.seed(42)
            np.random.seed(42)
            # Lazy import to avoid SciPy crash unless needed
            from node2vec import Node2Vec
            node2vec = Node2Vec(
                G,
                dimensions=64,
                walk_length=10,
                num_walks=20,
                workers=max(1, workers),
            )
            model = node2vec.fit(window=5, min_count=1, epochs=10)
            embeddings = {n: model.wv[n] for n in G.nodes()}
            logger.info("CPU embeddings computed in %.2f sec", time.time() - start)
        except Exception as e:
            logger.error("Error computing embeddings: %s", e)
            sys.exit(1)
    # L2 normalize all embeddings so dot product equals cosine similarity
    for k, v in list(embeddings.items()):
        try:
            norm = np.linalg.norm(v)
            if norm > 0:
                embeddings[k] = v / norm
        except Exception:
            continue
    return embeddings, gpu_used, gpu_unavailable


def top_k_recommendations(
    embeddings: Dict[str, np.ndarray],
    resume_ids: List[str],
    job_ids: List[str],
    top_k: int,
    resume_sk: Dict[str, List[str]],
    job_sk: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, float, str]]]:
    results: Dict[str, List[Tuple[str, float, str]]] = {}
    start = time.time()
    for rid in tqdm(resume_ids, desc="Computing recommendations"):
        try:
            r_vec = embeddings.get(f"resume_{rid}")
            if r_vec is None:
                continue
            scores: List[Tuple[str, float]] = []
            for jid in job_ids:
                j_vec = embeddings.get(f"job_{jid}")
                if j_vec is None:
                    continue
                scores.append((jid, float(np.dot(r_vec, j_vec))))
            # Sort by similarity descending, then by job_id to break ties deterministically
            scores.sort(key=lambda x: (-x[1], x[0]))
            enriched: List[Tuple[str, float, str]] = []
            for jid, sc in scores[:top_k]:
                missing = set(job_sk.get(jid, [])) - set(resume_sk.get(rid, []))
                enriched.append((jid, sc, "|".join(sorted(missing))))
            results[rid] = enriched
        except Exception as e:
            logger.warning("Error computing recommendations for %s: %s", rid, e)
    logger.info("Recommendations computed in %.2f sec", time.time() - start)
    return results


# --- New helpers for vectorised topa€‘K recommendations ---
def vectorized_top_k_both(
    embeddings: Dict[str, np.ndarray],
    resume_ids: List[str],
    job_ids: List[str],
    top_k: int,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]:
    """Compute top-K similarity both ways using cosine (vectors are L2-normalized)."""

    # Build matrices with the correct embedding keys ("resume_<id>", "job_<id>")
    r_vecs, valid_resume_ids = [], []
    for rid in resume_ids:
        v = embeddings.get(f"resume_{rid}")
        if v is not None:
            r_vecs.append(v)
            valid_resume_ids.append(rid)

    j_vecs, valid_job_ids = [], []
    for jid in job_ids:
        v = embeddings.get(f"job_{jid}")
        if v is not None:
            j_vecs.append(v)
            valid_job_ids.append(jid)

    if not r_vecs or not j_vecs:
        return {}, {}

    R = np.stack(r_vecs)          # (R, d)
    J = np.stack(j_vecs)          # (J, d)
    scores = R.dot(J.T)           # (R, J)

    kJ = min(top_k, len(valid_job_ids))
    kR = min(top_k, len(valid_resume_ids))

    resume_to_jobs: Dict[str, List[Tuple[str, float]]] = {}
    job_to_resumes: Dict[str, List[Tuple[str, float]]] = {}

    job_ids_arr = np.array(valid_job_ids, dtype=object)
    res_ids_arr = np.array(valid_resume_ids, dtype=object)

    # ---- resume a†’ jobs ----
    for i, rid in enumerate(valid_resume_ids):
        row = scores[i]
        if kJ == 0:
            resume_to_jobs[rid] = []
            continue
        top_idx = np.argpartition(-row, kJ - 1)[:kJ]
        # tie-break: score desc, then job_id asc
        order = np.lexsort((job_ids_arr[top_idx], -row[top_idx]))
        idxs = top_idx[order].tolist()
        resume_to_jobs[rid] = [(valid_job_ids[int(j)], float(row[int(j)])) for j in idxs]

    # ---- jobs a†’ resumes ----
    for j, jid in enumerate(valid_job_ids):
        col = scores[:, j]
        if kR == 0:
            job_to_resumes[jid] = []
            continue
        top_idx = np.argpartition(-col, kR - 1)[:kR]
        # tie-break: score desc, then resume_id asc
        order = np.lexsort((res_ids_arr[top_idx], -col[top_idx]))
        idxs = top_idx[order].tolist()
        job_to_resumes[jid] = [(valid_resume_ids[int(i)], float(col[int(i)])) for i in idxs]

    return resume_to_jobs, job_to_resumes


def write_topk_csv(
    path: Path,
    records: List[Tuple[str, str, float, int, str]],
    header: List[str],
) -> None:
    """
    Write a list of recommendation records to a CSV file.  Each record is a
    tuple corresponding to header columns.  This helper ensures robust
    file writing with UTFa€‘8 encoding.
    """
    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(records)
    except Exception as e:
        logger.error("Error writing %s: %s", path, e)
        sys.exit(1)


def compute_graph_hash(G: nx.Graph) -> str:
    """
    Compute a simple hash of the graph based on its edges.  Edges are sorted
    lexicographically and concatenated prior to hashing with MD5.  Node types
    are not considered in the hash; two graphs with identical undirected edge
    sets will yield the same hash.
    """
    import hashlib
    edge_strings = [f"{min(u, v)}-{max(u, v)}" for u, v in G.edges()]
    edge_strings.sort()
    data = ",".join(edge_strings).encode("utf-8")
    return hashlib.md5(data).hexdigest()


def compute_tfidf_baseline(
    resume_texts: List[str],
    job_texts: List[str],
    resume_ids: List[str],
    job_ids: List[str],
    top_k: int,
    *,
    use_gpu: bool = False,
    gpu_chunk: int = 2000,
    gpu_dtype: str = "float32",
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]:
    """
    Compute TFa€‘IDF cosine similarity baseline rankings in both directions.

    This routine uses scikita€‘learn's :class:`TfidfVectorizer` to build a shared
    vocabulary across resumes and job postings, then computes pairwise cosine
    similarities.  When ``use_gpu`` is False, the computation is performed
    entirely on the CPU using dense NumPy arrays (the original implementation).

    When ``use_gpu`` is True and a CUDAa€‘capable PyTorch installation is
    available, only the similarity scoring is offloaded to the GPU.  The
    TFa€‘IDF vectors are still generated on the CPU.  Similarities are
    computed in batches of size ``gpu_chunk`` to limit memory usage.  If
    PyTorch is not installed or CUDA is unavailable, the function logs
    a message and falls back to the CPU implementation.

    Parameters
    ----------
    resume_texts : list of str
        Sanitized resume texts.
    job_texts : list of str
        Sanitized job posting texts.
    resume_ids : list of str
        Identifier corresponding to each entry in ``resume_texts``.
    job_ids : list of str
        Identifier corresponding to each entry in ``job_texts``.
    top_k : int
        Number of top recommendations to return for each entity.
    use_gpu : bool, optional
        If True, attempt to compute similarities on the GPU using PyTorch.
        Defaults to False.
    gpu_chunk : int, optional
        When using the GPU, the number of job postings to process per
        batch.  Smaller values reduce memory usage.  Ignored when
        ``use_gpu`` is False.  Defaults to 2000.
    gpu_dtype : {'float32', 'float16'}, optional
        Floating point precision for GPU computations.  Lower precision
        (``float16``) reduces memory usage at the cost of some numerical
        precision.  Ignored when ``use_gpu`` is False.  Defaults to
        ``float32``.

    Returns
    -------
    resume_to_jobs : dict
        Mapping of each resume ID to a list of (job_id, score) pairs
        representing the top ``top_k`` most similar job postings.
    job_to_resumes : dict
        Mapping of each job ID to a list of (resume_id, score) pairs
        representing the top ``top_k`` most similar resumes.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        import sklearn  # noqa: F401
    except Exception as e:
        logger.error("scikita€‘learn is required for TFa€‘IDF baseline: %s", e)
        raise

    # Build a single vectorizer for all texts and fit it
    all_texts = resume_texts + job_texts
    vectorizer = TfidfVectorizer(stop_words=None)
    t0_fit = time.time()
    tfidf = vectorizer.fit_transform(all_texts)
    logger.info("TFa€‘IDF: fitted vectorizer on %d docs in %.2f sec",
            len(all_texts), time.time() - t0_fit)
    # Split into resume and job matrices (sparse CSR)
    R = tfidf[0 : len(resume_texts)]
    J = tfidf[len(resume_texts) :]

    # Fallback threshold: if no GPU usage requested, or PyTorch/CUDA is unavailable,
    # or an error occurs, we execute the original dense CPU code path.  The GPU
    # implementation uses sparse matrix multiplication to compute similarities
    # in batches without densifying the full document-term matrix.
    if use_gpu:
        try:
            import torch  # type: ignore

            # Verify CUDA availability; if unavailable, raise to trigger fallback
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            # Map dtype string to PyTorch dtype
            dtype_map = {"float32": torch.float32, "float16": torch.float16}
            torch_dtype = dtype_map.get(gpu_dtype, torch.float32)
            device = torch.device("cuda")

            n_resumes = R.shape[0]
            n_jobs = J.shape[0]

            # Convert resume matrix to CSR on CPU and then to a PyTorch sparse CSR tensor on GPU
            R_csr = R.tocsr()
            data_tensor = torch.tensor(R_csr.data, dtype=torch_dtype, device=device)
            indices_tensor = torch.tensor(R_csr.indices, dtype=torch.int64, device=device)
            indptr_tensor = torch.tensor(R_csr.indptr, dtype=torch.int64, device=device)
            R_sparse_t = torch.sparse_csr_tensor(indptr_tensor, indices_tensor, data_tensor,
                                                 size=R_csr.shape, dtype=torch_dtype, device=device)

            import heapq
            # Prepare mina€‘heaps for each resume and job.  Heaps store (score, index) pairs.
            resume_heaps: List[List[Tuple[float, int]]] = [[] for _ in range(n_resumes)]
            job_heaps: List[List[Tuple[float, int]]] = [[] for _ in range(n_jobs)]

            # Process jobs in batches to limit GPU memory usage
            for j_start in tqdm(range(0, n_jobs, max(1, gpu_chunk)),
                    total=math.ceil(n_jobs / max(1, gpu_chunk)),
                    desc="TFa€‘IDF GPU scoring", unit="blocks",
                    dynamic_ncols=True, smoothing=0.1,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                j_end = min(j_start + max(1, gpu_chunk), n_jobs)
                J_block_csr = J[j_start:j_end].tocsr()
                # Convert current block of jobs to dense matrix on the CPU, then move to GPU
                J_block_dense_np = J_block_csr.toarray().astype(np.float32 if torch_dtype == torch.float32 else np.float16)
                J_block_dense = torch.tensor(J_block_dense_np, dtype=torch_dtype, device=device)
                # Compute sparsea€‘dense matrix multiplication: (n_resumes x n_terms) * (n_terms x block_size)
                # Yields a dense (n_resumes x block_size) matrix of similarities
                with torch.no_grad():
                    block_scores = torch.sparse.mm(R_sparse_t, J_block_dense.T)
                # Move results back to CPU as NumPy array for heap processing
                block_scores_cpu = block_scores.cpu().numpy()
                # Iterate over resumes and jobs in the current block to update heaps
                for i in range(n_resumes):
                    row_scores = block_scores_cpu[i]
                    rh = resume_heaps[i]
                    for bj, score in enumerate(row_scores):
                        j_idx = j_start + bj
                        # Update resume heap with (score, j_idx)
                        if len(rh) < top_k:
                            heapq.heappush(rh, (float(score), j_idx))
                        else:
                            # Replace smallest if current score is higher or tiea€‘break on job ID
                            if (score > rh[0][0]) or (score == rh[0][0] and job_ids[j_idx] < job_ids[rh[0][1]]):
                                heapq.heappushpop(rh, (float(score), j_idx))
                        # Update job heap with (score, i)
                        jh = job_heaps[j_idx]
                        if len(jh) < top_k:
                            heapq.heappush(jh, (float(score), i))
                        else:
                            if (score > jh[0][0]) or (score == jh[0][0] and resume_ids[i] < resume_ids[jh[0][1]]):
                                heapq.heappushpop(jh, (float(score), i))
                # Explicitly free GPU memory for the current block before moving on
                del J_block_dense
                del block_scores
                del block_scores_cpu
                torch.cuda.empty_cache()

            # Construct result dictionaries from heaps.  Sort in descending order of score and
            # ascending order of identifier to preserve deterministic tiea€‘break behaviour.
            resume_to_jobs: Dict[str, List[Tuple[str, float]]] = {}
            for i, rid in enumerate(tqdm(resume_ids, desc="TFa€‘IDF GPU a†’ resume_to_jobs",
                             unit="resumes", dynamic_ncols=True, smoothing=0.1,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):
                h = resume_heaps[i]
                h_list = list(h)
                # Sort primarily by negative score, secondarily by job ID
                h_list.sort(key=lambda x: (-x[0], job_ids[x[1]]))
                recs: List[Tuple[str, float]] = []
                for score, j_idx in h_list[:top_k]:
                    recs.append((job_ids[j_idx], float(score)))
                resume_to_jobs[rid] = recs
            job_to_resumes: Dict[str, List[Tuple[str, float]]] = {}
            for j_idx, jid in enumerate(tqdm(job_ids, desc="TFa€‘IDF GPU a†’ job_to_resumes",
                                 unit="jobs", dynamic_ncols=True, smoothing=0.1,
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):
                h = job_heaps[j_idx]
                h_list = list(h)
                h_list.sort(key=lambda x: (-x[0], resume_ids[x[1]]))
                recs: List[Tuple[str, float]] = []
                for score, i_idx in h_list[:top_k]:
                    recs.append((resume_ids[i_idx], float(score)))
                job_to_resumes[jid] = recs
            return resume_to_jobs, job_to_resumes
        except Exception as e:
            # Any failure in GPU code falls back to CPU path.  Use INFO level to avoid
            # treating fallback as an error in baseline reporting.
            logger.info("TFa€‘IDF GPU acceleration disabled due to: %s", e)
            # Fall through to CPU path

    # CPU implementation (original).  Convert sparse matrices to dense and compute dot product.
    t0_cpu = time.time()

    # Warning: this may consume significant memory for large corpora.
    R_dense = R.toarray()
    J_dense = J.toarray()
    # Compute similarity via dense dot product; since TFa€‘IDF rows are L2 normalised,
    # the dot product equals cosine similarity.
    sim = R_dense.dot(J_dense.T)
    logger.info("TFa€‘IDF CPU: computed dense similarity (%dA—%d) in %.2f sec",
            R_dense.shape[0], J_dense.shape[0], time.time() - t0_cpu)

    resume_to_jobs: Dict[str, List[Tuple[str, float]]] = {}
    job_to_resumes: Dict[str, List[Tuple[str, float]]] = {}
    # Resumea†’jobs: build list of (score, job_id) pairs and sort descending by score then ascending by job_id
    for i, rid in enumerate(tqdm(resume_ids, desc="TFa€‘IDF CPU a†’ resume_to_jobs",
                             unit="resumes", dynamic_ncols=True, smoothing=0.1,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):

        row = sim[i]
        # Create list of tuples (score, job_id) for stable sorting
        pairs = [(float(row[j]), job_ids[j]) for j in range(len(job_ids))]
        # Sort by score descending, then job_id ascending for deterministic tiea€‘break
        pairs.sort(key=lambda x: (-x[0], x[1]))
        recs: List[Tuple[str, float]] = []
        for score, jid in pairs[:top_k]:
            recs.append((jid, float(score)))
        resume_to_jobs[rid] = recs
    # Joba†’resumes: build list of (score, resume_id) pairs and sort similarly
    # Access column j across all resumes
    for j, jid in enumerate(tqdm(job_ids, desc="TFa€‘IDF CPU a†’ job_to_resumes",
                             unit="jobs", dynamic_ncols=True, smoothing=0.1,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")):

        col = sim[:, j]
        pairs = [(float(col[i]), resume_ids[i]) for i in range(len(resume_ids))]
        pairs.sort(key=lambda x: (-x[0], x[1]))
        recs: List[Tuple[str, float]] = []
        for score, rid in pairs[:top_k]:
            recs.append((rid, float(score)))
        job_to_resumes[jid] = recs
    return resume_to_jobs, job_to_resumes


def compute_bm25_baseline(
    resume_texts: List[str],
    job_texts: List[str],
    resume_ids: List[str],
    job_ids: List[str],
    top_k: int,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]:
    """
    Compute BM25 similarity baseline rankings in both directions.  A simple
    implementation is used to avoid external dependencies.  The corpus for
    computing IDF and document lengths is based on the set of documents
    corresponding to the target side (e.g. when scoring resumes against jobs,
    the job texts form the document corpus).  Tokens are extracted using
    word characters; no stemming or stopa€‘word removal is applied.  Returns
    (resume_to_jobs, job_to_resumes).
    """
    import math
    # Tokenise documents
    def tokenize(t: str) -> List[str]:
        return re.findall(r"\b\w+\b", t.lower())
    # Precompute for jobs
    job_tokens: List[List[str]] = [tokenize(t) for t in job_texts]
    # Document frequency across jobs
    df_job: Dict[str, int] = {}
    for toks in job_tokens:
        for w in set(toks):
            df_job[w] = df_job.get(w, 0) + 1
    N_jobs = len(job_tokens)
    avgdl_jobs = sum(len(t) for t in job_tokens) / (N_jobs or 1)
    # Precompute IDF for jobs
    idf_job: Dict[str, float] = {}
    for w, df in df_job.items():
        # Add 1 to numerator and denominator for smoothing
        idf_job[w] = math.log((N_jobs - df + 0.5) / (df + 0.5) + 1)
    # Precompute term frequencies per job
    tf_job: List[Dict[str, int]] = []
    for toks in job_tokens:
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        tf_job.append(tf)
    # Resumea†’jobs: treat resumes as queries and jobs as documents
    k1 = 1.5; b = 0.75
    resume_to_jobs: Dict[str, List[Tuple[str, float]]] = {}
    for rid, qtext in zip(resume_ids, resume_texts):
        q_tokens = tokenize(qtext)
        # Term frequencies in query (not used by BM25; queries treated as bag of words)
        scores_row = []  # list of floats aligned with job_ids
        for j_idx, (doc_tokens, tf_dict) in enumerate(zip(job_tokens, tf_job)):
            score = 0.0
            doc_len = len(doc_tokens)
            for t in q_tokens:
                if t not in tf_dict:
                    continue
                f = tf_dict[t]
                idf = idf_job.get(t, 0.0)
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / (avgdl_jobs or 1)))
            scores_row.append(score)
        # Determine top K (stable tie-break: score desc, then job_id asc)
        scores_arr = np.asarray(scores_row, dtype=np.float32)
        job_ids_arr = np.asarray(job_ids, dtype=object)
        k = min(top_k, len(job_ids_arr))
        if k == len(job_ids_arr):
            idxs = np.argsort(-scores_arr, kind="mergesort")
        else:
            partial = np.argpartition(-scores_arr, k - 1)[:k].astype(np.int64, copy=False)
            order = np.lexsort((np.take(job_ids_arr, partial), -np.take(scores_arr, partial)))
            idxs = np.take(partial, order)
        recs: List[Tuple[str, float]] = []
        for j_idx in idxs[:top_k].tolist():
            j = int(j_idx)
            recs.append((str(job_ids_arr[j]), float(scores_arr[j])))
        resume_to_jobs[rid] = recs
    # Now compute joba†’resumes: treat jobs as queries and resumes as documents
    # Precompute for resumes
    resume_tokens: List[List[str]] = [tokenize(t) for t in resume_texts]
    df_res: Dict[str, int] = {}
    for toks in resume_tokens:
        for w in set(toks):
            df_res[w] = df_res.get(w, 0) + 1
    N_res = len(resume_tokens)
    avgdl_res = sum(len(t) for t in resume_tokens) / (N_res or 1)
    idf_res: Dict[str, float] = {}
    for w, df in df_res.items():
        idf_res[w] = math.log((N_res - df + 0.5) / (df + 0.5) + 1)
    tf_res: List[Dict[str, int]] = []
    for toks in resume_tokens:
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        tf_res.append(tf)
    job_to_resumes: Dict[str, List[Tuple[str, float]]] = {}
    for jid, qtext in zip(job_ids, job_texts):
        q_tokens = tokenize(qtext)
        scores_row: List[float] = []
        for i_idx, (doc_tokens, tf_dict) in enumerate(zip(resume_tokens, tf_res)):
            score = 0.0
            doc_len = len(doc_tokens)
            for t in q_tokens:
                if t not in tf_dict:
                    continue
                f = tf_dict[t]
                idf = idf_res.get(t, 0.0)
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / (avgdl_res or 1)))
            scores_row.append(score)
        scores_arr = np.array(scores_row, dtype=float)
        scores_arr = np.asarray(scores_row, dtype=np.float32)
        resume_ids_arr = np.asarray(resume_ids, dtype=object)
        k = min(top_k, len(resume_ids_arr))
        if k == len(resume_ids_arr):
            idxs = np.argsort(-scores_arr, kind="mergesort")
        else:
            partial = np.argpartition(-scores_arr, k - 1)[:k].astype(np.int64, copy=False)
            order = np.lexsort((np.take(resume_ids_arr, partial), -np.take(scores_arr, partial)))
            idxs = np.take(partial, order)
        recs: List[Tuple[str, float]] = []
        for i_idx in idxs[:top_k].tolist():
            i = int(i_idx)
            recs.append((str(resume_ids_arr[i]), float(scores_arr[i])))
        job_to_resumes[jid] = recs
    return resume_to_jobs, job_to_resumes

# =============== BM25 PARALLEL / CHECKPOINT HELPERS ===============
def _bm25_tokenize(t: str) -> List[str]:
    import re
    return re.findall(r"\b\w+\b", (t or "").lower())

def _bm25_prep_docs(docs: List[str], label: str = "docs"):
    import math, time
    tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    N = len(docs)

    # phase A: tokenize + document frequency
    start = time.time()
    for i, raw in enumerate(docs, 1):
        toks = _bm25_tokenize(raw)
        tokens.append(toks)
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
        if i % 1000 == 0 or i == N:
            _bm25_progress(i, N, start, f"prep:{label}")

    avgdl = sum(len(t) for t in tokens) / (N or 1)

    # phase B: per-doc term frequency
    tf_list: List[Dict[str, int]] = []
    start_tf = time.time()
    for i, toks in enumerate(tokens, 1):
        tf: Dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        tf_list.append(tf)
        if i % 1000 == 0 or i == N:
            _bm25_progress(i, N, start_tf, f"tf:{label}")

    idf = {w: math.log((N - dfw + 0.5) / (dfw + 0.5) + 1) for w, dfw in df.items()}
    return tokens, tf_list, idf, avgdl
# ---- Shared BM25 state (each worker gets its own copy via initializer) ----
_BM25 = {}  # type: dict
def _bm25_init_from_file(state_path: str):
    """Initializer for ProcessPool: each worker loads state from a pickle file exactly once."""
    global _BM25
    import pickle
    with open(state_path, "rb") as f:
        _BM25 = pickle.load(f)

def _bm25_init(state: dict):
    """Initializer for ProcessPool: executed once per worker."""
    global _BM25
    _BM25 = state

def _bm25_worker_job2res(j_idx: int, top_k: int):
    global _BM25
    import numpy as np
    k1 = 1.5; b = 0.75
    job_tokens      = _BM25["job_tokens"]
    resume_tokens   = _BM25["resume_tokens"]
    tf_res          = _BM25["tf_res"]
    idf_res         = _BM25["idf_res"]
    avgdl_res       = _BM25["avgdl_res"]
    job_ids         = _BM25["job_ids"]
    resume_ids      = _BM25["resume_ids"]

    q_tokens = job_tokens[j_idx]
    scores_row = []
    for doc_tokens, tf_dict in zip(resume_tokens, tf_res):
        score = 0.0
        dl = len(doc_tokens)
        for t in q_tokens:
            f = tf_dict.get(t)
            if not f: 
                continue
            idf = idf_res.get(t, 0.0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (avgdl_res or 1)))
        scores_row.append(score)

    scores_arr = np.asarray(scores_row, dtype=np.float32)
    res_ids_arr = np.asarray(resume_ids, dtype=object)
    k = min(top_k, len(resume_ids))
    if k == len(resume_ids):
        idxs = np.argsort(-scores_arr, kind="mergesort")
    else:
        partial = np.argpartition(-scores_arr, k - 1)[:k].astype(np.int64, copy=False)
        order = np.lexsort((np.take(res_ids_arr, partial), -np.take(scores_arr, partial)))
        idxs = np.take(partial, order)
    recs = [(str(resume_ids[int(i)]), float(scores_arr[int(i)])) for i in idxs[:top_k].tolist()]
    return (str(job_ids[j_idx]), recs)


def _bm25_worker_res2job(r_idx: int, top_k: int):
    global _BM25
    import numpy as np
    k1 = 1.5; b = 0.75
    resume_tokens   = _BM25["resume_tokens"]
    job_tokens      = _BM25["job_tokens"]
    tf_job          = _BM25["tf_job"]
    idf_job         = _BM25["idf_job"]
    avgdl_job       = _BM25["avgdl_job"]
    resume_ids      = _BM25["resume_ids"]
    job_ids         = _BM25["job_ids"]

    q_tokens = resume_tokens[r_idx]
    scores_row = []
    for doc_tokens, tf_dict in zip(job_tokens, tf_job):
        score = 0.0
        dl = len(doc_tokens)
        for t in q_tokens:
            f = tf_dict.get(t)
            if not f: 
                continue
            idf = idf_job.get(t, 0.0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / (avgdl_job or 1)))
        scores_row.append(score)

    scores_arr = np.asarray(scores_row, dtype=np.float32)
    job_ids_arr = np.asarray(job_ids, dtype=object)
    k = min(top_k, len(job_ids))
    if k == len(job_ids):
        idxs = np.argsort(-scores_arr, kind="mergesort")
    else:
        partial = np.argpartition(-scores_arr, k - 1)[:k].astype(np.int64, copy=False)
        order = np.lexsort((np.take(job_ids_arr, partial), -np.take(scores_arr, partial)))
        idxs = np.take(partial, order)
    recs = [(str(job_ids[int(j)]), float(scores_arr[int(j)])) for j in idxs[:top_k].tolist()]
    return (str(resume_ids[r_idx]), recs)

def _bm25_ckpt_load_ids(path: str) -> set:
    done = set()
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            jid_col = "job_id" if "job_id" in (rdr.fieldnames or []) else (rdr.fieldnames or [None])[0]
            for row in rdr:
                if jid_col and row.get(jid_col):
                    done.add(str(row[jid_col]))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return done

def _bm25_ckpt_append(path: str, rows: List[Dict[str, Any]], header: List[str]) -> None:
    first = not Path(path).exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def _bm25_progress(done: int, total: int, start_ts: float, label: str) -> None:
    pct = (done / total * 100) if total else 100.0
    el = time.time() - start_ts
    rate = done / el if el > 0 else 0.0
    rem = (total - done) / rate if rate > 0 else 0.0
    bar = "#" * int(pct / 4) + "-" * (25 - int(pct / 4))
    print(f"\r[BM25:{label}] |{bar}| {pct:5.1f}%  {done}/{total}  ({rate:,.1f}/s, ETA {rem:,.0f}s)", end="", flush=True)
    if done >= total:
        print()

def run_bm25_parallel(
    resume_texts: List[str], job_texts: List[str],
    resume_ids: List[str], job_ids: List[str],
    top_k: int,
    bm25_workers: int = 4, bm25_chunk: int = 500,
    bm25_ckpt: Optional[str] = None, bm25_resume: bool = False
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # ---- prep corpora once (done in parent; each worker gets a copy via initializer) ----
    job_tokens,   tf_job, idf_job, avgdl_job   = _bm25_prep_docs(job_texts)
    resume_tokens, tf_res, idf_res, avgdl_res  = _bm25_prep_docs(resume_texts)
    # Build shared state and persist to a temp file (Windows-friendly)
    state = {
        "job_tokens": job_tokens, "tf_job": tf_job, "idf_job": idf_job, "avgdl_job": avgdl_job,
        "resume_tokens": resume_tokens, "tf_res": tf_res, "idf_res": idf_res, "avgdl_res": avgdl_res,
        "job_ids": job_ids, "resume_ids": resume_ids
    }
    import os, tempfile, pickle, uuid
    state_path = os.path.join(tempfile.gettempdir(), f"bm25_state_{uuid.uuid4().hex}.pkl")
    with open(state_path, "wb") as _f:
        pickle.dump(state, _f, protocol=pickle.HIGHEST_PROTOCOL)
    



    # ================== Phase A: job a†’ resumes (checkpoint-able) ==================
    jtr: Dict[str, List[Tuple[str, float]]] = {}

    done_ids: set = set()
    if bm25_ckpt and bm25_resume:
        done_ids = _bm25_ckpt_load_ids(bm25_ckpt)

    pending_idxs = [j for j, jid in enumerate(job_ids) if str(jid) not in done_ids]
    total = len(pending_idxs)
    start_ts = time.time()
    header = ["job_id", "resume_id", "rank", "score"]

    if bm25_workers <= 1:
        processed = 0
        batch_rows: List[Dict[str, Any]] = []
        for j_idx in pending_idxs:
            jid, recs = _bm25_worker_job2res(j_idx, top_k)
            jtr[jid] = recs
            if bm25_ckpt:
                rank = 0
                for rid, sc in recs:
                    rank += 1
                    batch_rows.append({"job_id": jid, "resume_id": rid, "rank": rank, "score": sc})
                if len(batch_rows) >= 5000:
                    _bm25_ckpt_append(bm25_ckpt, batch_rows, header); batch_rows.clear()
            processed += 1
            _bm25_progress(processed, total, start_ts, "job->res")
        if bm25_ckpt and batch_rows:
            _bm25_ckpt_append(bm25_ckpt, batch_rows, header)

    else:
        
        with ProcessPoolExecutor(
            max_workers=bm25_workers,
            initializer=_bm25_init_from_file,
            initargs=(state_path,)
        ) as ex:
            futs = [ex.submit(_bm25_worker_job2res, j_idx, top_k) for j_idx in pending_idxs]
            batch_rows: List[Dict[str, Any]] = []
            for i, fut in enumerate(as_completed(futs), 1):
                jid, recs = fut.result()
                jtr[jid] = recs
                if bm25_ckpt:
                    rank = 0
                    for rid, sc in recs:
                        rank += 1
                        batch_rows.append({"job_id": jid, "resume_id": rid, "rank": rank, "score": sc})
                    if len(batch_rows) >= 5000:
                        _bm25_ckpt_append(bm25_ckpt, batch_rows, header); batch_rows.clear()
                    _bm25_progress(i, total, start_ts, "job->res")
            if bm25_ckpt and batch_rows:
                _bm25_ckpt_append(bm25_ckpt, batch_rows, header)

    # ================== Phase B: resume a†’ jobs (parallel; no ckpt) ==================
    rtj: Dict[str, List[Tuple[str, float]]] = {}

    # indices we will process
    r_pending = list(range(len(resume_ids)))
    r_total = len(r_pending)
    r_start = time.time()

    if bm25_workers <= 1:
        # single-process path
        for i, r_idx in enumerate(r_pending, 1):
            rid, recs = _bm25_worker_res2job(r_idx, top_k)
            rtj[rid] = recs
            _bm25_progress(i, r_total, r_start, "res->job")
    else:
        # multi-process path (Windows-safe); if spawn glitches, fall back gracefully
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            with ProcessPoolExecutor(
                max_workers=bm25_workers,
                initializer=_bm25_init_from_file,  # each worker loads the state from disk once
                initargs=(state_path,),
            ) as ex:
                futs = [ex.submit(_bm25_worker_res2job, r_idx, top_k) for r_idx in r_pending]
                for i, fut in enumerate(as_completed(futs), 1):
                    rid, recs = fut.result()
                    rtj[rid] = recs
                    _bm25_progress(i, r_total, r_start, "res->job")
        except EOFError:
            # Rare on Windows: retry sequentially rather than dying
            for i, r_idx in enumerate(r_pending, 1):
                rid, recs = _bm25_worker_res2job(r_idx, top_k)
                rtj[rid] = recs
                _bm25_progress(i, r_total, r_start, "res->job")
        finally:
            # cleanup the temp state file
            try:
                os.remove(state_path)
            except Exception:
                pass

    return rtj, jtr

# ============== END BM25 PARALLEL / CHECKPOINT HELPERS =============


def save_recs(recs: Dict[str, List[Tuple[str, float, str]]], out_path: Path) -> None:
    start = time.time()
    try:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["resume_id", "job_id", "similarity", "missing_skills"])
            for rid, lst in recs.items():
                for jid, sim, miss in lst:
                    writer.writerow([rid, jid, sim, miss])
        logger.info("Saved recommendations in %.2f sec to %s", time.time() - start, out_path)
    except Exception as e:
        logger.error("Error saving recommendations: %s", e)
        sys.exit(1)

import re, html
from csv import DictReader

_TAG_RE = re.compile(r"<[^>]+>")

# Optional FlashText import for fast keyword extraction.  If available, a
# KeywordProcessor will be built on first use for the current skill set.  A
# fallback compiled regular expression is used otherwise.  Caches are
# maintained per process; when the skills set changes the caches are
# reconstructed.
try:
    from flashtext import KeywordProcessor  # type: ignore
    _flashtext_available = True
except Exception:
    KeywordProcessor = None  # type: ignore
    _flashtext_available = False

# Pera€‘process caches for skill extraction
_flash_cache: Dict[frozenset, KeywordProcessor] = {}
_regex_cache: Dict[str, Any] = {"skills_set": None, "regex": None}

def _sanitize_text(x: str) -> str:
    if x is None:
        return ""
    s = html.unescape(str(x))
    s = _TAG_RE.sub(" ", s)          # strip HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _load_table_strict(path: str) -> list[dict[str, str]]:
    """Read CSV with DictReader (no dtype coercion). Returns list of dict rows as pure strings."""
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(DictReader(f)):
            # Ensure all values are strings; replace Nones
            clean = {k: ("" if v is None else str(v)) for k, v in row.items()}
            rows.append(clean)
    return rows

# New: load CSV rows with per-row error handling.  Returns a tuple of (rows, escaped_rows)
# escaped_rows is a list of dicts with keys: which, row_index, id_if_any, reason.
def _load_table_with_errors(path: str, id_col: str, which: str) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Load a CSV file and return rows as list of dicts along with a list of escaped rows.
    BOM-safe (utf-8-sig) and header keys normalized (strip BOM + whitespace).
    """
    rows: List[Dict[str, str]] = []
    escaped_rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:  # <-- utf-8-sig
            try:
                reader = csv.DictReader(f)
                # Normalize fieldnames to kill BOM/whitespace (e.g., "\ufeffID" -> "ID")
                if reader.fieldnames:
                    reader.fieldnames = [(h or "").lstrip("\ufeff").strip() for h in reader.fieldnames]
            except Exception as ex:
                escaped_rows.append({"which": which, "row_index": -1, "id_if_any": "", "reason": str(ex)})
                return rows, escaped_rows

            for idx, row in enumerate(reader):
                try:
                    clean: Dict[str, str] = {}
                    for k, v in row.items():
                        nk = (k or "").lstrip("\ufeff").strip()
                        clean[nk] = "" if v is None else str(v)
                    rows.append(clean)
                except Exception as e:
                    rid = ""
                    try:
                        rid = str(row.get(id_col, ""))
                    except Exception:
                        rid = ""
                    escaped_rows.append({"which": which, "row_index": idx, "id_if_any": rid, "reason": str(e)})
    except Exception as e:
        # file-level error
        escaped_rows.append({"which": which, "row_index": -1, "id_if_any": "", "reason": str(e)})
    return rows, escaped_rows

def main():
    # Initialize timing and parse arguments
    args = parse_args()
    # Normalize possible BOM/whitespace in CLI column names
    global GLOBAL_ARGS
    GLOBAL_ARGS = args
    def _norm(s: str) -> str:
        return (s or "").lstrip("\ufeff").strip()

    args.resume_id_col   = _norm(args.resume_id_col)
    args.resume_text_col = _norm(args.resume_text_col)
    args.job_id_col      = _norm(args.job_id_col)
    args.job_text_col    = _norm(args.job_text_col)
    total_start = time.time()
    # Set deterministic seeds for reproducibility across this run
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    # Determine output directory: prefer --output_dir, then the parent of --output, else cwd
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Derive from deprecated --output path if provided
        out_dir = Path(args.output).parent if args.output else Path('.')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Default BM25 checkpoint file if not supplied
    if getattr(args, "bm25_ckpt", None) is None and args.output_dir:
        args.bm25_ckpt = str(out_dir / f"job_to_resumes_top{args.top_k}_bm25.csv.part")
    # Determine checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Setup paths for caches and checkpoints
    ckpt = ckpt_dir / "checkpoint_stage2.pkl"
    resume_pkl = ckpt_dir / "resume_sk.pkl"
    job_pkl    = ckpt_dir / "job_sk.pkl"
    emb_pkl    = ckpt_dir / "embeddings.pkl"
    # Parse baselines; an empty string results in an empty list
    baseline_methods = [m.strip().lower() for m in (args.baseline.split(',') if args.baseline else []) if m.strip()]
    # Load resumes and jobs with per-row error handling
    escaped_rows: List[Dict[str, Any]] = []
    try:
        resumes_rows, escaped_resumes = _load_table_with_errors(args.resumes, args.resume_id_col, "resume")
        jobs_rows, escaped_jobs = _load_table_with_errors(args.jobs, args.job_id_col, "job")
        escaped_rows.extend(escaped_resumes)
        escaped_rows.extend(escaped_jobs)
        # Sanitize IDs and text
        for r in resumes_rows:
            try:
                r[args.resume_id_col] = str(r.get(args.resume_id_col, "")).strip()
            except Exception:
                r[args.resume_id_col] = ""
            try:
                r[args.resume_text_col] = _sanitize_text(r.get(args.resume_text_col, ""))
            except Exception:
                r[args.resume_text_col] = ""
        for j in jobs_rows:
            try:
                j[args.job_id_col] = str(j.get(args.job_id_col, "")).strip()
            except Exception:
                j[args.job_id_col] = ""
            try:
                j[args.job_text_col] = _sanitize_text(j.get(args.job_text_col, ""))
            except Exception:
                j[args.job_text_col] = ""
        # Minimal DataFrame shim to avoid pandas in resumes/jobs
        class _RowsShim:
            def __init__(self, rows, id_col):
                self._rows = rows
                self._id_col = id_col

            def __len__(self):
                return len(self._rows)

            def to_dict(self, orientation):
                # build_graph calls .to_dict("records")
                if orientation != "records":
                    raise ValueError("Only 'records' supported in _RowsShim")
                return list(self._rows)

            def __getitem__(self, key):
                if key != self._id_col:
                    raise KeyError(key)

                class _Col:
                    def __init__(self, vals):
                        self._vals = vals

                    class _Typed:
                        def __init__(self, vals):
                            # ensure strings
                            self._vals = [str(v) for v in vals]
                        def __iter__(self):
                            return iter(self._vals)
                        def tolist(self):
                            return list(self._vals)

                    def astype(self, _):
                        # return an iterable object with .tolist(), for both build_graph and main()
                        return _Col._Typed(self._vals)

                    def tolist(self):
                        # in case someone calls tolist() without astype
                        return list(self._vals)

                return _Col([r.get(self._id_col, "") for r in self._rows])


        resumes_df = _RowsShim(resumes_rows, args.resume_id_col)
        jobs_df    = _RowsShim(jobs_rows, args.job_id_col)
        logger.info("Loaded data in %.2f sec", time.time() - total_start)
        # Build simple lists for baseline computations
        resume_ids_list = [r.get(args.resume_id_col, "") for r in resumes_rows]
        resume_texts_list = [r.get(args.resume_text_col, "") for r in resumes_rows]
        job_ids_list    = [j.get(args.job_id_col, "") for j in jobs_rows]
        job_texts_list  = [j.get(args.job_text_col, "") for j in jobs_rows]

        # --- Baseline path: if any baselines specified, skip node2vec ---
        if baseline_methods:
            logger.info("Running baseline(s): %s", ", ".join(baseline_methods))

            # For each baseline method compute rankings
            for method in baseline_methods:
                method_start = time.time()
                try:
                    if method == "tfidf":
                        # Pass optional GPU flags to the TFa€‘IDF baseline.  When
                        # --use_gpu_tfidf is enabled, the compute_tfidf_baseline
                        # function will attempt to utilise a CUDA GPU for
                        # similarity calculations.  It gracefully falls back to
                        # the CPU path if PyTorch or CUDA is unavailable.
                        rtj, jtr = compute_tfidf_baseline(
                            resume_texts_list,
                            job_texts_list,
                            resume_ids_list,
                            job_ids_list,
                            args.top_k,
                            use_gpu=getattr(args, "use_gpu_tfidf", False),
                            gpu_chunk=getattr(args, "tfidf_gpu_chunk", 2000),
                            gpu_dtype=getattr(args, "tfidf_gpu_dtype", "float32"),
                        )
                    elif method == "bm25":
                        print(
                            f"[BM25] PARALLEL runner  workers={getattr(args,'bm25_workers',4)}  "
                            f"chunk={getattr(args,'bm25_chunk',500)}  "
                            f"ckpt={getattr(args,'bm25_ckpt',None)}  "
                            f"resume={getattr(args,'bm25_resume',False)}",
                            flush=True
                        )
                        rtj, jtr = run_bm25_parallel(
                            resume_texts_list, job_texts_list,
                            resume_ids_list, job_ids_list,
                            args.top_k,
                            bm25_workers=getattr(args, "bm25_workers", 4),
                            bm25_chunk=getattr(args, "bm25_chunk", 500),
                            bm25_ckpt=getattr(args, "bm25_ckpt", None),
                            bm25_resume=getattr(args, "bm25_resume", False),
                        )

                    else:
                        logger.error("Unsupported baseline: %s", method)
                        continue
                except Exception as e:
                    logger.error("Error computing %s baseline: %s", method, e)
                    continue
                # Write resumea†’jobs if requested
                if args.direction in ("resume_to_jobs", "both"):
                    records: List[Tuple[str, str, float, int, str]] = []
                    for rid, recs in tqdm(rtj.items(), desc=f"Write CSV ({method}) resumea†’jobs",
                      unit="resumes", dynamic_ncols=True, smoothing=0.1,
                      bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):

                        for rank_idx, (jid, sc) in enumerate(recs, start=1):
                            records.append((rid, jid, sc, rank_idx, method))
                    fname = f"resume_to_jobs_top{args.top_k}_{method}.csv"
                    write_topk_csv(out_dir / fname, records, ["resume_id", "job_id", "score", "rank", "source"])
                    logger.info("Wrote %s baseline recommendations to %s", method, out_dir / fname)
                # Write joba†’resumes if requested
                if args.direction in ("job_to_resumes", "both"):
                    records: List[Tuple[str, str, float, int, str]] = []
                    for jid, recs in tqdm(jtr.items(), desc=f"Write CSV ({method}) joba†’resumes",
                      unit="jobs", dynamic_ncols=True, smoothing=0.1,
                      bar_format="{l_bar}{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                        for rank_idx, (rid, sc) in enumerate(recs, start=1):
                            records.append((jid, rid, sc, rank_idx, method))
                    fname = f"job_to_resumes_top{args.top_k}_{method}.csv"
                    write_topk_csv(out_dir / fname, records, ["job_id", "resume_id", "score", "rank", "source"])
                    logger.info("Wrote %s baseline reverse recommendations to %s", method, out_dir / fname)
                    logger.info("%s baseline completed in %.2f sec", method, time.time() - method_start)
                    
            # Write escaped_rows.csv for baseline runs
            escaped_rows_path = ckpt_dir / "escaped_rows.csv"
            try:
                with open(escaped_rows_path, "w", encoding="utf-8", newline="") as ef:
                    w = csv.writer(ef)
                    w.writerow(["which", "row_index", "id_if_any", "reason"])
                    for item in escaped_rows:
                        w.writerow([
                            item.get("which", ""),
                            item.get("row_index", ""),
                            item.get("id_if_any", ""),
                            item.get("reason", ""),
                        ])
            except Exception as e:
                logger.error("Error writing escaped_rows.csv: %s", e)
            # Baseline manifest
            manifest_path = ckpt_dir / "run_manifest.json"
            try:
                manifest: Dict[str, Any] = {
                    "args": vars(args),
                    "baseline_methods": baseline_methods,
                    "top_k": args.top_k,
                    "start_time": total_start,
                    "end_time": time.time(),
                    "num_resumes": len(resume_ids_list),
                    "num_jobs": len(job_ids_list),
                    "escaped_rows_csv": str(escaped_rows_path),
                    "escaped_resumes": len([e for e in escaped_rows if e.get("which") == "resume"]),
                    "escaped_jobs": len([e for e in escaped_rows if e.get("which") == "job"]),
                }
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)
            except Exception as e:
                logger.error("Error writing run_manifest.json: %s", e)
                            # === EVAL (baseline end) ===
            try:
                node_p  = out_dir / "job_to_resumes_top10.csv"
                tfidf_p = out_dir / "job_to_resumes_top10_tfidf.csv"
                bm25_p  = out_dir / "job_to_resumes_top10_bm25.csv"
                node_map  = _load_job2res(node_p)
                tfidf_map = _load_job2res(tfidf_p)
                bm25_map  = _load_job2res(bm25_p)

                # Quick logs (coverage + overlaps)
                if node_map:
                    full, total, frac = _coverage_full_k(node_map, args.top_k)
                    logger.info("EVAL: node2vec coverage: %d/%d jobs (%.1f%%) with full top-%d",
                                full, total, 100*frac, args.top_k)
                if node_map and tfidf_map:
                    ov = _overlap_at_10(node_map, tfidf_map)
                    if ov is not None:
                        logger.info("EVAL: overlap@10 node2veca†”tfidf: %.3f", ov)
                if node_map and bm25_map:
                    ov = _overlap_at_10(node_map, bm25_map)
                    if ov is not None:
                        logger.info("EVAL: overlap@10 node2veca†”bm25: %.3f", ov)

                # Write overall_metrics.csv for the dashboard (proxy P@10 via consensus overlap)
                rows = []
                def _consensus(name, maps):
                    others = [k for k in maps if k != name and maps[k]]
                    vals = []
                    for o in others:
                        v = _overlap_at_10(maps[name], maps[o])
                        if v is not None:
                            vals.append(v)
                    return (sum(vals)/len(vals)) if vals else None
                maps = {"node2vec": node_map, "tfidf": tfidf_map, "bm25": bm25_map}
                for name in ("node2vec", "tfidf", "bm25"):
                    if maps.get(name):
                        val = _consensus(name, maps)
                        if val is not None:
                            rows.append({"source": name, "p_at_10": f"{val:.4f}"})
                if rows:
                    om_path = out_dir / "overall_metrics.csv"
                    with open(om_path, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["source", "p_at_10"])
                        w.writeheader()
                        for r in rows:
                            w.writerow(r)
                    logger.info("EVAL: wrote %s with %d rows", om_path, len(rows))
            except Exception as e:
                logger.warning("EVAL: metrics emission skipped due to error: %s", e)
            # === /EVAL ===

                # === EVAL (baseline end) ===
            try:
                node_p  = out_dir / "job_to_resumes_top10.csv"
                tfidf_p = out_dir / "job_to_resumes_top10_tfidf.csv"
                bm25_p  = out_dir / "job_to_resumes_top10_bm25.csv"
                node_map  = _load_job2res(node_p)
                tfidf_map = _load_job2res(tfidf_p)
                bm25_map  = _load_job2res(bm25_p)

                # Quick logs (coverage + overlaps)
                if node_map:
                    full, total, frac = _coverage_full_k(node_map, args.top_k)
                    logger.info("EVAL: node2vec coverage: %d/%d jobs (%.1f%%) with full top-%d",
                                full, total, 100*frac, args.top_k)
                if node_map and tfidf_map:
                    ov = _overlap_at_10(node_map, tfidf_map)
                    if ov is not None:
                        logger.info("EVAL: overlap@10 node2veca†”tfidf: %.3f", ov)
                if node_map and bm25_map:
                    ov = _overlap_at_10(node_map, bm25_map)
                    if ov is not None:
                        logger.info("EVAL: overlap@10 node2veca†”bm25: %.3f", ov)

                # Write overall_metrics.csv for the dashboard (proxy P@10 via consensus overlap)
                rows = []
                def _consensus(name, maps):
                    others = [k for k in maps if k != name and maps[k]]
                    vals = []
                    for o in others:
                        v = _overlap_at_10(maps[name], maps[o])
                        if v is not None:
                            vals.append(v)
                    return (sum(vals)/len(vals)) if vals else None
                maps = {"node2vec": node_map, "tfidf": tfidf_map, "bm25": bm25_map}
                for name in ("node2vec", "tfidf", "bm25"):
                    if maps.get(name):
                        val = _consensus(name, maps)
                        if val is not None:
                            rows.append({"source": name, "p_at_10": f"{val:.4f}"})
                if rows:
                    om_path = out_dir / "overall_metrics.csv"
                    with open(om_path, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["source", "p_at_10"])
                        w.writeheader()
                        for r in rows:
                            w.writerow(r)
                    logger.info("EVAL: wrote %s with %d rows", om_path, len(rows))
            except Exception as e:
                logger.warning("EVAL: metrics emission skipped due to error: %s", e)
            # === /EVAL ===

            # Baseline complete
            logger.info("Baseline run completed in %.2f sec", time.time() - total_start)
            return
    except Exception:
        logger.exception("Error loading resumes/jobs (DictReader path) a€“ this is not a pandas reader")
        sys.exit(1)

    # Node2Vec path
    # Build skills dictionary
    skills = build_skill_dictionary(Path(args.skills_en), Path(args.skills_txt), Path(args.tech_txt), Path(args.related))
    logger.info("Skills loaded: %d entries", len(skills))

    # --- Construct graph (cache-first) ---
    resume_sk: Optional[Dict[str, List[str]]] = None
    job_sk: Optional[Dict[str, List[str]]] = None
    G: Optional[nx.Graph] = None

    # Attempt to load skill caches and rebuild graph from them
    if resume_pkl.exists() and job_pkl.exists():
        try:
            with open(resume_pkl, "rb") as f:
                resume_sk = pickle.load(f)
            with open(job_pkl, "rb") as f:
                job_sk = pickle.load(f)
            logger.info("Loaded skill caches from %s and %s", resume_pkl, job_pkl)
            # Rebuild graph quickly from cached skills
            G = nx.Graph()
            # Pre-add resume and job nodes so zero-skill rows still exist
            for rid in resumes_df[args.resume_id_col].astype(str):
                G.add_node(f"resume_{rid}", node_type="resume")
            for jid in jobs_df[args.job_id_col].astype(str):
                G.add_node(f"job_{jid}", node_type="job")
            # Edges from cached skills
            for rid, sks in resume_sk.items():
                for sk in sks:
                    G.add_edge(f"resume_{rid}", f"skill_{sk}")
            for jid, sks in job_sk.items():
                for sk in sks:
                    G.add_edge(f"job_{jid}", f"skill_{sk}")
            # Label skill nodes
            for _, sks in resume_sk.items():
                for sk in sks:
                    G.nodes[f"skill_{sk}"]["node_type"] = "skill"
            for _, sks in job_sk.items():
                for sk in sks:
                    G.nodes[f"skill_{sk}"]["node_type"] = "skill"
            logger.info(
                "Graph reconstructed from caches: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges()
            )
        except Exception as e:
            logger.warning("Cache load failed (%s); rebuilding from raw CSVs.", e)
            resume_sk = None
            job_sk = None
            G = None
    # If no valid caches, build from scratch and then write caches
    if G is None or resume_sk is None or job_sk is None:
        logger.info("Building graph from raw CSVsa€¦")
        G, resume_sk, job_sk = build_graph(
            resumes_df, jobs_df, skills,
            args.resume_id_col, args.resume_text_col,
            args.job_id_col, args.job_text_col,
            workers=args.workers, chunks_per_worker=args.chunks_per_worker,
        )
        # Save caches for future runs
        try:
            with open(resume_pkl, "wb") as f:
                pickle.dump(resume_sk, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(job_pkl, "wb") as f:
                pickle.dump(job_sk, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved caches: %s, %s", resume_pkl, job_pkl)
        except Exception as e:
            logger.warning("Could not write caches: %s", e)
    # Compute or load embeddings
    graph_hash = compute_graph_hash(G)
    embeddings: Dict[str, np.ndarray] = {}
    gpu_used: bool = False
    gpu_unavailable: bool = False
    # Try to reuse embeddings if graph hash matches
    if emb_pkl.exists():
        try:
            with open(emb_pkl, "rb") as f:
                obj = pickle.load(f)
            if obj.get("graph_hash") == graph_hash and "embeddings" in obj:
                embeddings = obj["embeddings"]
                logger.info("Reused embeddings from cache; graph hash matched.")
            else:
                logger.info("Embedding cache graph hash mismatch; will recompute embeddings.")
        except Exception as e:
            logger.warning("Failed to load embeddings cache (%s); recomputing.", e)
    if not embeddings:
        embeddings, gpu_used, gpu_unavailable = compute_embeddings(
            G,
            workers=args.workers,
            use_gpu=args.use_gpu_node2vec,
        )
        # Save embeddings cache with graph hash
        try:
            with open(emb_pkl, "wb") as f:
                pickle.dump({"graph_hash": graph_hash, "embeddings": embeddings}, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved embeddings to %s", emb_pkl)
        except Exception as e:
            logger.warning("Could not write embeddings cache: %s", e)
    # Compute recommendations via vectorised topa€‘K
    resume_to_jobs, job_to_resumes = vectorized_top_k_both(
        embeddings,
        resume_ids_list,
        job_ids_list,
        args.top_k,
    )
    # Write node2vec recommendations according to direction
    if args.direction in ("resume_to_jobs", "both"):
        records: List[Tuple[str, str, float, int, str]] = []
        for rid, recs in resume_to_jobs.items():
            for rank_idx, (jid, sc) in enumerate(recs, start=1):
                records.append((rid, jid, sc, rank_idx, "node2vec"))
        fname = f"resume_to_jobs_top{args.top_k}.csv"
        write_topk_csv(out_dir / fname, records, ["resume_id", "job_id", "score", "rank", "source"])
        logger.info("Wrote node2vec resumea†’jobs to %s", out_dir / fname)
    if args.direction in ("job_to_resumes", "both"):
        records: List[Tuple[str, str, float, int, str]] = []
        for jid, recs in job_to_resumes.items():
            for rank_idx, (rid, sc) in enumerate(recs, start=1):
                records.append((jid, rid, sc, rank_idx, "node2vec"))
        fname = f"job_to_resumes_top{args.top_k}.csv"
        write_topk_csv(out_dir / fname, records, ["job_id", "resume_id", "score", "rank", "source"])
        logger.info("Wrote node2vec joba†’resumes to %s", out_dir / fname)
            # === EVAL (node2vec end) ===
    try:
        node_p  = out_dir / f"job_to_resumes_top{args.top_k}.csv"
        tfidf_p = out_dir / f"job_to_resumes_top{args.top_k}_tfidf.csv"
        bm25_p  = out_dir / f"job_to_resumes_top{args.top_k}_bm25.csv"

        node_map  = _load_job2res(node_p)
        tfidf_map = _load_job2res(tfidf_p)
        bm25_map  = _load_job2res(bm25_p)

        # quick logs
        if node_map:
            full, total, frac = _coverage_full_k(node_map, args.top_k)
            logger.info("EVAL: node2vec coverage: %d/%d jobs (%.1f%%) with full top-%d",
                        full, total, 100*frac, args.top_k)
        if node_map and tfidf_map:
            v = _overlap_at_10(node_map, tfidf_map)
            if v is not None:
                logger.info("EVAL: overlap@10 node2veca†”tfidf: %.3f", v)
        if node_map and bm25_map:
            v = _overlap_at_10(node_map, bm25_map)
            if v is not None:
                logger.info("EVAL: overlap@10 node2veca†”bm25: %.3f", v)

        # write overall_metrics.csv (proxy P@10 via consensus overlap)
        rows = []
        def _consensus(name, maps):
            others = [k for k in maps if k != name and maps[k]]
            vals = []
            for o in others:
                vv = _overlap_at_10(maps[name], maps[o])
                if vv is not None:
                    vals.append(vv)
            return (sum(vals)/len(vals)) if vals else None

        maps = {"node2vec": node_map, "tfidf": tfidf_map, "bm25": bm25_map}
        for name in ("node2vec", "tfidf", "bm25"):
            if maps.get(name):
                val = _consensus(name, maps)
                if val is not None:
                    rows.append({"source": name, "p_at_10": f"{val:.4f}"})

        if rows:
            om_path = out_dir / "overall_metrics.csv"
            with open(om_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["source","p_at_10"])
                w.writeheader()
                for r in rows: w.writerow(r)
            logger.info("EVAL: wrote %s with %d rows", om_path, len(rows))
    except Exception as e:
        logger.warning("EVAL: metrics emission skipped due to error: %s", e)
# === /EVAL ===

    # Always write escaped_rows.csv
    escaped_rows_path = ckpt_dir / "escaped_rows.csv"
    try:
        with open(escaped_rows_path, "w", encoding="utf-8", newline="") as ef:
            w = csv.writer(ef)
            w.writerow(["which", "row_index", "id_if_any", "reason"])
            for item in escaped_rows:
                w.writerow([
                    item.get("which", ""),
                    item.get("row_index", ""),
                    item.get("id_if_any", ""),
                    item.get("reason", ""),
                ])
    except Exception as e:
        logger.error("Error writing escaped_rows.csv: %s", e)
    # Determine nodes without embeddings
    no_embed_rows: List[Dict[str, Any]] = []
    for rid in resume_ids_list:
        if f"resume_{rid}" not in embeddings:
            no_embed_rows.append({"which": "resume", "id": rid, "reason": "no_embedding"})
    for jid in job_ids_list:
        if f"job_{jid}" not in embeddings:
            no_embed_rows.append({"which": "job", "id": jid, "reason": "no_embedding"})
    no_embed_path = ckpt_dir / "no_embedding.csv"
    try:
        with open(no_embed_path, "w", encoding="utf-8", newline="") as nf:
            w = csv.writer(nf)
            w.writerow(["which", "id", "reason"])
            for item in no_embed_rows:
                w.writerow([
                    item.get("which", ""),
                    item.get("id", ""),
                    item.get("reason", ""),
                ])
    except Exception as e:
        logger.error("Error writing no_embedding.csv: %s", e)
    # Count zero-skill resumes/jobs
    zero_skill_resumes = sum(1 for sks in resume_sk.values() if not sks)
    zero_skill_jobs = sum(1 for sks in job_sk.values() if not sks)
    # Write manifest for node2vec run
    manifest_path = ckpt_dir / "run_manifest.json"
    try:
        manifest: Dict[str, Any] = {
            "args": vars(args),
            "baseline_methods": baseline_methods,
            "workers": args.workers,
            "chunks_per_worker": args.chunks_per_worker,
            "top_k": args.top_k,
            "gpu_used": gpu_used,
            "gpu_unavailable": gpu_unavailable,
            "graph_hash": graph_hash,
            "start_time": total_start,
            "end_time": time.time(),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "zero_skill_resumes": zero_skill_resumes,
            "zero_skill_jobs": zero_skill_jobs,
            "escaped_resumes": len([e for e in escaped_rows if e.get("which") == "resume"]),
            "escaped_jobs": len([e for e in escaped_rows if e.get("which") == "job"]),
            "escaped_rows_csv": str(escaped_rows_path),
            "no_embedding_count": len(no_embed_rows),
            "no_embedding_csv": str(no_embed_path),
            "library_versions": {
                "numpy": np.__version__,
                "networkx": nx.__version__,
            },
        }
        # Include scikit-learn version if available
        try:
            import sklearn  # type: ignore
            manifest["library_versions"]["sklearn"] = sklearn.__version__  # type: ignore[attr-defined]
        except Exception:
            pass
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
    except Exception as e:
        logger.error("Error writing run_manifest.json: %s", e)
    # Write checkpoint file to indicate completion
    try:
        with open(ckpt, "wb") as f:
            pickle.dump({"stage2_complete": True}, f)
    except Exception as e:
        logger.error("Error writing checkpoint: %s", e)
    logger.info("Stage 2 total time: %.2f sec", time.time() - total_start)

if __name__ == "__main__":
    main()

