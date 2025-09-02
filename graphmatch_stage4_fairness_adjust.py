#!/usr/bin/env python3
"""
GraphMatch-AI - Stage 4: Fairness-Aware Re-Ranking 
=============================================================

This stage adjusts the Stage 2 recommendations to address disparities
identified by the Stage 3 fairness audit. It supports weighting by
**multiple protected attributes**. For each attribute, a weight is
computed for each group based on the selection rates in the fairness
metrics CSV. The weights are combined for each resume using the
geometric mean across attributes. Each recommendationa€™s score is
multiplied by the resumes combined weight and the recommendations are
re-ranked per job.

Key features:

* Avoids pandas entirely - uses built-in CSV handling (via ``csv``).
* Supports multiple protected attributes via ``--protected_attrs``.
* Saves both the adjusted recommendations and the per-group weights for
  transparency.

Example usage::

    python graphmatch_stage4_fairness_adjust.py \
      --recommendations D:/GraphMatchOutput/recommendations_stage2.csv \
      --resumes D:/graphmatch-ai/Test/Resume.csv \
      --resume_id_col ID \
      --metrics D:/GraphMatchOutput/fairness_metrics.csv \
      --protected_attrs gender ethnicity \
      --top_k 5 \
      --output_adjusted D:/GraphMatchOutput/recommendations_stage4_fair.csv \
      --output_weights D:/GraphMatchOutput/fairness_weights.csv

The adjusted recommendations will contain the same columns as the
Stage 2 recommendations but with updated similarity scores. The
weights CSV lists each attribute, group value and assigned weight.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Iterable, Any

# CSV helper functions (no pandas)
import csv
from typing import Any, Dict, List, Optional

def read_csv_rows(path: str, sep: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Robust CSV reader:
      - Opens with utf-8-sig to strip any BOM.
      - If sep is provided, use it; otherwise honor Excel's 'sep=' header
        and fall back to csv.Sniffer() over common delimiters.
      - Returns list of dict rows with strings (None -> "").
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # Peek to detect Excel 'sep=' and help Sniffer
        sample = f.read(65536)
        f.seek(0)
        first_line = f.readline()
        sep_hint = None
        if first_line.lower().startswith("sep="):
            raw = first_line[4:].strip()
            sep_hint = "\t" if raw in ("\\t", "\t") else (raw[:1] if raw else None)
            # leave file positioned after the hint
        else:
            # no hint; rewind to header
            f.seek(0)

        # Decide delimiter
        if sep is not None:
            delim = "\t" if sep == r"\t" else sep
        elif sep_hint:
            delim = sep_hint
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                delim = dialect.delimiter
            except Exception:
                delim = ","

        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            rows.append({(k if k is not None else ""): ("" if v is None else str(v)) for k, v in row.items()})
    return rows

def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str], sep: str = ",") -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StageA 4 a€“ Fairnessa€‘Aware Rea€‘Ranking of Recommendations (no pandas)"
    )
    parser.add_argument(
        "--recommendations",
        required=True,
        help="Path to the original recommendations CSV",
    )
    parser.add_argument(
        "--resumes",
        required=True,
        help="Path to the resumes CSV containing protected attribute values",
    )
    parser.add_argument(
        "--resume_id_col",
        default="ID",
        help="Name of the resume ID column (defaults to 'ID')",
    )
    parser.add_argument(
        "--job_id_col",
        default="job_id",
        help="Name of the job ID column in the recommendations CSV (defaults to 'job_id')",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to the fairness metrics CSV produced by StageA 3",
    )
    parser.add_argument(
        "--protected_attrs",
        nargs='+',
        required=True,
        help="One or more protected attribute columns to adjust for (e.g. gender, ethnicity)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommendations per resume (must match StageA 2)",
    )
    parser.add_argument(
        "--output_adjusted",
        required=True,
        help="Path to write the fairnessa€‘adjusted recommendations CSV",
    )
    parser.add_argument(
        "--output_weights",
        required=True,
        help="Path to write the group weights CSV for transparency",
    )
    return parser.parse_args()


def compute_weights(metrics: List[Dict[str, str]], attrs: Iterable[str], alpha: float = 50.0, clip_min: float = 0.85, clip_max: float = 1.20) -> Dict[str, Dict[str, float]]:
    """Compute pera€‘group fairness weights for each protected attribute.

    This implementation derives weights from representation parity statistics
    produced by StageA 3.  For each attribute and group we extract the
    selected share (``share_selected``), the base share (``base_share``) and
    the total number of selections (``total_selected``).  A shrunken
    estimate ``s_hat`` of the selected share is computed using a Bayesian
    shrinkage towards the mean of the base shares (``base_mean``) with
    smoothing parameter ``alpha``.  The raw weight is then
    ``base_share / max(s_hat, 1e-6)``, which is clipped to the range
    ``[clip_min, clip_max]``.  Finally, the weights are normalised so that
    their arithmetic mean is approximately 1.0 for each attribute.

    Parameters
    ----------
    metrics : list of dict
        Rows from the fairness metrics CSV produced by StageA 3.  Each row
        should contain at least ``attribute``, ``group``, ``share_selected``,
        ``base_share`` and ``total_selected``.
    attrs : iterable of str
        Attribute names for which to compute weights.  Names are
        casea€‘insensitive; we convert both StageA 3 attributes and resume
        attributes to lowera€‘case for matching.
    alpha : float, optional
        Smoothing parameter controlling the strength of shrinkage toward
        ``base_mean``.  Larger values produce more conservative weights.
    clip_min : float, optional
        Lower bound for weights.  Values below this threshold are clipped.
    clip_max : float, optional
        Upper bound for weights.  Values above this threshold are clipped.

    Returns
    -------
    dict
        Mapping from attribute (lowera€‘case) to a mapping of group (lowera€‘case)
        to its computed weight.
    """
    # Organise metrics by lowera€‘case attribute name, skipping summary rows
    by_attr: Dict[str, List[Dict[str, Any]]] = {}
    for row in metrics:
        attr_raw = str(row.get("attribute", "")).strip().lower()
        if not attr_raw:
            continue
        grp_raw = str(row.get("group", "")).strip().lower() or "missing"
        # Skip summary rows
        if grp_raw == "__summary__":
            continue
        # Parse numeric fields with fallbacks
        def safe_float(x: Any, default: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return default
        share_selected = safe_float(row.get("share_selected", "0"), 0.0)
        base_share = safe_float(row.get("base_share", "0"), 0.0)
        total_selected = safe_float(row.get("total_selected", "0"), 0.0)
        by_attr.setdefault(attr_raw, []).append(
            {
                "group": grp_raw,
                "share_selected": share_selected,
                "base_share": base_share,
                "total_selected": total_selected,
            }
        )
    weights: Dict[str, Dict[str, float]] = {}
    for attr in attrs:
        attr_lower = str(attr).strip().lower()
        rows = by_attr.get(attr_lower)
        if not rows:
            logger.warning("Attribute '%s' not found in metrics; assigning weight 1.0 to all groups", attr)
            weights[attr_lower] = {}
            continue
        # Compute base mean as the mean of base shares across groups.  If no
        # valid base shares, fall back to uniform distribution 1/n.
        base_values = [r["base_share"] for r in rows if r["base_share"] > 0]
        num_groups = len(rows)
        if base_values:
            base_mean = sum(base_values) / len(base_values)
        else:
            base_mean = 1.0 / num_groups if num_groups > 0 else 1.0
        # Use the same total_selected for all groups; if unspecified, assume 1.0
        N = rows[0]["total_selected"] if rows and rows[0]["total_selected"] > 0 else 1.0
        # Compute raw and clipped weights per group
        w_dict: Dict[str, float] = {}
        for r in rows:
            g = r["group"]
            share = r["share_selected"]
            base = r["base_share"]
            # Smoothed selected share s_hat
            s_hat = (share * N + alpha * base_mean) / (N + alpha)
            denom = max(s_hat, 1e-6)
            # Raw weight: base_share / s_hat; if base is zero weight will be zero
            w_raw = base / denom if base > 0 else 0.0
            # Clip into [clip_min, clip_max]
            w_clipped = max(clip_min, min(clip_max, w_raw))
            w_dict[g] = w_clipped
        # Renormalise so the mean weight a‰ˆ 1.0
        if w_dict:
            weight_sum = sum(w_dict.values())
            weight_mean = weight_sum / len(w_dict)
            weights[attr_lower] = {g: (w / weight_mean) for g, w in w_dict.items()}
        else:
            weights[attr_lower] = {}
    return weights


def adjust_recommendations(
    recs: List[Dict[str, str]],
    resumes: List[Dict[str, str]],
    resume_id_col: str,
    job_id_col: str,
    attrs: List[str],
    weights: Dict[str, Dict[str, float]],
    top_k: int,
) -> List[Dict[str, str]]:
    """Apply fairness weights and rea€‘rank recommendations by job.

    For each recommendation, multiply its original score by a combined
    weight computed as the geometric mean of the pera€‘attribute weights
    corresponding to the resumea€™s group values.  Recommendations are
    grouped by job, deduplicated on (job_id, resume_id), rea€‘ranked per
    job using the adjusted score, and trimmed to ``top_k``.  If no
    numeric score is present, the value defaults to 0.0.  The output
    rows include ``job_id``, ``resume_id``, ``score`` (the adjusted
    score), ``rank`` and ``source`` columns.

    Parameters
    ----------
    recs : list of dict
        Original recommendations containing at least the ``resume_id`` and
        ``job_id`` columns and a numeric similarity/score column.  The
        similarity column may be named ``score``, ``similarity`` or
        ``match_score``.
    resumes : list of dict
        Resume data containing the protected attribute columns.
    resume_id_col : str
        Column name in resumes that corresponds to the resume identifier.
    job_id_col : str
        Column name in recommendations that corresponds to the job identifier.
    attrs : list of str
        Protected attribute columns to adjust for.  Names are casea€‘insensitive.
    weights : dict
        Mapping from attribute (lowera€‘case) to (group -> weight).
    top_k : int
        Number of recommendations to retain per job.

    Returns
    -------
    list of dict
        Adjusted recommendation rows with columns ``job_id``, ``resume_id``,
        ``score``, ``rank`` and ``source``.
    """
    # Build resume lookup keyed by resume ID (string, stripped)
    resume_lookup: Dict[str, Dict[str, str]] = {}
    for row in resumes:
        rid_val = str(row.get(resume_id_col, "")).strip()
        if rid_val:
            resume_lookup[rid_val] = row
    # Group recommendations by job_id
    recs_by_job: Dict[str, List[Dict[str, Any]]] = {}
    for row in recs:
        # Determine job and resume IDs
        j_raw = row.get("job_id") if "job_id" in row else row.get(job_id_col)
        r_raw = row.get("resume_id") if "resume_id" in row else row.get(resume_id_col)
        job_id_val = str(j_raw).strip() if j_raw is not None else ""
        rid_val = str(r_raw).strip() if r_raw is not None else ""
        if not job_id_val or not rid_val:
            continue
        recs_by_job.setdefault(job_id_val, []).append(dict(row))  # copy to avoid side effects
    adjusted: List[Dict[str, str]] = []
    # For each job, deduplicate, compute adjusted scores and select top_k
    for job_id_val, lst in recs_by_job.items():
        # Remove duplicate (job, resume) pairs while preserving order
        seen_pairs: set = set()
        uniq: List[Dict[str, Any]] = []
        for r in lst:
            r_raw = r.get("resume_id") if "resume_id" in r else r.get(resume_id_col)
            rid_val = str(r_raw).strip() if r_raw is not None else ""
            pair = (job_id_val, rid_val)
            if not rid_val or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            uniq.append(r)
        # Compute adjusted score for each unique recommendation
        for row in uniq:
            r_raw = row.get("resume_id") if "resume_id" in row else row.get(resume_id_col)
            rid = str(r_raw).strip() if r_raw is not None else ""
            # Parse the original similarity/score; prefer 'score', fallback to 'similarity' or 'match_score'
            def parse_score(row_dict: Dict[str, Any]) -> float:
                for key in ("score", "similarity", "match_score"):
                    if key in row_dict and row_dict[key] not in (None, ""):
                        try:
                            return float(row_dict[key])
                        except Exception:
                            break
                return 0.0
            sim = parse_score(row)
            # Compute combined weight across attributes
            w_prod = 1.0
            n_attrs = 0
            for attr in attrs:
                attr_lc = str(attr).strip().lower()
                # Determine group value for this resume and attribute
                val = "missing"
                res_row = resume_lookup.get(rid)
                if res_row is not None:
                    v_raw = res_row.get(attr) or res_row.get(attr_lc)
                    if v_raw is not None:
                        val = str(v_raw).strip().lower() or "missing"
                w = weights.get(attr_lc, {}).get(val, 1.0)
                w_prod *= w
                n_attrs += 1
            combined_w = w_prod ** (1.0 / n_attrs) if n_attrs > 0 else 1.0
            row["_adj"] = sim * combined_w
        # Sort by adjusted score descending, then by resume ID ascending for stability
        uniq.sort(key=lambda d: (-d.get("_adj", 0.0), str(d.get("resume_id") if "resume_id" in d else d.get(resume_id_col, ""))))
        # Emit top_k rows with adjusted score, rank and source
        for rank_idx, row in enumerate(uniq[:top_k], start=1):
            r_raw = row.get("resume_id") if "resume_id" in row else row.get(resume_id_col)
            rid_out = str(r_raw).strip() if r_raw is not None else ""
            adj_score = row.get("_adj", 0.0)
            adjusted.append(
                {
                    "job_id": job_id_val,
                    "resume_id": rid_out,
                    "score": f"{adj_score:.6f}",
                    "rank": rank_idx,
                    "source": "node2vec_adjusted",
                }
            )
    # The adjusted list is already grouped by job and sorted within job; no further sorting required
    return adjusted


def main() -> None:
    args = parse_args()
    # Load input CSVs
    recs = read_csv_rows(args.recommendations)
    resumes = read_csv_rows(args.resumes)
    metrics = read_csv_rows(args.metrics)
    # Compute weights per attribute/group using representation parity metrics
    weights = compute_weights(metrics, args.protected_attrs)
    # Apply fairness adjustments and rea€‘rank per job
    adjusted = adjust_recommendations(
        recs=recs,
        resumes=resumes,
        resume_id_col=args.resume_id_col,
        job_id_col=args.job_id_col,
        attrs=args.protected_attrs,
        weights=weights,
        top_k=args.top_k,
    )
    # Write adjusted recommendations
    out_adj_path = Path(args.output_adjusted)
    out_adj_path.parent.mkdir(parents=True, exist_ok=True)
    # Define columns for adjusted CSV
    fieldnames = ["job_id", "resume_id", "score", "rank", "source"]
    write_csv_rows(str(out_adj_path), adjusted, fieldnames, sep=",")
    logger.info("Fairnessa€‘adjusted recommendations written to %s", out_adj_path)
    # Write weights CSV for transparency
    weight_rows: List[Dict[str, str]] = []
    for attr_name, groups in weights.items():
        for group_val, w in groups.items():
            # Use original attribute case if possible; fallback to lowera€‘case key
            weight_rows.append(
                {
                    "attribute": attr_name,
                    "group": group_val,
                    "weight": f"{w:.6f}",
                }
            )
    out_wt_path = Path(args.output_weights)
    out_wt_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(str(out_wt_path), weight_rows, ["attribute", "group", "weight"], sep=",")
    logger.info("Weights written to %s", out_wt_path)


if __name__ == "__main__":
    main()
