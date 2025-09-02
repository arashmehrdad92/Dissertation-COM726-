#!/usr/bin/env python3
"""
GraphMatcha€‘AI a€“ StageA 3: Fairness Auditing and Metrics
=====================================================

This stage takes the recommendations produced by StageA 2 along with the
original resume dataset containing one or more *protected attributes* (for
example, gender, ethnicity or veteran status) and computes simple
fairness metrics.  The intent is to quantify whether certain groups
receive disproportionately more or fewer recommendations compared with
their representation in the applicant pool.

The auditing process works as follows:

1.  Merge the recommendations with the resume data on the resume ID.
    This enriches each recommendation with the value(s) of the
    specified protected attribute columns.
2.  For each protected attribute and each group within that attribute
    (e.g. female/male/nona€‘binary within the ``gender`` column), count
    both the number of resumes and the number of recommendations.  The
    *selection rate* for a group is the ratio of the number of
    recommendations divided by the maximum possible recommendations
    (``resume_count * top_k``).
3.  Record the raw counts and selection rates for each group in a
    CSV file.  These values can be used to compute fairness statistics
    such as disparity ratios or demographic parity differences.

To run StageA 3 you might use a command like:

    python graphmatch_stage3_fairness.py \
        --recommendations D:/GraphMatchOutput/recommendations_stage2.csv \
        --resumes D:/graphmatcha€‘ai/Test/Resume.csv \
        --protected_cols gender ethnicity \
        --output_metrics D:/GraphMatchOutput/fairness_metrics.csv \
        --top_k 5

The stage makes no assumptions about the specific attribute names or
values; it simply treats each unique value as a group and computes
counts.  Missing values are counted as a separate ``missing`` group.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple, Optional
import datetime

# We avoid pandas here; use standard library CSV helpers instead
import csv
import math
import json
import hashlib
from typing import Optional
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

from typing import Optional

def read_csv_rows(path: str, sep: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Robust CSV reader:
      - Opens with utf-8-sig to strip any BOM
      - If sep is given, use it; otherwise sniff delimiter from sample
      - Honors Excel's 'sep=,' / 'sep=\t' first-line hint
      - Falls back to comma on failure
    """
    rows: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # Peek to help Sniffer
        sample = f.read(8192)
        f.seek(0)

        # Excel hint line: e.g., "sep=," or "sep=\t"
        first_line = f.readline()
        sep_hint: Optional[str] = None
        if first_line.lower().startswith("sep="):
            hint_raw = first_line.strip()[4:]
            sep_hint = "\t" if hint_raw.startswith("\\t") else (hint_raw[:1] if hint_raw else None)
            # keep file pointer where it is (already after the hint line)
        else:
            # no hint; rewind to start
            f.seek(0)

        # If caller forces a separator, use it
        if sep is not None:
            delimiter = "\t" if sep == r"\t" else sep
            reader = csv.DictReader(f, delimiter=delimiter)
        else:
            # Try to sniff; bias candidates by putting any hint first
            candidates = [",", "\t", ";", "|"]
            if sep_hint and sep_hint in candidates:
                candidates = [sep_hint] + [c for c in candidates if c != sep_hint]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
                reader = csv.DictReader(f, dialect=dialect)
            except Exception:
                # Fallback: use hint if present, else comma
                delimiter = sep_hint if sep_hint else ","
                reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            # normalize Nones to empty strings
            rows.append({(k if k is not None else ""): ("" if v is None else str(v)) for k, v in row.items()})

    return rows


def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str], sep: str = ",") -> None:
    """
    Write a list of dictionaries to a CSV file.  The header is given by
    ``fieldnames`` and values are converted to strings.  A delimiter can be
    specified via ``sep``.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def compute_sha256(path: str) -> str:
    """
    Compute the SHA256 hex digest of a file located at ``path``.  Reads the
    file in binary mode in chunks to avoid excessive memory usage.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float, float]:
    """
    Compute the Wilson score interval for a binomial proportion.  Returns a
    tuple ``(p, lo, hi)`` where ``p`` is the point estimate of the proportion
    (successes / trials) and ``lo``/``hi`` are the lower and upper bounds of
    the 95% confidence interval.  If ``trials`` is zero the interval is
    degenerate at 0.0.
    """
    if trials <= 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    den = 1.0 + z * z / trials
    center = p + z * z / (2 * trials)
    # compute margin
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * trials)) / trials)
    lo = (center - margin) / den
    hi = (center + margin) / den
    # clamp to [0,1]
    if lo < 0.0:
        lo = 0.0
    if hi > 1.0:
        hi = 1.0
    return (p, lo, hi)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse commanda€‘line arguments for StageA 3."""
    parser = argparse.ArgumentParser(
        description="StageA 3 a€“ Fairness Auditing and Metrics"
    )
    parser.add_argument(
        "--recommendations",
        required=True,
        help="Path to the recommendations CSV produced by StageA 2",
    )
    parser.add_argument(
        "--resumes",
        required=True,
        help="Path to the original resumes CSV (must include protected attributes)",
    )
    parser.add_argument(
        "--resume_id_col",
        default="ID",
        help="Name of the resume ID column in the resumes CSV (defaults to 'ID')",
    )
    parser.add_argument(
        "--protected_cols",
        nargs='+',
        required=True,
        help=(
            "One or more column names from the resumes CSV to audit for fairness."
            " Each column should represent a protected attribute (e.g. gender, race)."
        ),
    )
    parser.add_argument(
        "--output_metrics",
        required=True,
        help="Path to write the fairness metrics CSV",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommendations per job (defaults to 5; must match StageA 2)",
    )
    parser.add_argument(
        "--job_id_col",
        default="job_id",
        help="Name of the job ID column in the recommendations CSV (defaults to 'job_id')",
    )
    parser.add_argument(
        "--min_support",
        type=int,
        default=30,
        help=(
            "Minimum number of resumes required for a group to compute parity metrics."
            " Groups with fewer resumes will have parity metrics marked as 'NA'."
        ),
    )
    parser.add_argument(
        "--intersection_of",
        nargs='*',
        default=None,
        help=(
            "Optional: list of protected attributes to form intersectional groups."
            " Example: --intersection_of gender ethnicity will audit combined groups such as 'F|Hispanic'."
        ),
    )
    parser.add_argument(
        "--per_job_output",
        default=None,
        help=(
            "Optional path to write pera€‘job fairness metrics (CSV)."
            " Each row will show the share of each group within a job's topa€‘K recommendations."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional path to write a JSON manifest capturing configuration, input file hashes, and runtime metadata."
        ),
    )
    return parser.parse_args()


def audit_fairness(
    recs: List[Dict[str, str]],
    resumes: List[Dict[str, str]],
    protected_cols: Iterable[str],
    resume_id_col: str,
    job_id_col: str,
    top_k: int,
    min_support: int = 30,
    intersection_of: Iterable[str] | None = None,
    per_job: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    Compute fairness metrics for the specified protected attributes using a
    joba€‘centric representation parity definition.  Recommendations are
    grouped by job and trimmed to the topa€‘K unique resumes per job.  The
    selection share for a group is the number of recommendations it
    receives divided by the maximum number of recommendations available
    (``num_jobs * top_k``).  This denominator is fixed across groups.

    Parameters
    ----------
    recs : list of dict
        The recommendations with at least ``resume_id`` and ``job_id`` keys.
    resumes : list of dict
        The resume data containing the protected attribute columns.
    protected_cols : iterable of str
        Names of the columns to audit.
    resume_id_col : str
        Column name to join on between recs and resumes.
    job_id_col : str
        Column name identifying jobs in the recommendations CSV.
    top_k : int
        Number of recommendations per job.  All jobs are assumed to have
        this number of slots, even if fewer recommendations exist.
    min_support : int, optional
        Minimum number of resumes required in a group to compute parity
        metrics.  Groups with fewer resumes than ``min_support`` will have
        parity metrics marked as ``"NA"``.
    intersection_of : iterable of str, optional
        When provided, a new intersectional attribute will be computed by
        concatenating the values of the specified columns with a vertical
        bar (e.g. ``gender|ethnicity``).  The intersection attribute is
        audited alongside the individual protected attributes.
    per_job : bool, optional
        If True, also compute pera€‘job group shares and return them as a
        second result.  This can be useful for diagnosing which jobs
        drive fairness gaps.

    Returns
    -------
    (metrics, per_job_details) : Tuple[List[Dict[str, Any]], List[Dict[str, Any]] or None]
        The main metrics are returned as a list of dictionaries.  Each
        dictionary contains the fields specified in the output schema.  If
        ``per_job`` is True, a second list of dictionaries is returned
        containing pera€‘job details; otherwise the second element is None.
    """
    # Prepare list of attributes to audit, including intersectional slices
    attr_list: List[str] = list(protected_cols)
    intersection_name: str | None = None
    if intersection_of:
        # Remove duplicates and preserve order
        intersection_cols = []
        seen_cols = set()
        for col in intersection_of:
            if col not in seen_cols:
                intersection_cols.append(col)
                seen_cols.add(col)
        # Only create an intersection attribute if at least two columns specified
        if len(intersection_cols) >= 2:
            intersection_name = "|".join(intersection_cols)
            attr_list.append(intersection_name)
    # Build lookup for resumes by ID
    resume_lookup: Dict[str, Dict[str, str]] = {}
    for row in resumes:
        rid = str(row.get(resume_id_col, "")).strip()
        if not rid:
            continue
        resume_lookup[rid] = row
    # Initialise group counts and display name maps
    group_counts: Dict[str, Dict[str, int]] = {}
    display_map: Dict[str, Dict[str, str]] = {}
    total_resumes = len(resume_lookup)
    for attr in attr_list:
        group_counts[attr] = {}
        display_map[attr] = {}
    # Populate resume group counts
    for rid, row in resume_lookup.items():
        for attr in attr_list:
            if attr != intersection_name:
                val = str(row.get(attr, "")).strip()
                if not val:
                    val = "missing"
                key = val.lower() if val else "missing"
                display = val
            else:
                # intersection attribute
                parts: List[str] = []
                display_parts: List[str] = []
                for sub_attr in intersection_of or []:
                    sub_val = str(row.get(sub_attr, "")).strip()
                    if not sub_val:
                        sub_val = "missing"
                    parts.append(sub_val.lower())
                    display_parts.append(sub_val)
                key = "|".join(parts)
                display = "|".join(display_parts)
            group_counts[attr][key] = group_counts[attr].get(key, 0) + 1
            if key not in display_map[attr]:
                display_map[attr][key] = display
    # Group recommendations by job and deduplicate per job
    recs_by_job: Dict[str, List[Dict[str, str]]] = {}
    for row in recs:
        job_id = str(row.get(job_id_col, "")).strip()
        if not job_id:
            continue
        rid_raw = row.get("resume_id")
        rid_alt = row.get(resume_id_col)
        rid = str(rid_raw if rid_raw is not None else rid_alt).strip()
        if not rid:
            continue
        recs_by_job.setdefault(job_id, []).append(row)
    # Deduplicate and trim to top_k per job
    selected_recs: Dict[str, List[Dict[str, str]]] = {}
    less_than_k = 0
    # Helper to parse a numeric score; fallback to zero on error
    def parse_score(row: Dict[str, str]) -> float:
        for key in ("score", "similarity", "match_score"):
            if key in row and row[key] not in (None, ""):
                try:
                    return float(row[key])
                except Exception:
                    break
        return 0.0
    for job_id, lst in recs_by_job.items():
        # Sort by descending score then ascending resume_id for deterministic order
        sorted_rows = sorted(
            lst,
            key=lambda r: (-parse_score(r), str(r.get("resume_id") or r.get(resume_id_col, "")))
        )
        seen: set[str] = set()
        selected: List[Dict[str, str]] = []
        for r in sorted_rows:
            rid_raw = r.get("resume_id")
            rid_alt = r.get(resume_id_col)
            rid_val = str(rid_raw if rid_raw is not None else rid_alt).strip()
            if not rid_val or rid_val in seen:
                continue
            seen.add(rid_val)
            selected.append(r)
            if len(selected) >= top_k:
                break
        selected_recs[job_id] = selected
        if len(selected) < top_k:
            less_than_k += 1
    num_jobs = len(selected_recs)
    total_selected = num_jobs * top_k
    if less_than_k > 0:
        logger.warning("Top-K mismatch: %d/%d jobs have < K candidates.", less_than_k, num_jobs)
    # Coverage KPI: share of jobs that achieved full top-K
    jobs_full_k_pct = (100.0 * (num_jobs - less_than_k) / num_jobs) if num_jobs > 0 else 0.0
    # Initialise recommendation counts per attribute/group
    rec_counts: Dict[str, Dict[str, int]] = {attr: {} for attr in attr_list}
    # Optionally collect per-job details
    per_job_details: List[Dict[str, Any]] | None = [] if per_job else None
    # Accumulate counts
    for job_id, lst in selected_recs.items():
        # Per-job counts for per_job details
        job_attr_counts: Dict[str, Dict[str, int]] = {attr: {} for attr in attr_list} if per_job else {}
        for r in lst:
            rid_raw = r.get("resume_id")
            rid_alt = r.get(resume_id_col)
            rid_val = str(rid_raw if rid_raw is not None else rid_alt).strip()
            resume_row = resume_lookup.get(rid_val)
            if resume_row is None:
                continue
            for attr in attr_list:
                if attr != intersection_name:
                    val = str(resume_row.get(attr, "")).strip()
                    if not val:
                        val = "missing"
                    g_key = val.lower() if val else "missing"
                else:
                    parts = []
                    for sub_attr in intersection_of or []:
                        sub_val = str(resume_row.get(sub_attr, "")).strip()
                        if not sub_val:
                            sub_val = "missing"
                        parts.append(sub_val.lower())
                    g_key = "|".join(parts)
                rec_counts[attr][g_key] = rec_counts[attr].get(g_key, 0) + 1
                if per_job:
                    job_attr_counts[attr][g_key] = job_attr_counts[attr].get(g_key, 0) + 1
        # Compute per-job share if requested
        if per_job and lst:
            for attr in attr_list:
                for g_key, cnt in job_attr_counts[attr].items():
                    share = cnt / top_k if top_k else 0.0
                    per_job_details.append(
                        {
                            "job_id": job_id,
                            "attribute": attr,
                            "group": display_map[attr].get(g_key, g_key),
                            "rec_count": cnt,
                            "share_selected": f"{share:.6f}",
                            "top_k": top_k,  
                        }
                    )
    # Compute metrics per attribute
    metrics: List[Dict[str, Any]] = []
    for attr in attr_list:
        groups = group_counts.get(attr, {})
        total_resumes_attr = sum(groups.values())
        # Build list of tuples for sorting: negative resume_count for descending, then display name
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: (-groups[k], display_map[attr].get(k, k))
        )
        # Track worst parity ratio/diff for summary
        worst_ratio: float | None = None
        worst_diff: float | None = None
        for g_key in sorted_keys:
            resume_cnt = groups.get(g_key, 0)
            rec_cnt = rec_counts[attr].get(g_key, 0)
            share, lo, hi = wilson(rec_cnt, total_selected) if total_selected > 0 else (0.0, 0.0, 0.0)
            base_share = (resume_cnt / total_resumes_attr) if total_resumes_attr > 0 else 0.0
            note: str = ""
            # Determine parity metrics (ratio and diff) based on support
            if resume_cnt < min_support or base_share <= 0.0:
                ratio = "NA"
                diff = "NA"
                if resume_cnt < min_support:
                    note = "low_support"
            else:
                # Avoid division by tiny base share
                ratio_val = share / base_share if base_share > 0 else float('inf')
                diff_val  = (share - base_share) * 100.0   # percentage points
                                # Update worst values
                if worst_ratio is None or ratio_val < worst_ratio:
                    worst_ratio = ratio_val
                if worst_diff is None or diff_val < worst_diff:
                    worst_diff = diff_val
                ratio = f"{ratio_val:.6f}"
                diff = f"{diff_val:.6f}"
            # Append record
            metrics.append(
                {
                    "attribute": attr,
                    "group": display_map[attr].get(g_key, g_key),
                    "resume_count": str(resume_cnt),
                    "total_resumes": str(total_resumes_attr),
                    "rec_count": str(rec_cnt),
                    "total_selected": str(total_selected),
                    "share_selected": f"{share:.6f}",
                    "share_selected_lo": f"{lo:.6f}",
                    "share_selected_hi": f"{hi:.6f}",
                    "base_share": f"{base_share:.6f}",
                    "parity_ratio_vs_base": ratio,
                    "parity_diff_pp": diff,
                    "note": note,
                }
            )
        # Add summary row
        # Compute worst ratio/diff from computed values
        # If no valid group, worst_ratio remains None; treat as 1.0
        summary_ratio = worst_ratio if worst_ratio is not None else 1.0
        summary_diff = worst_diff if worst_diff is not None else 0.0
        rule_80_pass = (summary_ratio >= 0.8) if isinstance(summary_ratio, (int, float)) else False
        metrics.append(
            {
                "attribute": attr,
                "group": "__SUMMARY__",
                "resume_count": "",
                "total_resumes": str(total_resumes_attr),
                "rec_count": "",
                "total_selected": str(total_selected),
                "share_selected": "",
                "share_selected_lo": "",
                "share_selected_hi": "",
                "base_share": "",
                "parity_ratio_vs_base": f"{summary_ratio:.6f}" if isinstance(summary_ratio, (int, float)) else summary_ratio,
                "parity_diff_pp": f"{summary_diff:.6f}" if isinstance(summary_diff, (int, float)) else summary_diff,
                "note": f"rule_80_pass={str(rule_80_pass).lower()}; jobs_full_k={jobs_full_k_pct:.1f}% ({num_jobs - less_than_k}/{num_jobs})",
            }
        )
    # Sort metrics by attribute, then descending resume_count, then group name
    def metric_sort_key(row: Dict[str, Any]) -> Tuple:
        attr = row.get("attribute", "")
        # For summary rows, use resume_count = -1 to push them to bottom
        try:
            rc = int(row["resume_count"]) if row["resume_count"] != "" else -1
        except Exception:
            rc = -1
        group_name = row.get("group", "")
        return (attr, -rc, group_name)
    metrics.sort(key=metric_sort_key)
    return metrics, per_job_details


def main() -> None:
    args = parse_args()
    # Load recommendations and resumes via standard CSV helpers
    rec_rows = read_csv_rows(args.recommendations)
    res_rows = read_csv_rows(args.resumes)
    # ---- Validate protected columns exist in resumes header ----

    if not res_rows:
        logger.error("Resumes CSV is empty: %s", args.resumes)
        sys.exit(1)

    res_header = set(res_rows[0].keys())
    missing = [c for c in args.protected_cols if c not in res_header]
    if missing:
        preview = ", ".join(sorted(res_header)[:40]) + ("a€¦" if len(res_header) > 40 else "")
        logger.error(
            "Missing protected column(s) in resumes CSV: %s. Available columns: %s",
            ", ".join(missing), preview
        )
        sys.exit(1)

    # If intersectional auditing is requested, those columns must exist too
    if args.intersection_of:
        miss_int = [c for c in args.intersection_of if c not in res_header]
        if miss_int:
            logger.error(
                "intersection_of references missing column(s): %s. Available columns: %s",
                ", ".join(miss_int), ", ".join(sorted(res_header))
            )
            sys.exit(1)
    # Determine if pera€‘job details should be generated
    per_job_flag = args.per_job_output is not None and args.per_job_output != ""
    # Compute fairness metrics (and pera€‘job details if requested)
    metrics, per_job_details = audit_fairness(
        recs=rec_rows,
        resumes=res_rows,
        protected_cols=args.protected_cols,
        resume_id_col=args.resume_id_col,
        job_id_col=args.job_id_col,
        top_k=args.top_k,
        min_support=args.min_support,
        intersection_of=args.intersection_of,
        per_job=per_job_flag,
    )
    # Prepare output directory
    out_path = Path(args.output_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Define header for the main metrics CSV
    fieldnames = [
        "attribute",
        "group",
        "resume_count",
        "total_resumes",
        "rec_count",
        "total_selected",
        "share_selected",
        "share_selected_lo",
        "share_selected_hi",
        "base_share",
        "parity_ratio_vs_base",
        "parity_diff_pp",
        "note",
    ]
    # Write main metrics CSV
    write_csv_rows(str(out_path), metrics, fieldnames, sep=",")
    logger.info("Fairness metrics written to %s", out_path)
    # Write per-job details if requested
    if per_job_flag and per_job_details is not None:
        pj_path = Path(args.per_job_output)
        pj_path.parent.mkdir(parents=True, exist_ok=True)
        pj_fieldnames = [args.job_id_col, "attribute", "group", "rec_count", "share_selected", "top_k"]
        write_csv_rows(str(pj_path), per_job_details, pj_fieldnames, sep=",")
        logger.info("Per-job fairness details written to %s", pj_path)
    # Write manifest JSON if requested
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Compute file hashes
        rec_hash = compute_sha256(args.recommendations)
        res_hash = compute_sha256(args.resumes)
        manifest_data = {
            "recommendations_file": args.recommendations,
            "recommendations_hash": rec_hash,
            "resumes_file": args.resumes,
            "resumes_hash": res_hash,
            "protected_cols": list(args.protected_cols),
            "intersection_of": list(args.intersection_of) if args.intersection_of else None,
            "resume_id_col": args.resume_id_col,
            "job_id_col": args.job_id_col,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)
        logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
