# Create a brand-new Streamlit app for GraphMatch-AI, saving to /mnt/data/streamlit_app.py
# The app includes: Welcome, Perfect Fit, Full Dataset, SHAP Explainability, Fairness, and Settings pages.
# It uses sensible defaults based on the user's folder structure and allows overriding paths in Settings.
from textwrap import dedent
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# GraphMatch-AI Dashboard a   Streamlit
# Author: Arash Mehrdad (with assistant help)
# Pages: Welcome a   Perfect Fit a   Full Dataset a   Explainability (SHAP) a   Fairness a   Settings
# Usage: `streamlit run streamlit_app.py`
# Notes:
#  - Reads CSVs with a BOM/HTML-safe loader.
#  - Defaults to the folder structure you shared; you can override on the Settings page.
#  - Designed to cope with large CSVs by filtering on job_id when possible.

import os, io, sys, json, math, csv, textwrap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="GraphMatch-AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Utilities
# -------------------------------

def load_jobs_index(paths: dict, dataset: str = "full"):
    # pick the right postings file
    if dataset == "pf":
        files = [Path(paths["pf_processed"]) / "postings_clean.csv"]
    else:
        files = [Path(paths["root"]) / "postings_clean.csv", Path(paths["root"]) / "postings.csv"]

    df = pd.DataFrame()
    for f in files:
        if f.exists():
            tmp = read_csv_smart(str(f))
            if isinstance(tmp, pd.io.parsers.TextFileReader):
                tmp = next(tmp)
            df = tmp
            break

    if df.empty:
        return pd.DataFrame(columns=["job_id", "job_title", "option"])

    cols = {c.lower(): c for c in df.columns}
    jid = cols.get("job_id") or cols.get("id") or list(df.columns)[0]

    # try common title-ish headers first, then fallback to the first non-id text column
    title_keys = ["job_title", "title", "position", "role", "name", "posting_title", "jobname", "job", "text"]
    title = None
    for key in title_keys:
        if key in cols:
            title = cols[key]
            break
    if title is None:
        # pick the first object (string-like) column that is not the id
        for c in df.columns:
            if c != jid and df[c].dtype == object:
                title = c
                break
        if title is None:
            title = jid  # worst case

    out = df[[jid, title]].copy()
    out.columns = ["job_id", "job_title"]
    out["job_id"] = out["job_id"].astype(str).str.strip()
    out["job_title"] = out["job_title"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.slice(0, 100)
    out["option"] = out["job_title"] + "  |  id=" + out["job_id"]
    out = out.dropna().drop_duplicates(subset=["job_id"])
    return out.sort_values("job_title")


def _sniff_delimiter(sample: bytes):
    try:
        return csv.Sniffer().sniff(sample.decode("utf-8", "ignore")).delimiter
    except Exception:
        return ","

def read_csv_smart(path, usecols=None, chunksize=None):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        sample = f.read(65536)
    delim = _sniff_delimiter(sample)
    # Read with Python engine to be resilient to odd quoting
    return pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=delim, usecols=usecols, chunksize=chunksize)

def load_csv_filtered(path, filter_col, value, topk_only=True):
    rows = []
    try:
        for chunk in read_csv_smart(path, chunksize=100_000):
            if filter_col in chunk.columns:
                sub = chunk[chunk[filter_col].astype(str) == str(value)]
            else:
                continue
            if topk_only and "rank" in sub.columns:
                sub = sub.sort_values(by="rank").head(10)
            rows.append(sub)
        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame()
    except FileNotFoundError:
        return pd.DataFrame()

def default_paths():
    # Hard-coded to your project layout
    root = Path(r"D:\graphmatch-ai\v_1_0\Project")
    fo = root / "Final_output"
    return dict(
        root=str(root),
        stage2=str(fo / "stage2"),
        stage2p5=str(fo / "stage2p5_out"),
        stage3=str(fo / "stage3"),
        stage4=str(fo / "stage4"),
        labels=str(root / "labels_clean_top10_strict.csv"),
        # perfect-fit (pilot)
        pf_stage2=str(fo / "perfect_fit" / "stage2"),
        pf_processed=str(fo / "perfect_fit" / "processed"),
    )

def init_state():
    if "paths" not in st.session_state:
        st.session_state.paths = default_paths()
    if "job_id" not in st.session_state:
        st.session_state.job_id = ""
    if "dataset_choice" not in st.session_state:
        st.session_state.dataset_choice = "Full Dataset"

init_state()

P = st.session_state.paths  # convenience alias

# -------------------------------
# Sidebar Navigation
# -------------------------------
with st.sidebar:
    st.image("https://media.expert.ai/expertai/uploads/2022/07/Knowledge-Graph-Creation-1024x657.jpg", width=120)
    options = ["Welcome", "Perfect Fit (Small)", "Full Dataset", "Explainability (SHAP)", "Fairness (Stage 3->4)", "Settings"]
    page = st.selectbox("Navigate", options, index=0, key="nav")
    st.markdown("---")
    st.caption("v1.0  -  Arash Mehrdad")

# -------------------------------
# Common small helpers
# -------------------------------

def job_ids_in_topk(path: str):
    if not os.path.exists(path):
        return []

    def _extract_ids(df: pd.DataFrame):
        if df is None or df.empty:
            return []
        cols = {c.lower(): c for c in df.columns}
        jid = cols.get("job_id") or cols.get("jobid") or cols.get("job") or list(df.columns)[0]
        return df[jid].astype(str).str.strip().tolist()

    try:
        obj = read_csv_smart(path)
        if isinstance(obj, pd.io.parsers.TextFileReader):
            ids = set()
            for chunk in obj:
                ids.update(_extract_ids(chunk))
            return sorted(ids)
        else:
            return sorted(set(_extract_ids(obj)))
    except Exception:
        return []


def render_stage2_page(
    dataset_name: str,
    s2_dir: str,
    s4_dir: str | None,
    postings_dataset: str,   # "pf" or "full"
    key_prefix: str,         # unique widget keys
    show_shortlist: bool = True,
):
    st.title(dataset_name)

    # ---------- Overall metrics (read from files only) ----------
    st.subheader("Overall Metrics")
    if postings_dataset == "pf":
        metrics_target = str(Path(s2_dir) / "overall_metrics_true.csv")  # PF uses the exact file
    else:
        metrics_target = s2_dir  # Full: scan dir for overall_metrics*.csv

    show_overall_by_method(metrics_target, f"{dataset_name} - Overall")

    
    little_story(
        "This section compares all matching methods over the selected dataset. "
        "P@10 is the share of correct matches in the top 10 per job; "
        "nDCG@10 rewards putting correct items higher; MAP@10 averages precision across hits."
    )

    # ---------- Per-job Top-K (optional) ----------
    if not show_shortlist:
        return

    st.markdown("---")
    st.subheader("Explore a Job Shortlist")

    method = st.selectbox(
        "Method",
        [
            "job_to_resumes_top10_tfidf.csv",
            "job_to_resumes_top10_bm25.csv",
            "job_to_resumes_top10.csv",
            "job_to_resumes_top10_graphsage.csv",
            "job_to_resumes_top10_lightgcn.csv",
        ],
        key=f"{key_prefix}_method",
    )

    topk_path = str(Path(s2_dir) / method)
    topk_ids = job_ids_in_topk(topk_path)

    jobs_idx = load_jobs_index(st.session_state.paths, dataset=postings_dataset)
    if jobs_idx.empty:
        st.warning("Could not load job titles. Check Settings -> postings_clean.csv path.")
        return

    if topk_ids:
        jobs_idx = jobs_idx[jobs_idx["job_id"].isin(topk_ids)]

    options = jobs_idx["option"].tolist()
    if not options:
        st.warning("No jobs found in this Top-K file. Try another method.")
        return

    pick = st.selectbox("Choose a job", options, index=0, key=f"{key_prefix}_job")
    job_id = jobs_idx[jobs_idx["option"].eq(pick)].iloc[0]["job_id"]
    st.caption(f"Selected job_id: {job_id}")

    post_path = None
    if s4_dir:
        use_adjusted = st.checkbox(
            "Compare with Stage 4 adjusted (if available)",
            value=True,
            key=f"{key_prefix}_adj",
        )
        cand = Path(s4_dir) / "job_to_resumes_top10_adjusted.csv"
        if use_adjusted and cand.exists():
            post_path = str(cand)

    shortlist_compare(job_id, topk_path, post_path)

    little_story(
        "The table shows the top-K resumes for the selected job. "
        "If Stage 4 is provided, we also show delta rank (post minus pre) for each resume."
    )



    # ---------- Overall metrics ----------
    st.subheader("Overall Metrics")

    if postings_dataset == "pf":
        # exact PF file: ...\perfect_fit\stage2\overall_metrics_true.csv
        metrics_target = str(Path(s2_dir) / "overall_metrics_true.csv")
    else:
        # for full dataset, pass the folder so we pick overall_metrics*.csv
        metrics_target = s2_dir

    show_overall_by_method(metrics_target, f"{dataset_name} - Overall")


    
    little_story("This section compares all matching methods over the selected dataset. "
                 "P@10 is the share of correct matches in the top 10 per job; "
                 "nDCG@10 rewards putting correct items higher; MAP@10 averages precision across hits.")

    # ---------- Explore a Job Shortlist ----------
    st.markdown("---")
    st.subheader("Explore a Job Shortlist")

    # Build a method->file mapping. Unsuffixed file = Node2Vec if metrics say so.
    def _has(name: str) -> bool:
        return (Path(s2_dir) / name).exists()

    method_map = {}

    if _has("job_to_resumes_top10_tfidf.csv"):
        method_map["TF-IDF"] = Path(s2_dir) / "job_to_resumes_top10_tfidf.csv"
    if _has("job_to_resumes_top10_bm25.csv"):
        method_map["BM25"] = Path(s2_dir) / "job_to_resumes_top10_bm25.csv"
    if _has("job_to_resumes_top10_graphsage.csv"):
        method_map["GraphSAGE"] = Path(s2_dir) / "job_to_resumes_top10_graphsage.csv"
    if _has("job_to_resumes_top10_lightgcn.csv"):
        method_map["LightGCN"] = Path(s2_dir) / "job_to_resumes_top10_lightgcn.csv"

    # Unsuffixed file used by your pipeline for Node2Vec on the big dataset
    if _has("job_to_resumes_top10.csv"):
        overall_df = load_overall_table(s2_dir) or pd.DataFrame()
        meth_col = next((c for c in overall_df.columns if str(c).lower() == "method"), None)
        has_node2vec = False
        if meth_col:
            has_node2vec = overall_df[meth_col].astype(str).str.lower().str.contains(r"\bnode2vec\b|\bn2v\b").any()
        label_for_unsuffixed = "Node2Vec" if has_node2vec else "Default"
        method_map[label_for_unsuffixed] = Path(s2_dir) / "job_to_resumes_top10.csv"

    if not method_map:
        st.warning("No Top-K files found in the Stage 2 folder.")
        return

    method_label = st.selectbox("Method", list(method_map.keys()), key=f"{key_prefix}_method")
    topk_path = str(method_map[method_label])

    



    topk_ids = job_ids_in_topk(topk_path)

    jobs_idx = load_jobs_index(st.session_state.paths, dataset=postings_dataset)
    if jobs_idx.empty:
        st.warning("Could not load job titles for this dataset. Check Settings paths for postings_clean.csv.")
        return

    if topk_ids:
        jobs_idx = jobs_idx[jobs_idx["job_id"].isin(topk_ids)]

    options = jobs_idx["option"].tolist()
    if not options:
        st.warning("No jobs found in this Top-K file. Try another method.")
        return

    pick = st.selectbox("Choose a job", options, index=0, key=f"{key_prefix}_job")
    sel = jobs_idx[jobs_idx["option"].eq(pick)]
    if sel.empty:
        st.warning("Could not resolve selected job. Try again.")
        return

    job_id = sel.iloc[0]["job_id"]
    st.caption(f"Selected job_id: {job_id}")

    post_path = None
    if s4_dir:
        use_adjusted = st.checkbox(
            "Compare with Stage 4 adjusted (if available)",
            value=True,
            key=f"{key_prefix}_adj",
        )
        cand = Path(s4_dir) / "job_to_resumes_top10_adjusted.csv"
        if use_adjusted and cand.exists():
            post_path = str(cand)

    shortlist_compare(job_id, topk_path, post_path)

    little_story("The table shows the top-K resumes for the selected job. "
                 "If Stage 4 is provided, we also show delta rank (post minus pre) for each resume.")


def load_fairness_csv(path: str):
    if not os.path.exists(path):
        return None

    df = read_csv_smart(path)
    if isinstance(df, pd.io.parsers.TextFileReader):
        df = next(df)

    # Normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Case A: already key/value columns
    if set(df.columns) >= {"key", "value"}:
        out = df[["key", "value"]].copy()

    # Case B: single text column like "metric,0.84" or "metric: 0.84"
    elif df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        m = s.str.extract(r"^(?P<key>.*?)[,=:]\s*(?P<value>-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
        out = m.dropna(how="all")

    # Case C: 2+ columns but no obvious numeric; pick first text + first other col
    else:
        key_col = next((c for c in df.columns if df[c].dtype == object), df.columns[0])
        val_col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        if val_col is None:
            # fallback to the second column; we’ll try to pull numbers later
            val_col = df.columns[1]
        out = df[[key_col, val_col]].copy()
        out.columns = ["key", "value"]

    # Coerce value; if NaN, try to pull a number from key text (e.g., "...=0.812")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    need = out["value"].isna()
    if need.any():
        extracted = out.loc[need, "key"].astype(str).str.extract(r"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        out.loc[need, "value"] = pd.to_numeric(extracted, errors="coerce")

    out = out.dropna(subset=["value"])
    out["key"] = out["key"].astype(str)
    return out




def show_overall_by_method(path_or_dir: str, title: str):
    df_raw = load_overall_table(path_or_dir)
    if df_raw is None:
        st.info("No overall metrics file found in this path.")
        st.caption(f"Source attempted: {path_or_dir}")
        return None

    df = normalize_overall_df(df_raw)
    if df.empty:
        st.info("Overall metrics file loaded but no numeric metrics found.")
        st.caption(f"Source: {path_or_dir}")
        st.dataframe(df_raw.head())
        return None
    # --- map 'default' to 'node2vec' when the unsuffixed Node2Vec file is present ---
    try:
        base = Path(path_or_dir)
        base_dir = base if base.is_dir() else base.parent
        if base_dir.exists() and (base_dir / "job_to_resumes_top10.csv").exists():
            meth_lower = df["method"].astype(str).str.lower()
            has_n2v   = meth_lower.eq("node2vec").any()
            has_def   = meth_lower.eq("default").any()
            if has_def and not has_n2v:
                df.loc[meth_lower.eq("default"), "method"] = "node2vec"
    except Exception:
        pass

        # --- keep node2vec in the plot ---
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).str.strip().str.lower()
        allowed = ["tfidf", "bm25", "node2vec", "lightgcn", "graphsage", "default"]
        df = df[df["method"].isin(allowed)]
        df["method"] = pd.Categorical(df["method"], categories=allowed, ordered=True)
        df = df.sort_values(["method"])


    st.caption(f"Source: {path_or_dir}")

    # long for grouped bar``
    long = df.melt(
        id_vars=["method"],
        value_vars=["p_at_10", "ndcg_at_10", "map_at_10"],
        var_name="metric",
        value_name="value",
    )

    fig = px.bar(long, x="metric", y="value", color="method", barmode="group", title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)
    return df




def load_overall_table(path_or_dir: str) -> pd.DataFrame | None:
    """
    Load an overall metrics table from either:
      - a direct CSV file path, or
      - a directory containing overall_metrics*.csv.
    """
    p = Path(path_or_dir)
    try:
        if p.is_file():
            return pd.read_csv(p)

        if not p.exists():
            return None

        for name in ["overall_metrics_true.csv", "overall_metrics.csv"]:
            cand = p / name
            if cand.exists():
                return pd.read_csv(cand)
    except Exception:
        return None
    return None

def normalize_overall_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # --- find the method-name column robustly ---
    lower = {c.lower(): c for c in d.columns}
    method_col = None

    # direct matches / common aliases
    for k in ["method", "model", "algo", "approach", "name"]:
        if k in lower:
            method_col = lower[k]
            break

    # if still not found: pick the first non-numeric, non-metric-looking column
    if method_col is None:
        metric_like = {"p_at_10", "ndcg_at_10", "map_at_10", "jobs_scored",
                       "p@10", "ndcg@10", "map@10", "p10", "ndcg10", "map10"}
        candidates = [
            c for c in d.columns
            if not pd.api.types.is_numeric_dtype(d[c])
            and c.lower() not in metric_like
        ]
        method_col = candidates[0] if candidates else None

    if method_col is None:
        # ultimate fallback: synthesize names (rare)
        d["method"] = [f"m{i}" for i in range(len(d))]
    else:
        d = d.rename(columns={method_col: "method"})

    # --- map metric columns flexibly ---
    d.columns = [c.lower() for c in d.columns]
    rename_map = {}
    for base in ["p_at_10", "ndcg_at_10", "map_at_10", "jobs_scored"]:
        if base in d.columns:
            rename_map[base] = base
        elif f"{base}_true" in d.columns:
            rename_map[f"{base}_true"] = base
        elif base.replace("_at_", "@") in d.columns:  # e.g., "p@10"
            rename_map[base.replace("_at_", "@")] = base
        elif base.replace("_at_", "") in d.columns:   # e.g., "p10"
            rename_map[base.replace("_at_", "")] = base
        elif base.replace("_at_", "") .replace("_", "") in d.columns:  # e.g., "ndcg10"
            rename_map[base.replace("_at_", "").replace("_", "")] = base
    d = d.rename(columns=rename_map)

    # ensure columns exist
    for col in ["p_at_10", "ndcg_at_10", "map_at_10", "jobs_scored"]:
        if col not in d.columns:
            d[col] = pd.NA

    # coerce numerics
    for col in ["p_at_10", "ndcg_at_10", "map_at_10", "jobs_scored"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # final shape
    d = d[["method", "p_at_10", "ndcg_at_10", "map_at_10", "jobs_scored"]]
    d = d.dropna(how="all", subset=["p_at_10", "ndcg_at_10", "map_at_10"])
    return d



def _norm(name: str) -> str:
    s = "".join(ch for ch in name.lower() if ch.isalnum())
    s = (s
         .replace("precision10","p10")
         .replace("pat10","p10")
         .replace("p10","p10")
         .replace("ndcg10","ndcg10")
         .replace("map10","map10"))
    return s

def metrics_from_overall_csv(path):
    try:
        df = read_csv_smart(path)
        if isinstance(df, pd.io.parsers.TextFileReader):
            df = next(df)

        cols = { _norm(c): c for c in df.columns }
        # single-row layout
        if len(df) == 1 and {"p10","ndcg10","map10"} <= set(cols.keys()):
            r = df.iloc[0]
            return dict(p_at_10=float(r[cols["p10"]]),
                        ndcg_at_10=float(r[cols["ndcg10"]]),
                        map_at_10=float(r[cols["map10"]]))

        # multi-row layout (choose tfidf row if exists, else first)
        low = {c.lower(): c for c in df.columns}
        if "method" in low:
            mcol = low["method"]
            row = df[df[mcol].astype(str).str.lower().eq("tfidf")]
            if row.empty:
                row = df.iloc[[0]]
            row = row.iloc[0]
            if {"p10","ndcg10","map10"} <= set(cols.keys()):
                return dict(p_at_10=float(row[cols["p10"]]),
                            ndcg_at_10=float(row[cols["ndcg10"]]),
                            map_at_10=float(row[cols["map10"]]))
    except Exception:
        pass
    return None

def find_overall_metrics(folder: str):
    for name in ["overall_metrics_true.csv","pf_overall_metrics_true.csv","overall_metrics.csv"]:
        p = Path(folder) / name
        if p.exists():
            m = metrics_from_overall_csv(str(p))
            if m:
                return m, str(p)
    return None, None

# ---- compute from Top-K + labels (useful for Perfect Fit fallback) ----
def _dcg_binary(rels):
    # rels is a list of 0/1 with rank starting at 1
    import math
    return sum(rel / math.log2(i+2) for i, rel in enumerate(rels))


def nav_to(page_name: str):
    st.session_state["nav"] = page_name


def job_ids_in_topk(topk_path: str):
    ids = set()
    if not os.path.exists(topk_path):
        return ids
    for chunk in read_csv_smart(topk_path, usecols=None, chunksize=200_000):
        cols = {c.lower(): c for c in chunk.columns}
        jid = cols.get("job_id") or cols.get("job") or cols.get("jid")
        if not jid:
            continue
        ids.update(chunk[jid].astype(str).str.strip().tolist())
    return ids



def _norm(name: str) -> str:
    s = "".join(ch for ch in name.lower() if ch.isalnum())
    s = (s
         .replace("precision10","p10")
         .replace("pat10","p10")
         .replace("p10","p10")
         .replace("ndcg10","ndcg10")
         .replace("map10","map10"))
    return s

def metrics_from_overall_csv(path):
    try:
        df = read_csv_smart(path)
        if isinstance(df, pd.io.parsers.TextFileReader):
            df = next(df)

        cols = { _norm(c): c for c in df.columns }
        # single-row layout
        if len(df) == 1 and {"p10","ndcg10","map10"} <= set(cols.keys()):
            r = df.iloc[0]
            return dict(p_at_10=float(r[cols["p10"]]),
                        ndcg_at_10=float(r[cols["ndcg10"]]),
                        map_at_10=float(r[cols["map10"]]))

        # multi-row layout (choose tfidf row if exists, else first)
        low = {c.lower(): c for c in df.columns}
        if "method" in low:
            mcol = low["method"]
            row = df[df[mcol].astype(str).str.lower().eq("tfidf")]
            if row.empty:
                row = df.iloc[[0]]
            row = row.iloc[0]
            if {"p10","ndcg10","map10"} <= set(cols.keys()):
                return dict(p_at_10=float(row[cols["p10"]]),
                            ndcg_at_10=float(row[cols["ndcg10"]]),
                            map_at_10=float(row[cols["map10"]]))
    except Exception:
        pass
    return None

def find_overall_metrics(folder: str):
    for name in ["overall_metrics_true.csv","pf_overall_metrics_true.csv","overall_metrics.csv"]:
        p = Path(folder) / name
        if p.exists():
            m = metrics_from_overall_csv(str(p))
            if m:
                return m, str(p)
    return None, None

# ---- compute from Top-K + labels (useful for Perfect Fit fallback) ----
def _dcg_binary(rels):
    # rels is a list of 0/1 with rank starting at 1
    import math
    return sum(rel / math.log2(i+2) for i, rel in enumerate(rels))

def compute_metrics_from_topk_labels(topk_csv: str, labels_csv: str):
    if not (os.path.exists(topk_csv) and os.path.exists(labels_csv)):
        return None
    # positives by (job_id, resume_id) and positive count by job
    pos = set()
    pos_count = {}
    for chunk in read_csv_smart(labels_csv, chunksize=100_000):
        if not {"job_id","resume_id","label"} <= set(chunk.columns):
            continue
        c = chunk[["job_id","resume_id","label"]].copy()
        c["job_id"] = c["job_id"].astype(str).str.strip()
        c["resume_id"] = c["resume_id"].astype(str).str.strip()
        c["label"] = c["label"].astype(float)
        p = c[c["label"] > 0.5]
        for j, r in zip(p["job_id"], p["resume_id"]):
            pos.add((j, r))
        pc = p.groupby("job_id")["label"].count().to_dict()
        for j, cnt in pc.items():
            pos_count[j] = pos_count.get(j, 0) + int(cnt)

    # walk top-k rows
    per_job = {}
    for chunk in read_csv_smart(topk_csv, chunksize=200_000):
        # try to find columns
        cols = {c.lower(): c for c in chunk.columns}
        jid = cols.get("job_id") or cols.get("job") or cols.get("jid")
        rid = cols.get("resume_id") or cols.get("resume") or cols.get("rid")
        rank = cols.get("rank")
        if not (jid and rid):
            continue
        sub = chunk[[jid, rid] + ([rank] if rank else [])].copy()
        sub[jid] = sub[jid].astype(str).str.strip()
        sub[rid] = sub[rid].astype(str).str.strip()
        if rank:
            sub = sub.sort_values(rank).head(10)
        else:
            sub["__rk__"] = range(1, len(sub)+1)
            rank = "__rk__"
        for _, row in sub.iterrows():
            j, r = row[jid], row[rid]
            rel = 1 if (j, r) in pos else 0
            per_job.setdefault(j, []).append(rel)

    if not per_job:
        return None

    p10s, ndcgs, aps = [], [], []
    for j, rels in per_job.items():
        rels = rels[:10] + [0]*(10 - len(rels))
        p10 = sum(rels) / 10.0
        # nDCG@10
        dcg = _dcg_binary(rels)
        ideal_n = min(10, pos_count.get(j, sum(rels)))
        idcg = _dcg_binary([1]*ideal_n + [0]*(10-ideal_n)) if ideal_n > 0 else 0.0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        # MAP@10
        hits = 0
        prec_sum = 0.0
        for i, rel in enumerate(rels, start=1):
            if rel:
                hits += 1
                prec_sum += hits / i
        denom = min(10, pos_count.get(j, hits)) if pos_count else (hits if hits>0 else 1)
        ap = (prec_sum / denom) if denom > 0 else 0.0

        p10s.append(p10); ndcgs.append(ndcg); aps.append(ap)

    return dict(p_at_10=float(np.mean(p10s)),
                ndcg_at_10=float(np.mean(ndcgs)),
                map_at_10=float(np.mean(aps)))


def render_fairness_figures(pre_csv: str, post_csv: str | None):
    pre = load_fairness_csv(pre_csv)
    st.caption(f"Using: pre = {pre_csv}" + (f" | post = {post_csv}" if post_csv else ""))
    post = load_fairness_csv(post_csv) if post_csv else None

    if pre is None or pre.empty:
        st.warning("fairlearn_pre.csv could not be parsed. Showing raw preview for debugging.")
        try:
            st.dataframe(read_csv_smart(pre_csv if isinstance(pre_csv, str) else str(pre_csv)).head(20))
        except Exception:
            pass
        return

    # Always show a quick preview so we can see the keys available
    with st.expander("Preview (first rows)"):
        st.dataframe(pre.head(20))
        if post is not None:
            st.dataframe(post.head(20))

    # ---------- Named metrics: dp_ratio / dp_difference ----------
    def pick_metric(df, pat):
        m = df[df["key"].astype(str).str.contains(pat, case=False, na=False)]
        return None if m.empty else pd.to_numeric(m.iloc[0]["value"], errors="coerce")

    rows = []
    for name, pat in [("dp_ratio", "dp.*ratio"), ("dp_difference", "dp.*difference")]:

        v_pre = pick_metric(pre, pat)
        v_post = pick_metric(post, pat) if post is not None else None
        if pd.notna(v_pre): rows.append(dict(metric=name, phase="pre", value=float(v_pre)))
        if pd.notna(v_post): rows.append(dict(metric=name, phase="post", value=float(v_post)))

    if rows:
        dfm = pd.DataFrame(rows)
        fig = px.bar(dfm, x="metric", y="value", color="phase", barmode="group",
                     title="Overall fairness metrics")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Group selection rates ----------
    def group_rates(df):
        g = df[df["key"].astype(str).str.contains("selection_rate", case=False, na=False)].copy()
        if g.empty:
            return g
        g["value"] = pd.to_numeric(g["value"], errors="coerce")

        # Try to extract the group part whether the key looks like:
        #  "selection_rate:gender:male|ethnicity:black"  OR
        #  "gender:male|ethnicity:black:selection_rate"  OR
        #  "selection_rate=gender:male|ethnicity:black"
        txt = g["key"].str.lower()

        grp = txt.str.replace(r".*selection_rate[=:]\s*", "", regex=True)             # after selection_rate
        grp = grp.where(~grp.str.contains("selection_rate"),                            # or before selection_rate
                        txt.str.replace(r"[=:]?\s*selection_rate.*", "", regex=True))

        g["group"] = grp.str.replace(r"^[:=\s,]+|[:=\s,]+$", "", regex=True)
        g = g[g["value"].notna()]
        return g[["group", "value"]]


    gpre = group_rates(pre)
    gpost = group_rates(post) if post is not None else pd.DataFrame(columns=["group", "value"])

    if not gpre.empty:
        if not gpost.empty:
            merged = gpre.merge(gpost, on="group", how="outer", suffixes=("_pre", "_post")).fillna(0)
            melt = merged.melt(id_vars=["group"], value_vars=["value_pre", "value_post"],
                               var_name="phase", value_name="value")
            melt["phase"] = melt["phase"].map({"value_pre": "pre", "value_post": "post"})
            fig2 = px.bar(melt, x="group", y="value", color="phase", barmode="group",
                          title="Selection rates by group")
        else:
            fig2 = px.bar(gpre, x="group", y="value", title="Selection rates by group (pre)")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- GENERIC FALLBACK (put at the end) ----------
    # If nothing above produced visible bars, still show the top numeric keys so the page is not blank.
    drew_anything = bool(rows) or (not gpre.empty)
    if not drew_anything:
        if "value" in pre.columns and pre["value"].notna().any():
            top_pre = pre.copy()
            top_pre["value"] = pd.to_numeric(top_pre["value"], errors="coerce")
            top_pre = top_pre[top_pre["value"].notna()].sort_values("value", ascending=False).head(10)
            st.plotly_chart(px.bar(top_pre, x="key", y="value", title="Top metrics (pre)"),
                            use_container_width=True)
        if post is not None and "value" in post.columns and post["value"].notna().any():
            top_post = post.copy()
            top_post["value"] = pd.to_numeric(top_post["value"], errors="coerce")
            top_post = top_post[top_post["value"].notna()].sort_values("value", ascending=False).head(10)
            st.plotly_chart(px.bar(top_post, x="key", y="value", title="Top metrics (post)"),
                            use_container_width=True)

    with st.expander("Help - What are we checking?"):
        st.markdown(
            "We inspect selection rates and parity ratios across protected groups. "
            "The 80 percent rule: each group's rate should be at least 0.8 times the max group's rate. "
            "Post-adjustment should improve worst-group parity while preserving match quality."
        )



def explain_metrics_box():
    with st.expander("What do P@10, nDCG@10, and MAP@10 mean? (Help)"):
        st.markdown(
            """
            - **Precision@10 (P@10)**: Of the top 10 recommended resumes for a job, what fraction are correct matches (based on labels)?
            - **nDCG@10**: Normalized Discounted Cumulative Gain. Rewards correct matches near the top more than those lower down; scaled 0-1.
            - **MAP@10**: Mean Average Precision. Averages precision every time we see a correct match within top 10, then averages over jobs.

            Why nDCG vs MAP? nDCG rewards getting true matches near the top more heavily; MAP averages precision at each hit. They can diverge if correct items are scattered.")
            """
        )


def little_story(text):
    st.markdown("> " + str(text))



def _norm(name: str) -> str:
    # normalize header keys like "P@10", "p_at_10", "Precision@10" -> "p10"
    s = "".join(ch for ch in name.lower() if ch.isalnum())
    s = s.replace("precision10", "p10").replace("pat10", "p10")
    s = s.replace("p10", "p10").replace("ndcg10", "ndcg10").replace("map10", "map10")
    return s

def metrics_from_overall_csv(path):
    try:
        df = read_csv_smart(path)
        if isinstance(df, pd.io.parsers.TextFileReader):
            df = next(df)

        # case A: single-row overall file
        if len(df) == 1:
            cols = { _norm(c): c for c in df.columns }
            p  = float(df.iloc[0][cols.get("p10")])
            nd = float(df.iloc[0][cols.get("ndcg10")])
            mp = float(df.iloc[0][cols.get("map10")])
            return dict(p_at_10=p, ndcg_at_10=nd, map_at_10=mp)

        # case B: multi-row with a "method" column; take "tfidf" row if present else first row
        lc = { c.lower(): c for c in df.columns }
        if "method" in [c.lower() for c in df.columns]:
            mcol = lc["method"]
            row = df[df[mcol].astype(str).str.lower().eq("tfidf")]
            if row.empty:
                row = df.iloc[[0]]
            cols = { _norm(c): c for c in df.columns }
            p  = float(row.iloc[0][cols.get("p10")])
            nd = float(row.iloc[0][cols.get("ndcg10")])
            mp = float(row.iloc[0][cols.get("map10")])
            return dict(p_at_10=p, ndcg_at_10=nd, map_at_10=mp)

        return None
    except Exception:
        return None

def find_overall_metrics(folder: str):
    # tries the common filenames you have
    cands = [
        Path(folder) / "overall_metrics_true.csv",
        Path(folder) / "pf_overall_metrics_true.csv",
        Path(folder) / "overall_metrics.csv",
    ]
    for p in cands:
        if p.exists():
            m = metrics_from_overall_csv(str(p))
            if m:
                return m, str(p)
    return None, None


def show_overall_cards(mets, title="Overall (from file)"):
    st.caption(title)
    a, b, c = st.columns(3)
    if mets:
        a.metric("Precision@10", f"{mets.get('p_at_10', float('nan')):.4f}")
        b.metric("nDCG@10",      f"{mets.get('ndcg_at_10', float('nan')):.4f}")
        c.metric("MAP@10",       f"{mets.get('map_at_10', float('nan')):.4f}")
    else:
        a.metric("Precision@10", "NA")
        b.metric("nDCG@10",      "NA")
        c.metric("MAP@10",       "NA")

def plot_overall_bar(method_rows, title):
    if not method_rows:
        st.info("No metrics available.")
        return
    df = pd.DataFrame(method_rows)
    fig = px.bar(df, x="metric", y="value", color="method", barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)

def shortlist_compare(job_id, pre_path, post_path=None):
    # tolerant filter: accept column aliases
    def _filter(df, value):
        cols = {c.lower(): c for c in df.columns}
        for key in ["job_id","job","jid"]:
            if key in cols:
                col = cols[key]
                return df[df[col].astype(str).str.strip() == str(value)]
        return pd.DataFrame()

    pre_rows = []
    for chunk in read_csv_smart(pre_path, chunksize=100_000):
        sub = _filter(chunk, job_id)
        if not sub.empty:
            pre_rows.append(sub)
    if not pre_rows:
        st.warning("No rows found for this job in the selected file. Try a different method file.")
        return

    pre = pd.concat(pre_rows, ignore_index=True)
    if "rank" in pre.columns:
        pre = pre.sort_values("rank").head(10)

    st.subheader("Top-K (Pre-adjust)")
    st.caption("These are the resumes ranked highest for this job by the selected method.")
    st.dataframe(pre)

    if post_path and os.path.exists(post_path):
        post_rows = []
        for chunk in read_csv_smart(post_path, chunksize=100_000):
            sub = _filter(chunk, job_id)
            if not sub.empty:
                post_rows.append(sub)
        if post_rows:
            post = pd.concat(post_rows, ignore_index=True)
            if "rank" in post.columns:
                post = post.sort_values("rank").head(10)
            st.subheader("Top-K (Post-adjust)")
            st.caption("After fairness adjustment. Delta rank shows how each resume moved.")
            cols = {c.lower(): c for c in post.columns}
            rcol = cols.get("rank","rank")
            merged = pre.merge(post[["resume_id", rcol]].rename(columns={rcol: "rank_post"}), on="resume_id", how="left")
            if "rank" in merged.columns:
                merged["delta_rank"] = merged["rank_post"] - merged["rank"]
            st.dataframe(merged)

            # small figure: bar of delta_rank (negative = improved)
            if "delta_rank" in merged.columns:
                fig = px.bar(merged, x="resume_id", y="delta_rank", title="Delta rank (post - pre)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows for this job in the adjusted file.")



def shap_global_view(shap_dir):
    path = Path(shap_dir) / "shap_global.csv"
    if not path.exists():
        st.info("shap_global.csv not found. Generate Stage 2.5 first.")
        return
    df = read_csv_smart(path)
    if isinstance(df, pd.io.parsers.TextFileReader):
        df = next(df)
    # Expect columns: feature, mean_abs_shap
    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        st.warning("Unexpected SHAP global format; need columns: feature, mean_abs_shap")
        st.dataframe(df.head(10))
        return
    st.subheader("Global Feature Importance")
    fig = px.bar(df.sort_values("mean_abs_shap", ascending=False), x="feature", y="mean_abs_shap", title="Mean |SHAP| by Feature")
    st.plotly_chart(fig, use_container_width=True)

def shap_per_job_view(shap_dir, job_id):
    path = Path(shap_dir) / f"shap_job_{job_id}.csv"
    if not path.exists():
        st.info("Per-job SHAP file not found for this job.")
        return
    df = read_csv_smart(path)
    if isinstance(df, pd.io.parsers.TextFileReader):
        df = next(df)
    st.subheader(f"Feature Importance a   Job {job_id}")
    fig = px.bar(df.sort_values(df.columns[-1], ascending=False), x=df.columns[0], y=df.columns[-1], title="Per-Job SHAP (approx.)")
    st.plotly_chart(fig, use_container_width=True)

def shap_pair_factors(shap_dir, job_id, resume_id):
    path = Path(shap_dir) / "shap_pairs_top.csv"
    if not path.exists():
        st.info("shap_pairs_top.csv not found.")
        return
    for chunk in read_csv_smart(path, chunksize=200_000):
        if "job_id" not in chunk.columns or "resume_id" not in chunk.columns:
            continue
        m = chunk[
            (chunk["job_id"].astype(str) == str(job_id)) &
            (chunk["resume_id"].astype(str) == str(resume_id))
        ]
        if len(m):
            st.dataframe(m.head(1))
            return
    st.info("No row for that (job_id, resume_id) in shap_pairs_top.csv.")


# -------------------------------
# Page: Welcome
# -------------------------------
if page == "Welcome":
    st.title("Welcome to GraphMatch-AI")
    st.markdown(
        """
        This dashboard presents the end-to-end resume->job matching pipeline:

        1. Stage 2 - Matching (TF-IDF, BM25, Node2Vec, LightGCN, GraphSAGE): produce top-K shortlists per job.
        2. Stage 2.5 - Explainability: a small surrogate model with SHAP to explain which signals matter.
        3. Stage 3 - Fairness Audit: pre-adjustment bias metrics.
        4. Stage 4 - Fairness Adjustment: re-ranked lists to improve parity.

        **Dissertation goal:** build a transparent and fair matching pipeline for large-scale job marketplaces.
        **Two datasets:** a small *Perfect Fit* pilot set and a large, real-world corpus.
        """
    )

    def welcome_pipeline_figure():
        """
        Compact stacked horizontal bars to show the pipeline flow.
        Edit the counts below to change segment widths.
        """
        # Counts (adjust if you want different widths)
        resumes = 10
        jobs = 10
        to_shap = 6
        to_audit = 14      # should be (resumes + jobs) - to_shap if you want them to sum
        to_adjust = 9

        stages = [
            ("Stage 2 inputs", {"Resumes": resumes, "Jobs": jobs}),
            ("From Matching",  {"To SHAP": to_shap, "To Audit": to_audit}),
            ("From Audit",     {"To Adjust": to_adjust})
        ]

        # Build category list and colors
        cats = []
        for _, parts in stages:
            for k in parts.keys():
                if k not in cats:
                    cats.append(k)
        palette = sns.color_palette("tab10", n_colors=len(cats))
        cat_color = {c: palette[i] for i, c in enumerate(cats)}

        # Plot
        fig, ax = plt.subplots(figsize=(8.5, 2.6))
        y_positions = np.arange(len(stages))

        for i, (stage_name, parts) in enumerate(stages):
            left = 0.0
            total = sum(parts.values())
            for cat in cats:
                v = float(parts.get(cat, 0.0))
                if v > 0:
                    ax.barh(i, v, left=left, color=cat_color[cat], edgecolor="black", linewidth=0.3)
                    # Label each segment
                    ax.text(left + v/2.0, i, f"{cat}\n{int(v)}", ha="center", va="center", fontsize=9, color="white")
                    left += v
            # Stage label on the left side
            ax.text(-0.02 * max(1.0, left), i, stage_name, ha="right", va="center", fontsize=10, color="#E5E7EB")

        ax.set_yticks([])          # we render our own labels; hide ticks
        ax.set_xlabel("count")
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout(pad=0.5)
        return fig

        

    st.caption("Flow of records through matching, explainability, audit, and adjustment. Link width ≈ volume; hover to see counts and share.")


    fig = welcome_pipeline_figure()
    st.pyplot(fig, use_container_width=True)
    st.caption("Flow through matching, explainability, audit, and adjustment. Segment width reflects simple counts.")
    # Pilot highlights (computed if files exist)
    try:
        pf2 = Path(st.session_state.paths["pf_stage2"])
        mets_pf, _ = find_overall_metrics(str(pf2))


    except Exception:
        mets_pf = None

    little_story("Tip: Use the Settings page to confirm file paths. Defaults are pre-filled for your project structure.")
    st.markdown("---")

    st.subheader("Quick Links")
    c1, c2, c3, C4, C5 = st.columns(5)
    c1.button("Perfect Fit (Small)",   on_click=nav_to, args=("Perfect Fit (Small)",))
    c2.button("Full Dataset",          on_click=nav_to, args=("Full Dataset",))
    c3.button("Explainability (SHAP)", on_click=nav_to, args=("Explainability (SHAP)",))
    C4.button("Fairness (Stage 3->4)", on_click=nav_to, args=("Fairness (Stage 3->4)",))
    C5.button("Settings",              on_click=nav_to, args=("Settings",))

    # Pilot study mini-summary card (if files exist)
    try:
        pf2 = Path(st.session_state.paths["pf_stage2"])
        labels_pf = Path(st.session_state.paths["pf_processed"]) / "labels_perfectfit_top10.csv"
        tfidf_pf = pf2 / "job_to_resumes_top10_tfidf.csv"
        mets_pf = compute_metrics_from_topk_labels(str(tfidf_pf), str(labels_pf))
    except Exception:
        mets_pf = None

    if mets_pf:
        st.markdown("### Pilot study summary")
        a, b, c = st.columns(3)
        a.metric("P@10",   f"{mets_pf['p_at_10']:.4f}")
        b.metric("nDCG@10",f"{mets_pf['ndcg_at_10']:.4f}")
        c.metric("MAP@10", f"{mets_pf['map_at_10']:.4f}")
        little_story("In the pilot set we validate the pipeline end-to-end on a tiny curated sample. \
    Higher nDCG@10 means correct matches tend to appear near the top; MAP@10 averages precision across hits within the top 10.")


   


# -------------------------------
# Page: Perfect Fit (Small)
# -------------------------------

elif page == "Perfect Fit (Small)":
    P = st.session_state.paths
    pf2 = Path(P["pf_stage2"])

    # Optional: warn if PF overall metrics file is missing
    pf_overall = pf2 / "overall_metrics_true.csv"
    if not pf_overall.exists():
        st.warning(
            "Perfect Fit overall_metrics_true.csv not found in "
            "Final_output\\perfect_fit\\stage2. The overall table will be empty."
        )

    # One unified renderer - reads metrics only from files, no fallback compute
    render_stage2_page(
    dataset_name="Perfect Fit - Pilot Dataset",
    s2_dir=str(Path(P["pf_stage2"])),   # folder: ...\perfect_fit\stage2
    s4_dir=None,
    postings_dataset="pf",              # tells the renderer to use the exact CSV
    key_prefix="pf",
    show_shortlist=False,               # PF only: hide shortlist section
    )

    explain_metrics_box()
    little_story(
        "Here we sanity-check all matching methods on the tiny Perfect Fit set, "
        "then drill into a single job to see its Top-K list."
    )



# -------------------------------
# Page: Full Dataset
# -------------------------------
elif page == "Full Dataset":
    P = st.session_state.paths
    s2_full = str(Path(P["stage2"]))
    s4_full = str(Path(P["stage4"]))
    labels_strict = P.get("labels", "")  # defaults to labels_clean_top10_strict.csv

    render_stage2_page(
        dataset_name="Full Dataset - Stage 2 Outputs",
        s2_dir=s2_full,                     # folder: ...\Final_output\stage2
        s4_dir=s4_full,                     # folder: ...\Final_output\stage4
        postings_dataset="full",            # folder scan for overall_metrics*.csv
        key_prefix="full",
    )   # (no labels_path arg)


    
    explain_metrics_box()
    little_story("This page tracks overall match quality for every method on the full corpus. \
    Use the job picker below to inspect a single job. If you also load Stage 4 adjusted results, the table shows delta rank per resume (post minus pre).")




# -------------------------------
# Page: Explainability (SHAP)
# -------------------------------
elif page == "Explainability (SHAP)":
    st.title("Stage 2.5 - Explainability with SHAP")
    st.markdown(
        "We train a small surrogate model on simple features (inverse ranks, token overlap, lengths) "
        "and compute SHAP values to explain what signals drive matches."
    )

    shap_dir = Path(P["stage2p5"])
    if not shap_dir.exists():
        st.error(f"Folder not found: {shap_dir}")
    else:
        shap_global_view(shap_dir)
        st.markdown("---")

        jobs_idx = load_jobs_index(st.session_state.paths, dataset="full")  # titles for convenience

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Per-Job Importance")
            if jobs_idx.empty:
                st.info("Load postings to enable job-title picker.")
                job_for_shap = st.text_input("job_id for per-job SHAP", st.session_state.job_id or "")
            else:
                pick = st.selectbox("Choose a job", jobs_idx["option"].tolist())
                job_for_shap = jobs_idx[jobs_idx["option"].eq(pick)].iloc[0]["job_id"]
                st.caption(f"Selected job_id: {job_for_shap}")
            if job_for_shap:
                shap_per_job_view(shap_dir, job_for_shap)

        with c2:
            st.subheader("Per-Pair Top Factors")
            # resume list for selected job
            rlist = []
            sp = Path(shap_dir) / "shap_pairs_top.csv"
            if sp.exists() and job_for_shap:
                for chunk in read_csv_smart(str(sp), chunksize=200_000):
                    if not {"job_id","resume_id"} <= set(chunk.columns):
                        continue
                    rlist.extend(chunk[chunk["job_id"].astype(str).eq(str(job_for_shap))]["resume_id"].astype(str).tolist())
                    if len(rlist) > 200:
                        break
                rlist = sorted(set(rlist))
            if rlist:
                resume_id = st.selectbox("Choose a resume_id", rlist)
                shap_pair_factors(shap_dir, job_for_shap, resume_id)
            else:
                j = st.text_input("job_id (pair)", "")
                r = st.text_input("resume_id (pair)", "")
                if j and r:
                    shap_pair_factors(shap_dir, j, r)

        with st.expander("Help - How to read SHAP here"):
            st.markdown(
                """
                - Global: which features matter overall (mean absolute SHAP).
                - Per-job: which features are influential for a specific job.
                - Per-pair: the top 1-3 features that drove a specific job-resume score.
                """
            )



# -------------------------------
# Page: Fairness (Stage 3 a   4)
# -------------------------------
elif page == "Fairness (Stage 3->4)":
    st.title("Fairness Audit & Adjustment")
    st.info("We compare overall parity (dp_ratio, dp_difference) and selection rates per group, pre vs post adjustment. Goal: improve worst-group parity while keeping match quality.")
    s3 = Path(P["stage3"]); s4 = Path(P["stage4"])
    pre = s3 / "fairlearn_pre.csv"
    post = s4 / "fairlearn_post.csv"
    if not pre.exists():
        st.warning("fairlearn_pre.csv not found.")
    else:
        render_fairness_figures(str(pre), str(post) if post.exists() else None)


    with st.expander("Help - What are we checking?"):
        st.markdown(
            """
            We inspect selection rates and parity ratios across protected groups (e.g., gender/ethnicity).
            A common heuristic is the 80% rule (four-fifths rule): a group's selection rate should be at least 0.8 times the max group's rate.
            Post-adjustment should improve the worst ratios while preserving match quality.
            """
        )


# -------------------------------
# Page: Settings
# -------------------------------
elif page == "Settings":
    st.title("Settings")
    st.caption("Set or confirm your folder paths. Defaults use your project structure.")

    P = st.session_state.paths
    def path_input(label, key):
        p = st.text_input(label, P.get(key, ""))
        if p != P.get(key, ""):
            P[key] = p

    col1, col2 = st.columns(2)
    with col1:
        path_input("Project root", "root")
        path_input("Stage 2 folder", "stage2")
        path_input("Stage 2.5 (SHAP) folder", "stage2p5")
        path_input("Stage 3 folder", "stage3")
        path_input("Stage 4 folder", "stage4")
        path_input("Labels CSV (optional, for custom eval)", "labels")
    with col2:
        path_input("Perfect-fit Stage 2 folder", "pf_stage2")
        path_input("Perfect-fit processed folder", "pf_processed")

    if st.button("Reset to detected defaults"):
        st.session_state.paths = default_paths()
        st.experimental_rerun()

    st.markdown("### Current Paths")
    st.json(st.session_state.paths)

    with st.expander("Help a   Folder layout expected"):
            st.markdown(
                            """
                            - **Final_output/stage2**: `job_to_resumes_top10_*.csv`, `overall_metrics*.csv`, `run_manifest.json`
                            - **Final_output/stage2p5_out**: `shap_global.csv`, `shap_pairs_top.csv`, `shap_job_<id>.csv`
                            - **Final_output/stage3**: `fairlearn_pre.csv`
                            - **Final_output/stage4**: `job_to_resumes_top10_adjusted.csv`, `fairlearn_post.csv`, `fairness_weights.csv`
                            - **Final_output/perfect_fit/stage2**: mini set mirrors Stage 2 structure
                            """
                        )


