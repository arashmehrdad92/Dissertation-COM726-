
# GraphMatch-AI: End-to-End Pipeline (README)

This README documents the full pipeline I used for the dissertation project "GraphMatch-AI (Skill-Graph Resume-Job Matching with Bias Audit)". It covers Stage 1 to Stage 4 scripts, the surrogate SHAP explainer, and the two Colab pipelines I attached: `Final_pipeline_colab_1.py` and `Final_pipeline_colab_2.py`.

The pipeline is designed to be reproducible on Google Colab and locally on Windows (PowerShell examples provided). All file names are ASCII only.


## 1. Overview

**Goal.** Match resumes to jobs using text and graph methods, then audit and adjust results for fairness.

**Stages.**
- Stage 1: Clean resume CSV and fix IDs.
- Stage 2: Build the graph and baselines; run Node2Vec; produce top-K recommendations and accuracy metrics.
- Stage 2.5: Train a small surrogate model and explain features with SHAP.
- Stage 3: Fairness pre-audit (selection rate, demographic parity) on top-K.
- Stage 4: Fairness-aware re-ranking and transparency weights, then post-audit.
- Optional: Perfect-Fit small dataset build + experiments.
- Optional: GNN training (LightGCN and GraphSAGE) with BPR on the main dataset.

**Colab scripts.**
- `Final_pipeline_colab_1.py`: One-click orchestration for copying Stage 2 artifacts into a clean `Final_output` folder, computing truth metrics, running fairness stages, and optionally building and experimenting on the Perfect-Fit subset.
- `Final_pipeline_colab_2.py`: Trains LightGCN and GraphSAGE on the main dataset using BPR, produces J2R and R2J rankings, and appends to the truth metrics table.


## 2. Folder layout

Recommended Google Drive structure (adjust if needed):

```
MyDrive/
  Project/
    Resume_clean.csv
    postings_clean.csv
    labels_clean_top10_strict.csv          # optional but recommended
    skills_en.csv
    Skills.txt
    Technology Skills.txt
    related_skills.csv

    graphmatch_stage1_preprocess.py
    graphmatch_stage2_graph.py
    graphmatch_stage3_fairness.py
    graphmatch_stage4_fairness_adjust.py
    fairlearn_audit_topk.py
    shap_stage2p5_surrogate.py

    stage2_out/   # already generated artifacts (CSV, JSON) from Stage 2
    stage3_out/   # fairness metrics from Stage 3 (optional if already run)
    stage4_out/   # fairness adjusted CSV + weights (optional if already run)

  Final_output/
    stage2/
    stage3/
    stage4/
    perfect_fit/
      processed/
      stage2/
```

Notes:
- If my Stage 2 files use different suffixes (e.g., `_node2vec`), update the paths accordingly when calling Stage 2.5, 3, and 4.
- `Final_output/` is a clean sink for deliverables produced by the Colab scripts.


## 3. Data inputs

Minimum needed:
- `Resume_clean.csv` with ID and text columns. Common names: `ID`, `Resume_str`.
- `postings_clean.csv` with job ID and text columns. Common names: `job_id`, `description`.
- Optional labels: `labels_clean_top10_strict.csv` with columns `job_id`, `resume_id`, `label` in {0,1}.

Skill resources (optional but recommended for Stage 2 graph features):
- `skills_en.csv` (ESCO-like list),
- `Skills.txt`,
- `Technology Skills.txt`,
- `related_skills.csv`.

Keep all CSV files in UTF-8 with BOM tolerance.


## 4. Quickstart on Colab

Open both `Final_pipeline_colab_1.py` and `Final_pipeline_colab_2.py` in Colab and run top to bottom.

### 4.1 Colab pipeline A: Final_pipeline_colab_1.py

What it does:
1) Runtime check and Drive mount.
2) Define paths:
   - `PROJ` points to Project.
   - `Final_output` is the sink for clean deliverables.
3) Install deps: numpy, pandas, scikit-learn, tqdm, networkx, fairlearn, shap, rank-bm25, and PyTorch Geometric wheels matched to the runtime CUDA.
4) Copy Stage 2 files from `Project/stage2_out` into `Final_output/stage2` without re-running Node2Vec on the main dataset.
5) Compute truth metrics into `Final_output/stage2/overall_metrics_true.csv` when `labels_clean_top10_strict.csv` is present.
6) Stage 3 pre-audit: generate fairness table and group selection rates.
7) Stage 4 re-ranking: compute group weights and produce fairness-adjusted rankings; run a post-audit including TPR if labels are available.
8) Deliverables summary print-out.
9) Optional: Kaggle Perfect-Fit rescue and build (including a simple GPU Node2Vec, LightGCN/GraphSAGE demo, and P@10 evaluation).

Outputs in `Final_output/`:
- `stage2/`: copied top-K CSVs (TF-IDF, BM25, Node2Vec, and any JSON stats), plus `overall_metrics_true.csv`.
- `stage3/`: `fairness_pre.csv` (or similar), summarizing selection rates and parity.
- `stage4/`: `recommendations_stage4_fair.csv`, `fairness_weights.csv`, and post-audit CSV.


### 4.2 Colab pipeline B: Final_pipeline_colab_2.py

What it does:
1) Runtime check and Drive mount.
2) Consumes top-K candidates from TF-IDF and BM25 to define a feasible graph.
3) Builds a bipartite graph of jobs and resumes, then trains two models with BPR:
   - LightGCN
   - Mean-aggregator GraphSAGE
4) Uses early stopping against P@10, nDCG@10, or MAP@10 sampled metrics.
5) Ranks both directions:
   - job_to_resumes_top10
   - resume_to_jobs_top10
6) Appends the evaluation results to `Final_output/stage2/overall_metrics_true.csv`.
7) Saves per-model CSV outputs in `Final_output/stage2/`.

This script is self-contained and tuned for Colab GPU.


## 5. Command-line, local runs (PowerShell examples)

Below are typical invocations if I want to run the core stages locally. Change paths to match my system. All commands are PowerShell-style multi-line.

### 5.1 Stage 1: Clean resume CSV and fix IDs

```powershell
python .\graphmatch_stage1_preprocess.py `
  --input "Resume.csv" `
  --output "Resume_clean.csv" `
  --map "Resume_id_repair_map.csv" `
  --id-col "ID" `
  --resume-text-col "Resume_str" `
  --min-digits 7 `
  --emit-sep inherit
```

### 5.2 Stage 2: Build graph, baselines, Node2Vec, and top-K

```powershell
python .\graphmatch_stage2_graph.py `
  --resumes "Resume_clean.csv" `
  --jobs "postings_clean.csv" `
  --skills_en "skills_en.csv" `
  --skills_txt "Skills.txt" `
  --tech_txt "Technology Skills.txt" `
  --related "related_skills.csv" `
  --top_k 10 `
  --direction both `
  --baseline tfidf,bm25 `
  --use_gpu_node2vec `
  --n2v_dim 128 --n2v_epochs 20 `
  --n2v_walk_length 40 --n2v_context 10 `
  --n2v_walks_per_node 15 `
  --n2v_loader_bs 2048 --n2v_loader_workers 2 `
  --output_dir "stage2_out"
```

Expected Stage 2 outputs (file names may vary slightly):
- `job_to_resumes_top10_tfidf.csv`
- `job_to_resumes_top10_bm25.csv`
- `job_to_resumes_top10.csv` or `job_to_resumes_top10_node2vec.csv`
- mirrored `resume_to_jobs_top10_*.csv`
- optional stats JSON (coverage, overlaps)

### 5.3 Stage 2.5: Surrogate + SHAP explanation

```powershell
python .\shap_stage2p5_surrogate.py `
  --bm25   "stage2_out\job_to_resumes_top10_bm25.csv" `
  --tfidf  "stage2_out\job_to_resumes_top10_tfidf.csv" `
  --n2v    "stage2_out\job_to_resumes_top10.csv" `
  --jobs   "postings_clean.csv" `
  --resumes "Resume_clean.csv" `
  --labels "labels_clean_top10_strict.csv" `
  --out_dir "stage2_out\surrogate"
```

Key outputs:
- `shap_global.csv` with mean abs SHAP per feature.
- `shap_pairs_top.csv` with the top features per evaluated job-resume pair.
- `shap_job_<job_id>.csv` per job.


### 5.4 Stage 3: Fairness audit on top-K

```powershell
python .\graphmatch_stage3_fairness.py `
  --recommendations "stage2_out\job_to_resumes_top10_tfidf.csv" `
  --resumes "Resume_clean.csv" `
  --resume_id_col "ID" `
  --protected_cols gender ethnicity `
  --output_metrics "stage3_out\fairness_metrics.csv" `
  --top_k 10
```

This writes selection rates and parity metrics by group. Optionally add `--per_job_output` for per-job tables.


### 5.5 Stage 4: Fairness-aware re-ranking

```powershell
python .\graphmatch_stage4_fairness_adjust.py `
  --recommendations "stage2_out\job_to_resumes_top10_tfidf.csv" `
  --resumes "Resume_clean.csv" `
  --resume_id_col "ID" `
  --metrics "stage3_out\fairness_metrics.csv" `
  --protected_attrs gender ethnicity `
  --top_k 10 `
  --output_adjusted "stage4_out\recommendations_stage4_fair.csv" `
  --output_weights "stage4_out\fairness_weights.csv"
```

Optional: run a post-audit and TPR comparison with:

```powershell
python .\fairlearn_audit_topk.py `
  --selected "stage4_out\recommendations_stage4_fair.csv" `
  --resumes  "Resume_clean.csv" `
  --sensitive_cols gender ethnicity `
  --labels "labels_clean_top10_strict.csv" `
  --top_k 10 `
  --out_csv "stage4_out\post_audit_with_tpr.csv"
```


## 6. Metrics

- P@10: Number of relevant resumes in top 10 divided by 10 per job, averaged over jobs with at least one labelled item.
- nDCG@10: Gain is 1 for each relevant item, discounted by log2(rank+1), normalized by ideal DCG.
- MAP@10: Mean of the precision at each relevant hit within the top 10 per job, averaged over jobs.


## 7. Fairness metrics

- Selection rate by group: Share of top-K recommendations allocated to each group.
- Demographic parity difference: Max minus min selection rates across groups (lower is better).
- Demographic parity ratio: Min selection rate divided by max selection rate (closer to 1 is better).
- Optional on labelled subset: True Positive Rate (TPR) by group and TPR difference.


## 8. Model notes

- TF-IDF and BM25 are strong lexical baselines.
- Node2Vec embeddings capture shared-skill structure on the resume-job-skill graph.
- GNNs (LightGCN, GraphSAGE) are trained with BPR using candidate pools from TF-IDF and BM25 to keep the training graph focused. Both job_to_resumes and resume_to_jobs rankings are produced.


## 9. Reproducibility

- Keep all source and output paths stable.
- If labels are updated, re-run truth metrics and the fairness post-audit.
- For Colab, ensure the PyTorch Geometric wheels match the Colab CUDA version. The provided commands in `Final_pipeline_colab_1.py` handle this automatically.


## 10. Troubleshooting

- CSV delimiter issues: all scripts are BOM-tolerant and handle Excel "sep=" headers. If a script cannot find a column, check header names and whitespace.
- Node2Vec file names: if my Node2Vec CSV is named with `_node2vec`, update downstream script arguments accordingly.
- Memory: ranking uses blockwise top-K to avoid OOM. Reduce block size or K if needed.
- Missing protected attributes: Stage 3 will still run but groups will show as missing or unknown.


## 11. Deliverables checklist

- Stage 2 top-K CSVs for all methods used (TF-IDF, BM25, Node2Vec, and optionally LightGCN/GraphSAGE).
- Stage 2 truth metrics CSV (if labels available).
- Stage 3 fairness metrics CSV.
- Stage 4 adjusted recommendations and fairness weights CSV.
- Optional SHAP surrogate CSVs.
- Optional Perfect-Fit processed files and Stage 2 outputs.


## 12. How to cite

Please use Harvard style for any external datasets I included (e.g., Kaggle). Cite scripts and methods in the dissertation text as part of the Methods section.

