
import argparse, csv, collections, pandas as pd, numpy as np, os
from fairlearn.metrics import selection_rate, true_positive_rate, MetricFrame, demographic_parity_difference, demographic_parity_ratio

def load_ranked(path, jid="job_id", rid="resume_id"):
    import csv, os, collections
    per = collections.defaultdict(list)
    if not path or not os.path.exists(path):
        return per
    with open(path, encoding="utf-8-sig") as f:
        dr = csv.DictReader(f)
        has_rank = ("rank" in (dr.fieldnames or []))
        tmp = collections.defaultdict(list)
        for r in dr:
            j  = str(r.get(jid))
            rr = str(r.get(rid))
            if has_rank:
                try:
                    rk = int(r.get("rank") or 10**9)
                except:
                    rk = 10**9
            else:
                rk = len(tmp[j]) + 1  # enumerate in file order
            tmp[j].append((rr, rk))
    for j, rows in tmp.items():
        per[j] = sorted(rows, key=lambda x: x[1])
    return per

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--candidate_union", required=False)
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--sensitive_cols", nargs="+", required=True)
    ap.add_argument("--labels", required=False)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_csv", required=True)
    args=ap.parse_args()

    sel = load_ranked(args.selected)
    cand=collections.defaultdict(set)
    if args.candidate_union:
        for p in args.candidate_union.split(","):
            for j,rows in load_ranked(p).items():
                cand[j].update([r for r,_ in rows])
    for j,rows in sel.items():
        cand[j].update([r for r,_ in rows])

    sens_map={}
    for r in csv.DictReader(open(args.resumes, encoding="utf-8-sig")):
        rid=str(r.get("ID")); 
        parts=[]
        for c in args.sensitive_cols:
            parts.append(f"{c}={(r.get(c) or '').strip()}")
        sens_map[rid]="|".join(parts)

    rows=[]
    for j, pool in cand.items():
        sel_ranks={rid:rk for rid,rk in sel.get(j, [])}
        for rid in pool:
            y_pred = 1 if (rid in sel_ranks and sel_ranks[rid] <= args.top_k) else 0
            group = sens_map.get(rid,"unknown")
            rows.append((j, rid, y_pred, group))
    if not rows: raise SystemExit("No rows to audit.")
    df=pd.DataFrame(rows, columns=["job_id","resume_id","y_pred","group"])

    # labels (optional)
    df_lab=None
    if args.labels and os.path.exists(args.labels):
        L={}
        for r in csv.DictReader(open(args.labels, encoding="utf-8-sig")):
            L[(str(r["job_id"]), str(r["resume_id"]))] = str(r.get("label","0")).lower() in ("1","true","t","yes","y","1.0")
        y_true=[L.get((j,r), None) for j,r in zip(df["job_id"], df["resume_id"])]
        df["y_true"]=y_true
        df_lab=df.dropna(subset=["y_true"]).copy()
        df_lab["y_true"]=df_lab["y_true"].astype(bool)

    mf_sel = MetricFrame(metrics=selection_rate, y_true=np.zeros(len(df)), y_pred=df["y_pred"].values, sensitive_features=df["group"])
    dp_diff = demographic_parity_difference(y_true=np.zeros(len(df)), y_pred=df["y_pred"].values, sensitive_features=df["group"])
    dp_ratio= demographic_parity_ratio     (y_true=np.zeros(len(df)), y_pred=df["y_pred"].values, sensitive_features=df["group"])

    out=[]
    out.append(["metric","overall","note"])
    out.append(["selection_rate_overall", f"{float((df['y_pred']==1).mean()):.6f}", "mean of y_pred"])
    out.append(["dp_difference", f"{dp_diff:.6f}", "max-min group selection rates"])
    out.append(["dp_ratio", f"{dp_ratio:.6f}", "min/max group selection rates"])

    out.append([])
    out.append(["group","count","selection_rate"])
    for g,val in mf_sel.by_group.items():
        out.append([g, int((df["group"]==g).sum()), f"{float(val):.6f}"])

    if df_lab is not None and len(df_lab)>0:
        mf_tpr = MetricFrame(metrics=true_positive_rate, y_true=df_lab["y_true"].values, y_pred=df_lab["y_pred"].values, sensitive_features=df_lab["group"])
        out.append([])
        out.append(["metric","overall","note"]); 
        tprs={g:float(v) for g,v in mf_tpr.by_group.items()}
        tpr_diff=(max(tprs.values())-min(tprs.values())) if len(tprs)>1 else 0.0
        out.append(["tpr_diff", f"{tpr_diff:.6f}", "max-min TPR on labelled subset"])
        out.append(["group","count_labelled","TPR"])
        for g,v in tprs.items():
            out.append([g, int((df_lab["group"]==g).sum()), f"{v:.6f}"])

    with open(args.out_csv,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); [w.writerow(r if r else []) for r in out]
    print("Wrote", args.out_csv)

if __name__=="__main__": main()
