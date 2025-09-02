#!/usr/bin/env python3
"""
GraphMatch-AI a€“ Stage 1: Resume CSV Preprocessing (no pandas)

What it does
------------
- Opens Resume.csv with UTF-8 BOM tolerance and Excel "sep=," / "sep=\t" handling
- Sniffs delimiter if no 'sep=' line is present
- Normalizes header names (strips BOM + whitespace)
- Validates & repairs the ID column:
    * keeps good IDs as-is
    * repairs IDs embedded in HTML/text by extracting a long digit token (e.g., 1895824384)
    * otherwise synthesizes a deterministic unique ID (hash of Resume_str or row content + row index)
- Ensures IDs are unique (adds a short stable hash suffix on collision)
- Writes a mapping CSV showing kept/repaired/synthesized IDs
- Optionally writes in-place, keeping a .bak

Usage (Windows CMD)
-------------------
python graphmatch_stage1_preprocess.py ^
  --input Resume.csv ^
  --output Resume_clean.csv ^
  --map Resume_id_repair_map.csv ^
  --id-col ID ^
  --resume-text-col Resume_str ^
  --min-digits 7 ^
  --emit-sep inherit

Tips
----
- Use the cleaned file for Stage 2/3/4:  --resumes "Resume_clean.csv"
- If you want to overwrite the original, add:  --inplace
"""
import re
import html as _html
import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import tempfile
from typing import Tuple, List
from html.parser import HTMLParser
# --------- helpers ---------
import re, html as _html
from html.parser import HTMLParser

TAG_BREAKS = {"p","br","div","li","section","article","tr","td","th",
              "h1","h2","h3","h4","h5","h6","hr","header","footer"}
TAG_SKIP   = {"script","style","noscript","template"}
WS_RX      = re.compile(r"\s+")

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip  = 0
    def handle_starttag(self, tag, attrs):
        t = (tag or "").lower()
        if t in TAG_SKIP:
            self._skip += 1
        elif self._skip == 0 and t in TAG_BREAKS:
            self.parts.append(" ")
    def handle_endtag(self, tag):
        t = (tag or "").lower()
        if t in TAG_SKIP and self._skip > 0:
            self._skip -= 1
        elif self._skip == 0 and t in TAG_BREAKS:
            self.parts.append(" ")
    def handle_data(self, data):
        if self._skip == 0 and data:
            self.parts.append(data)
    def handle_comment(self, data):
        pass
    def text(self):
        return "".join(self.parts)

def strip_html_text(x: str) -> str:
    s = "" if x is None else str(x)
    s = _html.unescape(s)           # decode &lt;div&gt; a†’ <div>
    try:
        p = _TextExtractor(); p.feed(s); p.close()
        out = p.text()
    except Exception:
        out = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>|<!--.*?-->|<[^>]+>", " ", s)
    out = _html.unescape(out)       # decode &nbsp; etc.
    return WS_RX.sub(" ", out).strip()


def _norm(s: str) -> str:
    """Strip BOM and surrounding whitespace from a header/arg string."""
    return (s or "").lstrip("\ufeff").strip()

def detect_delimiter_and_skip(path: str) -> Tuple[str, int, str]:
    """
    Return (delimiter, skiprows_for_sep_line, sep_text_if_any).
    - Handles UTF-8 BOM and Excel 'sep=,' / 'sep=\\t' first line.
    - Falls back to csv.Sniffer(), else comma.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(65536)
    first = sample.splitlines()[0] if sample else ""
    if first.lower().startswith("sep="):
        raw = first[4:].strip()
        if raw in ("\\t", "\t"):
            return "\t", 1, "sep=\t"
        return (raw[:1] if raw else ","), 1, f"sep={raw[:1] if raw else ','}"
    if sample:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            return dialect.delimiter, 0, ""
        except Exception:
            return ",", 0, ""
    return ",", 0, ""

GOOD_ID_RE = re.compile(r"[A-Za-z0-9_-]{3,}$")

def good_id_format(s: str) -> bool:
    return bool(GOOD_ID_RE.fullmatch(s or ""))

def looks_like_bad_id(s: str) -> bool:
    s = (s or "").strip().lower()
    if not s:
        return True
    if any(t in s for t in ("<", ">", "div", "http", "margin-left", "class=")):
        return True
    return not good_id_format(s)

def extract_candidate_from_text(text: str, min_digits: int) -> str:
    # 1) long digit run that looks like a real identifier
    m = re.search(rf"\b(\d{{{min_digits},}})\b", text)
    if m:
        return m.group(1)
    # 2) id="SOMETHING"
    m = re.search(r'id="([A-Za-z0-9_-]{6,})"', text)
    if m:
        return m.group(1)
    return ""

def stable_hash(blob: str) -> str:
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()

def synth_id_from_row(row: List[str], header: List[str], row_index_1based: int) -> str:
    """
    Deterministic synthetic ID:
      - Prefer Resume_str column content if present (more stable),
      - else hash the whole row,
      - add row index to avoid identical hashes across dup rows.
    """
    try:
        idx = header.index("Resume_str")
        base = (row[idx] or "")
    except ValueError:
        base = "||".join((c if c is not None else "") for c in row)
    h = stable_hash(base + f"#{row_index_1based}")[:10]
    return f"syn_{h}_{row_index_1based}"


# --------- core ---------
def process(input_path: str,
            output_path: str,
            map_path: str,
            id_col: str,
            resume_text_col: str,
            min_digits: int,
            emit_sep: str,
            inplace: bool,
            strip_html_cols: list[str] | None = None,
            strip_html_all: bool = False) -> None:
    

    delim, skip, sep_text = detect_delimiter_and_skip(input_path)

    # Prepare temp outputs
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    tmp_out = tempfile.NamedTemporaryFile("w", delete=False, dir=out_dir, encoding="utf-8-sig", newline="")
    tmp_map = tempfile.NamedTemporaryFile("w", delete=False, dir=out_dir, encoding="utf-8-sig", newline="")
    tmp_out_path, tmp_map_path = tmp_out.name, tmp_map.name
    tmp_out.close(); tmp_map.close()

    kept = repaired = synthesized = collisions = 0
    used_ids = set()

    with open(input_path, "r", encoding="utf-8-sig", newline="") as inf, \
         open(tmp_out_path, "w", encoding="utf-8-sig", newline="") as outf, \
         open(tmp_map_path, "w", encoding="utf-8-sig", newline="") as mapf:

        # Decide whether to re-emit sep header
        if skip == 1 and (emit_sep == "inherit" or emit_sep == "yes"):
            outf.write(sep_text + "\n")
            inf.readline()  # skip sep line in input
        elif skip == 1:
            inf.readline()  # skip sep line, but don't emit

        rdr = csv.reader(inf, delimiter=delim)
        wtr = csv.writer(outf, delimiter=delim, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        mw  = csv.writer(mapf, delimiter=",", lineterminator="\r\n")

        header = next(rdr, None)
        if header is None:
            raise SystemExit(f"ERROR: '{input_path}' is empty (no header row).")
        header = [_norm(h) for h in header]

        # Resolve ID column (case-insensitive fallback)
        id_col_norm = _norm(id_col)
        if id_col_norm not in header:
            lower_map = {h.lower(): h for h in header}
            if id_col_norm.lower() in lower_map:
                id_col_norm = lower_map[id_col_norm.lower()]
            else:
                raise SystemExit(f"ERROR: ID column '{id_col}' not found. Headers: {header}")

        id_idx = header.index(id_col_norm)

        # Ensure Resume_str is present for hashing preference (not required)
        has_resume_str = ("Resume_str" in header)

        # Write header as-is
        wtr.writerow(header)
        mw.writerow(["row_index_1based", "old_id_sample", "new_id", "action"])

        # Row index accounting: +1 for header, +1 if sep line existed
        start_index = 3 if skip == 1 else 2

        for n, row in enumerate(rdr, start=start_index):
            if len(row) < len(header):
                row = list(row) + [""] * (len(header) - len(row))
            else:
                row = list(row)
            # Strip HTML in selected columns by name
            if strip_html_cols:
                for col in strip_html_cols:
                    try:
                        idx = header.index(col)
                        row[idx] = strip_html_text(row[idx])
                    except ValueError:
                        pass  # column not present a†’ ignore

            # Optional safety net: strip any column that still looks like HTML
            if strip_html_all:
                for i, val in enumerate(row):
                    v = "" if val is None else str(val)
                    if "<" in v and ">" in v and "</" in v:
                        row[i] = strip_html_text(v)

            old_id_raw = (row[id_idx] or "").strip()
            if looks_like_bad_id(old_id_raw):
                # Search across row text for a strong candidate
                candidate = extract_candidate_from_text(" ".join(row), min_digits)
                if candidate and good_id_format(candidate):
                    new_id = candidate
                    action = "repaired"
                else:
                    new_id = synth_id_from_row(row, header, n)
                    action = "synth"
            else:
                new_id = old_id_raw
                action = "kept"

            # Ensure uniqueness (stable suffix on collision)
            if new_id in used_ids:
                # Add a short stable hash suffix derived from content + index
                suf = stable_hash(("||".join(row) + f"@{n}"))[:6]
                new_id = f"{new_id}_{suf}"
                if new_id in used_ids:  # extremely unlikely second collision
                    new_id = f"{new_id}x"
                collisions += 1
            used_ids.add(new_id)

            row[id_idx] = new_id
            wtr.writerow(row)
            mw.writerow([n, old_id_raw[:120], new_id, action])

            if action == "kept":
                kept += 1
            elif action == "repaired":
                repaired += 1
            else:
                synthesized += 1

    # Move output into place
    if inplace:
        # backup original first
        backup = input_path + ".bak_before_stage1"
        shutil.copy2(input_path, backup)
        shutil.move(tmp_out_path, input_path)
        shutil.move(tmp_map_path, map_path)
        print(f"OK: wrote IN-PLACE '{input_path}'  (backup at '{backup}')")
    else:
        shutil.move(tmp_out_path, output_path)
        shutil.move(tmp_map_path, map_path)
        print(f"OK: wrote '{output_path}'")

    print(f"Map : {map_path}")
    print(f"Delimiter='{delim}' | sep_header={'yes' if skip==1 else 'no'}")
    print(f"Rows kept        : {kept}")
    print(f"Rows repaired    : {repaired}")
    print(f"Rows synthesized : {synthesized}")
    print(f"Total out        : {kept + repaired + synthesized}")
    print(f"ID collisions resolved: {collisions}")
def process_jobs(input_path: str,
                 output_path: str,
                 job_text_col: str = "description",
                 merge_skill_col: str | None = "skills_desc",
                 keep_cols_csv: str | None = None,
                 drop_cols_csv: str | None = None,
                 min_desc_chars: int = 40,
                 strip_html_all: bool = False) -> None:
    """Clean and slim postings.csv."""
    import csv

    with open(input_path, encoding="utf-8-sig", newline="") as f_in:
        rdr = csv.DictReader(f_in)
        header = [ _norm(h) for h in (rdr.fieldnames or []) ]

        default_keep = [
            "job_id","title","company_name","location",
            job_text_col, "listed_time","formatted_experience_level",
            "work_type","remote_allowed",
            "min_salary","med_salary","max_salary","currency","pay_period",
            "compensation_type","normalized_salary",
            "job_posting_url","posting_domain","company_id","zip_code","fips"
        ]
        keep_cols  = [ _norm(c) for c in (keep_cols_csv or "").split(",") if c.strip() ] or default_keep
        drop_cols  = set(_norm(c) for c in (drop_cols_csv or "").split(",") if c.strip())

        # keep only existing columns
        keep_cols = [c for c in keep_cols if c in header]
        if job_text_col not in keep_cols and job_text_col in header:
            keep_cols.append(job_text_col)

        with open(output_path, "w", encoding="utf-8-sig", newline="") as f_out:
            wtr = None
            seen_job = set()
            n_in = n_out = n_short = n_empty = n_dupe = 0

            for row in rdr:
                n_in += 1
                row = { _norm(k): v for k,v in row.items() }

                # merge description + skills
                desc = str(row.get(job_text_col,"") or "")
                if merge_skill_col and (merge_skill_col in row):
                    desc = (desc + " " + str(row.get(merge_skill_col,"") or "")).strip()

                # clean HTML/entities
                desc_clean = strip_html_text(desc) if (strip_html_all or "<" in desc or "&lt;" in desc) else desc.strip()

                # drop empties / too short
                if not desc_clean:
                    n_empty += 1; continue
                if len(desc_clean) < min_desc_chars:
                    n_short += 1; continue

                # dedupe by job_id (keep first)
                job_id = str(row.get("job_id","")).strip()
                if job_id:
                    if job_id in seen_job:
                        n_dupe += 1; continue
                    seen_job.add(job_id)

                # build slim row
                out = {}
                for c in keep_cols:
                    if c in drop_cols:
                        continue
                    out[c] = desc_clean if c == job_text_col else (row.get(c,"") or "").strip()

                if wtr is None:
                    wtr = csv.DictWriter(f_out, fieldnames=list(out.keys()))
                    wtr.writeheader()
                wtr.writerow(out)
                n_out += 1

    print(f"[jobs] in={n_in} out={n_out} dropped_empty={n_empty} "
          f"dropped_short<{min_desc_chars}={n_short} deduped={n_dupe} "
          f"kept_cols={len([c for c in keep_cols if c not in drop_cols])}")
# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="Stage 1: preprocess Resume.csv (ID cleanup, BOM/delimiter tolerant)")
    ap.add_argument("--input", required=True, help="Path to source Resume.csv")
    ap.add_argument("--output", default="Resume_clean.csv", help="Path to write cleaned CSV (ignored if --inplace)")
    ap.add_argument("--map", default="Resume_id_repair_map.csv", help="Path to write ID repair map CSV")
    ap.add_argument("--id-col", default="ID", help="Name of the resume ID column (default: ID)")
    ap.add_argument("--resume-text-col", default="Resume_str", help="Name of resume text column (for hashing preference)")
    ap.add_argument("--min-digits", type=int, default=7, help="Min digits to consider a numeric token a real ID")
    ap.add_argument("--emit-sep", choices=["inherit", "yes", "no"], default="inherit",
                    help="Emit 'sep=...' line: inherit from input (default), force yes, or no")
    ap.add_argument("--inplace", action="store_true", help="Overwrite --input (keeps a .bak)")
    ap.add_argument("--strip_html_cols",
                default="Resume_str",
                help="Comma-separated list of text columns to HTML-strip (default: Resume_str). Use '' to disable.")
    ap.add_argument("--strip_html_all", action="store_true",
        help="Additionally strip HTML from any column that still contains tags (heuristic).")
        # --- jobs cleaning args ---
    ap.add_argument("--jobs_in",  help="Path to postings.csv to clean")
    ap.add_argument("--jobs_out", help="Where to write cleaned postings (e.g., postings_clean.csv)")
    ap.add_argument("--job_text_col", default="description", help="Job text column to clean/keep")
    ap.add_argument("--job_merge_col", default="skills_desc", help="If present, merge into job_text_col before cleaning")
    ap.add_argument("--job_keep_cols", default="", help="Comma-separated keep list (default set if empty)")
    ap.add_argument("--job_drop_cols", default="", help="Comma-separated columns to drop from keep list")
    ap.add_argument("--min_job_desc_chars", type=int, default=40, help="Drop jobs with description shorter than this")
    ap.add_argument("--strip_html_jobs_all", action="store_true", help="Strip HTML from job descriptions even if no tags detected")

    args = ap.parse_args()

    strip_cols_norm = [_norm(c) for c in (args.strip_html_cols or "").split(",") if c.strip()]


    # Normalize arg strings (kill any stray BOM/whitespace)
    args.id_col = _norm(args.id_col)
    args.resume_text_col = _norm(args.resume_text_col)

    process(
    input_path=args.input,
    output_path=args.output,
    map_path=args.map,
    id_col=args.id_col,
    resume_text_col=args.resume_text_col,
    min_digits=args.min_digits,
    emit_sep=args.emit_sep,
    inplace=args.inplace,
    strip_html_cols=strip_cols_norm,   # << added
    strip_html_all=args.strip_html_all # << added
    )
    # Optional postings cleaning step (runs only if both paths are provided)
    if args.jobs_in and args.jobs_out:
        process_jobs(
            input_path=args.jobs_in,
            output_path=args.jobs_out,
            job_text_col=_norm(args.job_text_col),
            merge_skill_col=_norm(args.job_merge_col) if args.job_merge_col else None,
            keep_cols_csv=args.job_keep_cols,
            drop_cols_csv=args.job_drop_cols,
            min_desc_chars=args.min_job_desc_chars,
            strip_html_all=bool(args.strip_html_jobs_all),
        )

if __name__ == "__main__":
    main()

