#!/usr/bin/env python3
"""
extract_features.py -- turn raw LLM generations into a dataset.csv with the 24 confidence
features the analysis pipeline expects.  THIS is the script that lets you ADD MORE MODELS
and MORE BENCHMARKS -- which is the only thing that genuinely strengthens the results.

It converts a generation file (one JSON object per line) plus the SQLite databases into:
    results/<name>/dataset.csv
with columns:
    exec_success_rate, agreement_ratio, n_distinct_results, semantic_entropy,
    chosen_support_ratio, chosen_executes, parse_success_rate, struct_agreement_ratio,
    n_distinct_structures, chosen_struct_support, n_joins, n_subqueries, n_aggs, parse_ok,
    chosen_mean_logprob, chosen_perplexity, chosen_token_entropy, mean_candidate_logprob,
    logprob_spread, mean_candidate_entropy, frac_columns_valid, frac_tables_valid,
    n_unknown_identifiers, schema_table_ratio, label_correct, example_id, difficulty

---------------------------------------------------------------------------------------------
HOW TO USE
---------------------------------------------------------------------------------------------
1. Install deps:      pip install sqlglot numpy
2. Edit the CONFIG block below to match YOUR generation.jsonl field names and DB paths.
3. Run:               python extract_features.py --gen path/to/generation.jsonl \
                                                  --dbdir path/to/databases \
                                                  --out results/<benchmark>__<model>/dataset.csv
4. Repeat for each new model / benchmark, then run run_experiments.py -- the new cells are
   picked up automatically.

WHAT EACH RAW RECORD MUST CONTAIN (rename in CONFIG to match your files):
  example_id   : unique id
  db_id        : database name (maps to <dbdir>/<db_id>/<db_id>.sqlite by default)
  gold_sql     : reference SQL (used only to compute the correctness label)
  candidates   : list of K generated SQL strings (the K samples)
  cand_logprobs: (optional) list of per-candidate mean token log-probabilities
  difficulty   : (optional) 'simple'/'moderate'/'challenging' etc.

If you don't have logprobs, leave CAND_LOGPROBS_FIELD = None; the four token-uncertainty
features will be written as NaN and the analysis pipeline will median-impute them (the
learned fusion still works; only that feature family is weakened).

NOTE: this is a template. Your generation format is yours, so you must set the field names
below.  Everything else (execution, parsing, schema grounding, consistency, entropy) is
implemented for you.  If a record errors, its row is written with NaNs rather than crashing
the run.
"""
import os, sys, json, argparse, sqlite3, math
from collections import Counter
import numpy as np

# ------------------------------------------------------------------ CONFIG (EDIT ME)
EXAMPLE_ID_FIELD   = "example_id"     # or "question_id", "id", ...
DB_ID_FIELD        = "db_id"
GOLD_SQL_FIELD     = "gold_sql"       # or "query", "gold", ...
CANDIDATES_FIELD   = "candidates"     # list of generated SQL strings
CAND_LOGPROBS_FIELD= "cand_logprobs"  # list of per-candidate mean logprob, or None
DIFFICULTY_FIELD   = "difficulty"     # or None
# database file location:  <dbdir>/<db_id>/<db_id>.sqlite   (Spider/BIRD layout).
def db_path(dbdir, db_id):
    return os.path.join(dbdir, db_id, f"{db_id}.sqlite")
# ------------------------------------------------------------------ end CONFIG

try:
    import sqlglot
    from sqlglot import exp
    HAVE_SQLGLOT = True
except Exception:
    HAVE_SQLGLOT = False
    print("WARNING: sqlglot not installed -> structural & schema features will be NaN. "
          "Run: pip install sqlglot")

COLS = ["exec_success_rate","agreement_ratio","n_distinct_results","semantic_entropy",
        "chosen_support_ratio","chosen_executes","parse_success_rate","struct_agreement_ratio",
        "n_distinct_structures","chosen_struct_support","n_joins","n_subqueries","n_aggs",
        "parse_ok","chosen_mean_logprob","chosen_perplexity","chosen_token_entropy",
        "mean_candidate_logprob","logprob_spread","mean_candidate_entropy","frac_columns_valid",
        "frac_tables_valid","n_unknown_identifiers","schema_table_ratio","label_correct",
        "example_id","difficulty"]


def execute(dbfile, sql, timeout=5):
    """Run SQL on the SQLite DB; return a hashable result signature or None on error."""
    try:
        con = sqlite3.connect(dbfile, timeout=timeout)
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        # order-independent multiset signature
        return frozenset(Counter(map(lambda r: tuple(map(str, r)), rows)).items())
    except Exception:
        try: con.close()
        except Exception: pass
        return None


def entropy(counts):
    tot = sum(counts)
    if tot == 0: return 0.0
    return -sum((c/tot)*math.log(c/tot + 1e-12) for c in counts if c > 0)


def schema_tables_columns(dbfile):
    """Return (set_of_tables, set_of_columns) for schema grounding."""
    tables, cols = set(), set()
    try:
        con = sqlite3.connect(dbfile); cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (t,) in cur.fetchall():
            tables.add(t.lower())
            try:
                cur.execute(f'PRAGMA table_info("{t}")')
                for row in cur.fetchall():
                    cols.add(str(row[1]).lower())
            except Exception:
                pass
        con.close()
    except Exception:
        pass
    return tables, cols


def struct_signature(sql):
    """Normalised AST signature (structure ignoring literals) via sqlglot."""
    if not HAVE_SQLGLOT: return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        # replace literals with a placeholder so structure, not values, is compared
        for lit in tree.find_all(exp.Literal):
            lit.replace(exp.Literal.string("?"))
        return tree.sql(normalize=True)
    except Exception:
        return None


def struct_counts(sql):
    """(n_joins, n_subqueries, n_aggs, parse_ok) for the chosen query."""
    if not HAVE_SQLGLOT: return (np.nan, np.nan, np.nan, 0)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        n_join = len(list(tree.find_all(exp.Join)))
        n_sub  = max(0, len(list(tree.find_all(exp.Subquery))))
        aggs   = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)
        n_agg  = sum(len(list(tree.find_all(a))) for a in aggs)
        return (n_join, n_sub, n_agg, 1)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)


def schema_grounding(sql, tables, cols):
    """(frac_columns_valid, frac_tables_valid, n_unknown_identifiers, schema_table_ratio)."""
    if not HAVE_SQLGLOT or not tables:
        return (np.nan, np.nan, np.nan, np.nan)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        used_tables = set(t.name.lower() for t in tree.find_all(exp.Table) if t.name)
        used_cols   = set(c.name.lower() for c in tree.find_all(exp.Column) if c.name)
        tv = [t in tables for t in used_tables] or [True]
        cv = [c in cols for c in used_cols] or [True]
        unknown = sum(1 for t in used_tables if t not in tables) + \
                  sum(1 for c in used_cols if c not in cols)
        ratio = (len(used_tables)/len(tables)) if tables else np.nan
        return (float(np.mean(cv)), float(np.mean(tv)), int(unknown), float(ratio))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


def features_for_record(rec, dbdir):
    ex_id = rec.get(EXAMPLE_ID_FIELD)
    db_id = rec.get(DB_ID_FIELD)
    gold  = rec.get(GOLD_SQL_FIELD)
    cands = rec.get(CANDIDATES_FIELD) or []
    logps = rec.get(CAND_LOGPROBS_FIELD) if CAND_LOGPROBS_FIELD else None
    diff  = rec.get(DIFFICULTY_FIELD) if DIFFICULTY_FIELD else ""
    dbf   = db_path(dbdir, db_id)
    K = len(cands)
    row = {c: np.nan for c in COLS}
    row["example_id"] = ex_id; row["difficulty"] = diff

    if K == 0 or not os.path.exists(dbf):
        row["label_correct"] = 0
        return row

    # execute candidates
    results = [execute(dbf, c) for c in cands]
    ok = [r is not None for r in results]
    row["exec_success_rate"] = float(np.mean(ok))
    valid = [r for r in results if r is not None]
    # majority (chosen) result
    if valid:
        cnt = Counter(valid)
        chosen_res, chosen_n = cnt.most_common(1)[0]
        row["n_distinct_results"] = len(cnt)
        row["agreement_ratio"] = chosen_n / K
        row["chosen_support_ratio"] = chosen_n / len(valid)
        row["semantic_entropy"] = entropy(list(cnt.values()))
        # chosen query = first candidate whose result == majority
        chosen_idx = next((i for i,r in enumerate(results) if r == chosen_res), 0)
        row["chosen_executes"] = 1
    else:
        chosen_idx = 0; chosen_res = None
        row["n_distinct_results"] = 0; row["agreement_ratio"] = 0.0
        row["chosen_support_ratio"] = 0.0; row["semantic_entropy"] = 0.0
        row["chosen_executes"] = 0
    chosen_sql = cands[chosen_idx]

    # structural agreement across candidates
    structs = [struct_signature(c) for c in cands]
    sok = [s is not None for s in structs]
    row["parse_success_rate"] = float(np.mean(sok))
    svalid = [s for s in structs if s is not None]
    if svalid:
        scnt = Counter(svalid)
        cs, csn = scnt.most_common(1)[0]
        row["n_distinct_structures"] = len(scnt)
        row["struct_agreement_ratio"] = csn / K
        row["chosen_struct_support"] = scnt.get(structs[chosen_idx], 0) / K
    else:
        row["n_distinct_structures"] = 0; row["struct_agreement_ratio"] = 0.0
        row["chosen_struct_support"] = 0.0
    nj, ns, na, pok = struct_counts(chosen_sql)
    row["n_joins"], row["n_subqueries"], row["n_aggs"], row["parse_ok"] = nj, ns, na, pok

    # token uncertainty (only if logprobs provided)
    if logps and len(logps) == K:
        lp = np.array(logps, float)
        row["chosen_mean_logprob"] = float(lp[chosen_idx])
        row["chosen_perplexity"] = float(np.exp(-lp[chosen_idx]))
        row["chosen_token_entropy"] = float(-lp[chosen_idx])
        row["mean_candidate_logprob"] = float(lp.mean())
        row["logprob_spread"] = float(lp.std())
        row["mean_candidate_entropy"] = float((-lp).mean())

    # schema grounding for chosen query
    tables, cols = schema_tables_columns(dbf)
    fcv, ftv, unk, ratio = schema_grounding(chosen_sql, tables, cols)
    row["frac_columns_valid"], row["frac_tables_valid"] = fcv, ftv
    row["n_unknown_identifiers"], row["schema_table_ratio"] = unk, ratio

    # correctness label: chosen result == gold result
    gold_res = execute(dbf, gold) if gold else None
    row["label_correct"] = int(chosen_res is not None and gold_res is not None and chosen_res == gold_res)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="path to generation.jsonl")
    ap.add_argument("--dbdir", required=True, help="directory containing the SQLite databases")
    ap.add_argument("--out", required=True, help="output dataset.csv path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    with open(args.gen) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except Exception:
                print(f"skip malformed line {i}"); continue
            rows.append(features_for_record(rec, args.dbdir))
            if (i+1) % 50 == 0: print(f"  processed {i+1} records ...")

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows: w.writerow(r)
    n = len(rows); acc = np.mean([r["label_correct"] for r in rows]) if rows else 0
    print(f"\nWrote {args.out}: {n} rows, base accuracy = {acc:.3f}")
    if n < 40:
        print("WARNING: fewer than 40 rows -> the analysis pipeline will skip this cell as "
              "statistically unreliable. Generate more examples for a usable model cell.")


if __name__ == "__main__":
    main()
