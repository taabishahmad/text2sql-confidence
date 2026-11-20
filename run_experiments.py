#!/usr/bin/env python3
"""
run_experiments.py -- Certified Abstention for Text-to-SQL: full experiment suite.

Auto-discovers every results/<name>/dataset.csv and runs, for each:
  1. Backend comparison (heuristic / CRC / RCPS / LtT) across risk tolerances
  2. Score certifiability (learned fusion / token logprob / self-consistency / semantic entropy)
  3. Calibration-size study
  4. Sensitivity to delta and grid size m
  5. Error analysis (correct vs incorrect feature means)
  6. Per-difficulty breakdown (if a 'difficulty' column exists)
Plus, across dataset pairs:
  7. Distribution-shift stress test (calibrate on source, deploy on target)
  8. Deployment economics

Outputs (written to ./outputs/):
  - all_results.json      : every number, machine-readable
  - summary.txt           : human-readable summary
  - tables/*.tex          : ready-to-paste LaTeX tables
Run:  python run_experiments.py
Then zip the whole ./outputs/ folder and send it back.

Adding MORE data (more models / benchmarks) needs NO code change: just drop another
results/<name>/dataset.csv with the same 24 feature columns + label_correct (+ optional
difficulty) and re-run.
"""
import os, sys, json, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conformal import calibrate_threshold           # Learn-then-Test
from baselines import heuristic, crc, rcps, evaluate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

# ----------------------------- config -----------------------------
ALPHAS      = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
DELTA       = 0.10
N_TRIALS    = 300          # random calibration/deployment splits
GRID_M      = 100
N_MIN       = 20
SEED        = 0
MIN_ROWS    = 40           # skip datasets smaller than this (unreliable)
OUT         = "outputs"
NON_FEATURE = {"label_correct", "example_id", "difficulty"}
SELF_CONSISTENCY_COL = "agreement_ratio"
SEMANTIC_ENTROPY_COL = "semantic_entropy"
TOKEN_LOGPROB_COL    = "chosen_mean_logprob"
# ------------------------------------------------------------------


def features(df):
    return df[[c for c in df.columns if c not in NON_FEATURE]].dropna(axis=1, how="all")


def oof_fusion(df, y, seed=42):
    """Out-of-fold, isotonic-calibrated learned fusion score (no label leakage)."""
    X = features(df)
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X))
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = HistGradientBoostingClassifier(max_depth=3, random_state=seed).fit(X.iloc[tr], y[tr])
        ir = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(X.iloc[tr])[:, 1], y[tr])
        oof[te] = ir.predict(m.predict_proba(X.iloc[te])[:, 1])
    return oof


def norm01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-9)


def backend_bench(conf, y, alpha, delta=DELTA, trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y)
    M = {
        "Heuristic": lambda c, l: heuristic(c, l, alpha, N_MIN, GRID_M),
        "CRC":       lambda c, l: crc(c, l, alpha, N_MIN, GRID_M),
        "RCPS":      lambda c, l: rcps(c, l, alpha, delta, N_MIN, GRID_M),
        "LtT":       lambda c, l: calibrate_threshold(c, l, alpha, delta, GRID_M, N_MIN),
    }
    agg = {k: {"cov": [], "risk": [], "viol": 0, "cert": 0} for k in M}
    for _ in range(trials):
        idx = rng.permutation(n); nc = n // 2; cal, te = idx[:nc], idx[nc:]
        for name, fn in M.items():
            lam = fn(conf[cal], y[cal])
            r = evaluate(conf[te], y[te], lam)
            if r is None:
                continue
            a = agg[name]; a["cert"] += 1; a["cov"].append(r["coverage"]); a["risk"].append(r["risk"])
            if r["risk"] > alpha:
                a["viol"] += 1
    out = {}
    for k, a in agg.items():
        out[k] = {
            "cert_rate": a["cert"] / trials,
            "coverage": float(np.mean(a["cov"])) if a["cov"] else None,
            "risk": float(np.mean(a["risk"])) if a["risk"] else None,
            "viol_rate": a["viol"] / a["cert"] if a["cert"] else None,
        }
    return out


def score_certifiability(df, y, alpha=0.25, delta=DELTA, trials=N_TRIALS, seed=SEED):
    scores = {"Learned fusion": oof_fusion(df, y)}
    if SELF_CONSISTENCY_COL in df:
        scores["Self-consistency"] = norm01(df[SELF_CONSISTENCY_COL].to_numpy(float))
    if SEMANTIC_ENTROPY_COL in df:
        scores["Semantic entropy"] = norm01(-df[SEMANTIC_ENTROPY_COL].to_numpy(float))
    if TOKEN_LOGPROB_COL in df:
        v = df[TOKEN_LOGPROB_COL]
        scores["Token logprob"] = norm01(v.fillna(v.median()).to_numpy(float))
    rng = np.random.default_rng(seed); n = len(y); out = {}
    for name, s in scores.items():
        cov = []; cert = 0
        for _ in range(trials):
            idx = rng.permutation(n); nc = n // 2; cal, te = idx[:nc], idx[nc:]
            lam = calibrate_threshold(s[cal], y[cal], alpha, delta, GRID_M, N_MIN)
            r = evaluate(s[te], y[te], lam)
            if r is None:
                continue
            cert += 1; cov.append(r["coverage"])
        out[name] = {"cert_rate": cert / trials, "coverage": float(np.mean(cov)) if cov else None}
    return out


def calibration_size(df, y, alpha=0.25, delta=DELTA, trials=N_TRIALS, seed=SEED):
    s = oof_fusion(df, y); rng = np.random.default_rng(seed); n = len(y); out = {}
    for ncal in [50, 100, 150, 200, 250, 300]:
        if ncal >= n:
            continue
        cov = []; cert = 0
        for _ in range(trials):
            idx = rng.permutation(n); cal, te = idx[:ncal], idx[ncal:]
            lam = calibrate_threshold(s[cal], y[cal], alpha, delta, GRID_M, N_MIN)
            r = evaluate(s[te], y[te], lam)
            if r is None:
                continue
            cert += 1; cov.append(r["coverage"])
        out[str(ncal)] = {"cert_rate": cert / trials, "coverage": float(np.mean(cov)) if cov else None}
    return out


def sensitivity(df, y, alpha=0.25, trials=200, seed=SEED):
    s = oof_fusion(df, y); rng = np.random.default_rng(seed); n = len(y)
    def run(delta, m):
        cov = []; cert = 0; viol = 0
        for _ in range(trials):
            idx = rng.permutation(n); nc = n // 2; cal, te = idx[:nc], idx[nc:]
            lam = calibrate_threshold(s[cal], y[cal], alpha, delta, m, N_MIN)
            r = evaluate(s[te], y[te], lam)
            if r is None:
                continue
            cert += 1; cov.append(r["coverage"])
            if r["risk"] > alpha:
                viol += 1
        return {"cert_rate": cert / trials, "coverage": float(np.mean(cov)) if cov else None,
                "viol_rate": viol / cert if cert else None}
    out = {"delta": {}, "grid": {}}
    for d in [0.01, 0.05, 0.10, 0.20]:
        out["delta"][str(d)] = run(d, GRID_M)
    for m in [20, 50, 100, 200, 500]:
        out["grid"][str(m)] = run(DELTA, m)
    return out


def error_analysis(df, y):
    cols = ["agreement_ratio", "semantic_entropy", "struct_agreement_ratio",
            "n_joins", "frac_columns_valid", "chosen_perplexity"]
    out = {}
    for c in cols:
        if c in df:
            out[c] = {"correct": float(df.loc[y == 1, c].mean()),
                      "incorrect": float(df.loc[y == 0, c].mean())}
    return out


def per_difficulty(df):
    if "difficulty" not in df:
        return None
    out = {}
    for d, g in df.groupby("difficulty"):
        out[str(d)] = {"n": int(len(g)), "accuracy": float(g["label_correct"].mean())}
    return out


def shift_test(df_src, y_src, df_tgt, y_tgt, alpha=0.25, delta=DELTA, trials=200, seed=SEED):
    Xs, Xt = features(df_src), features(df_tgt)
    common = [c for c in Xs.columns if c in Xt.columns]
    imp = SimpleImputer(strategy="median").fit(Xs[common])
    Xs2 = pd.DataFrame(imp.transform(Xs[common])); Xt2 = pd.DataFrame(imp.transform(Xt[common]))
    m = HistGradientBoostingClassifier(max_depth=3, random_state=42).fit(Xs2, y_src)
    ps = m.predict_proba(Xs2)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip").fit(ps, y_src)
    conf_s = ir.predict(ps); conf_t = ir.predict(m.predict_proba(Xt2)[:, 1])
    rng = np.random.default_rng(seed); out = {}
    for name, fn in {"RCPS": lambda c, l: rcps(c, l, alpha, delta, N_MIN, GRID_M),
                     "LtT": lambda c, l: calibrate_threshold(c, l, alpha, delta, GRID_M, N_MIN)}.items():
        # in-distribution (source->source)
        vi = 0; ci = 0; covi = []
        for _ in range(trials):
            idx = rng.permutation(len(y_src)); nc = len(y_src) // 2; cal, te = idx[:nc], idx[nc:]
            lam = fn(conf_s[cal], y_src[cal]); r = evaluate(conf_s[te], y_src[te], lam)
            if r is None:
                continue
            ci += 1; covi.append(r["coverage"])
            if r["risk"] > alpha:
                vi += 1
        # shifted (source->target)
        vs = 0; cs = 0; covs = []; risks = []
        for _ in range(trials):
            cidx = rng.choice(len(y_src), len(y_src) // 2, replace=False)
            tidx = rng.choice(len(y_tgt), len(y_tgt) // 2, replace=False)
            lam = fn(conf_s[cidx], y_src[cidx]); r = evaluate(conf_t[tidx], y_tgt[tidx], lam)
            if r is None:
                continue
            cs += 1; covs.append(r["coverage"]); risks.append(r["risk"])
            if r["risk"] > alpha:
                vs += 1
        out[name] = {
            "in_dist": {"cert_rate": ci / trials, "coverage": float(np.mean(covi)) if covi else None,
                        "viol_rate": vi / ci if ci else None},
            "shifted": {"cert_rate": cs / trials, "coverage": float(np.mean(covs)) if covs else None,
                        "risk": float(np.mean(risks)) if risks else None,
                        "viol_rate": vs / cs if cs else None},
        }
    return out


def economics(df, y, alpha=0.25, delta=DELTA):
    s = oof_fusion(df, y)
    lam = rcps(s, y, alpha, delta, N_MIN, GRID_M) or calibrate_threshold(s, y, alpha, delta, GRID_M, N_MIN)
    r = evaluate(s, y, lam)
    if r is None:
        return None
    cov, risk, acc = r["coverage"], r["risk"], float(y.mean())
    N = 1000; out = {"coverage": cov, "risk": risk}
    for ce, cr in [(100, 5), (500, 5), (1000, 10)]:
        no_abs = N * (1 - acc) * ce
        cert = N * cov * risk * ce + N * (1 - cov) * cr
        out[f"{ce}/{cr}"] = {"no_abstention": no_abs, "certified": cert,
                             "saving_pct": 100 * (1 - cert / no_abs)}
    return out


# ----------------------------- LaTeX helpers -----------------------------
def tex_backend(tag, res):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"$\alpha$ & Backend & Coverage & Risk & Viol. \\", r"\midrule"]
    for a in ["0.2", "0.25", "0.3"]:
        if a not in res:
            continue
        for bk in ["Heuristic", "CRC", "RCPS", "LtT"]:
            v = res[a][bk]
            cov = f"{v['coverage']:.3f}" if v['coverage'] is not None else "---"
            rk = f"{v['risk']:.3f}" if v['risk'] is not None else "---"
            vl = f"{100*v['viol_rate']:.1f}\\%" if v['viol_rate'] is not None else "refuses"
            lines.append(f"{a} & {bk} & {cov} & {rk} & {vl} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    paths = sorted(glob.glob("results/*/dataset.csv"))
    if not paths:
        print("ERROR: no results/*/dataset.csv found. Run from the folder containing results/.")
        sys.exit(1)

    data = {}
    for p in paths:
        name = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p)
        if "label_correct" not in df or len(df) < MIN_ROWS:
            print(f"skip {name} (rows={len(df)})")
            continue
        data[name] = df
        print(f"loaded {name}: {len(df)} rows, acc={df['label_correct'].mean():.3f}")

    ALL = {"config": {"alphas": ALPHAS, "delta": DELTA, "n_trials": N_TRIALS,
                      "grid_m": GRID_M, "n_min": N_MIN, "seed": SEED}, "datasets": {}}
    summ = ["CERTIFIED ABSTENTION -- EXPERIMENT SUMMARY", "=" * 50]

    for name, df in data.items():
        y = df["label_correct"].to_numpy(int)
        print(f"\n=== {name} (n={len(y)}, acc={y.mean():.3f}) ===")
        conf = oof_fusion(df, y)
        d = {"n": int(len(y)), "base_accuracy": float(y.mean())}

        print("  [1/6] backend comparison ...")
        d["backends"] = {str(a): backend_bench(conf, y, a) for a in ALPHAS}
        print("  [2/6] score certifiability ...")
        d["score_certifiability"] = score_certifiability(df, y)
        print("  [3/6] calibration size ...")
        d["calibration_size"] = calibration_size(df, y)
        print("  [4/6] sensitivity ...")
        d["sensitivity"] = sensitivity(df, y)
        print("  [5/6] error analysis + per-difficulty ...")
        d["error_analysis"] = error_analysis(df, y)
        d["per_difficulty"] = per_difficulty(df)
        print("  [6/6] economics ...")
        d["economics"] = economics(df, y)

        ALL["datasets"][name] = d
        with open(f"{OUT}/tables/backends_{name}.tex", "w") as f:
            f.write(tex_backend(name, d["backends"]))

        # summary lines
        summ.append(f"\n{name}  n={len(y)}  acc={y.mean():.3f}")
        for a in ["0.2", "0.25", "0.3"]:
            b = d["backends"][a]
            summ.append(f"  alpha={a}: " + " | ".join(
                f"{k} cov={b[k]['coverage'] and round(b[k]['coverage'],2)} "
                f"viol={None if b[k]['viol_rate'] is None else round(100*b[k]['viol_rate'])}%"
                for k in ["Heuristic", "CRC", "RCPS", "LtT"]))

    # distribution shift: Spider(gpt) -> BIRD(gpt) if both present
    src = next((n for n in data if "spider" in n and "gpt" in n), None)
    tgt = next((n for n in data if "bird" in n and "gpt" in n), None)
    if src and tgt:
        print(f"\n=== distribution shift: {src} -> {tgt} ===")
        ys = data[src]["label_correct"].to_numpy(int)
        yt = data[tgt]["label_correct"].to_numpy(int)
        ALL["distribution_shift"] = {"source": src, "target": tgt,
                                     "result": shift_test(data[src], ys, data[tgt], yt)}

    with open(f"{OUT}/all_results.json", "w") as f:
        json.dump(ALL, f, indent=2)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write("\n".join(summ))
    print(f"\nDONE. Wrote {OUT}/all_results.json, {OUT}/summary.txt, {OUT}/tables/*.tex")
    print("Now: zip the outputs/ folder and send it back.")


if __name__ == "__main__":
    main()
