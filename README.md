# Certified Abstention for Text-to-SQL

**A distribution-free risk-control comparison for trustworthy natural-language database interfaces.**

Code and data for the paper *"Certified abstention for text-to-SQL: a distribution-free
risk-control comparison for trustworthy natural-language database interfaces"*
(Tabish Ahmad, Imtiaz Alam).

LLM-generated SQL fails *silently*: a wrong query parses, executes, and returns a
well-formed but incorrect table that no downstream check flags. The usual safeguard — answer
only when a confidence score clears a threshold tuned on a calibration set — ships with an
implicit promise that at most a fraction `α` of answered queries are wrong. **This repository
shows that promise has no statistical backing, and provides the machinery to replace it with
one that does.**

We frame selective answering as *distribution-free risk control* and compare four calibration
backends on identical data: heuristic thresholding, Conformal Risk Control (CRC),
Risk-Controlling Prediction Sets (RCPS), and Learn-then-Test (LtT). Everything here is
reproducible from the included per-query data with a single command.

---

## Headline result

On Spider / GPT-4o-mini (300 random calibration/deployment splits, `δ = 0.10`):

| Backend | Guarantee | Coverage @ α=0.25 | Per-deployment violation rate |
|---|---|---:|---:|
| Heuristic thresholding | none | 0.88 | **51.7%** |
| Conformal Risk Control (CRC) | marginal `E[R] ≤ α` | 0.88 | **47.3%** |
| Risk-Controlling Prediction Sets (RCPS) | high-prob. `Pr[R ≤ α] ≥ 1−δ` | 0.75 | **6.4%** |
| Learn-then-Test (LtT) | high-prob. (multiple testing) | 0.55 | **0.5%** |

The average-risk methods breach their promised error rate in roughly **half** of individual
deployments, even though their *mean* risk sits on target. Only the high-probability backends
keep the promise within the `δ = 10%` allowance. Certification, not calibration, is the right
primitive for trusting AI-generated SQL.

---

## Repository layout

```
.
├── baselines.py            # Heuristic, CRC, RCPS backends + Hoeffding–Bentkus upper bound
├── conformal.py            # Learn-then-Test (LtT) calibration via Bonferroni multiple testing
├── extract_features.py     # Build the 24-feature confidence vector from K sampled queries
├── run_experiments.py      # Full experiment suite -> outputs/all_results.json (+ LaTeX tables)
├── results/                # Per-query features + correctness labels (the experiment inputs)
│   ├── spider__gpt-4o-mini/dataset.csv
│   ├── bird__gpt-4o-mini/dataset.csv
│   ├── spider__llama-3.3-70b-versatile/dataset.csv
│   └── bird__llama-3.1-8b-instant/dataset.csv
├── outputs/
│   └── all_results.json    # Every number in the paper, machine-readable
├── figures/
│   ├── make_figures.py     # Regenerates figures 2–9 from outputs/all_results.json
│   └── make_architecture.py# Regenerates figure 1 (the framework diagram)
├── scripts/
│   ├── push.bat            # Commit & push to GitHub (Windows CMD)
│   └── push.sh             # Commit & push to GitHub (bash / git-bash)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_experiments.py          # -> outputs/all_results.json, summary.txt, tables/*.tex
python figures/make_figures.py     # -> figures/fig2..fig9 .png
python figures/make_architecture.py# -> figures/fig1_architecture.png
```

`run_experiments.py` auto-discovers every `results/<name>/dataset.csv`, runs all eight studies
(backend comparison, score certifiability, calibration-size, sensitivity to `δ` and grid size,
error analysis, per-difficulty breakdown, distribution-shift stress test, deployment economics),
and writes a single `all_results.json` plus ready-to-paste LaTeX tables. It is deterministic
(`seed = 0`), so the numbers reproduce exactly.

## The data

Each `results/<benchmark>__<model>/dataset.csv` has one row per evaluation question. The columns
are the **24 confidence features** (execution consistency, structural agreement, schema
grounding, and token-uncertainty signals computed from `K = 5` sampled queries — never from the
gold query), plus:

| Column | Meaning |
|---|---|
| `label_correct` | 1 if the chosen query's execution matches the gold query, else 0 |
| `example_id` | benchmark question id |
| `difficulty` | benchmark difficulty label where available (BIRD), else `unknown` |

Benchmarks are the public dev splits of [Spider](https://yale-lily.github.io/spider) and
[BIRD](https://bird-bench.github.io/); correctness is execution match under SQLite. Generators
are GPT-4o-mini (Spider `n=600`, BIRD `n=367`), Llama-3.3-70B (Spider `n=80`) and
Llama-3.1-8B (BIRD `n=82`).

## The four backends

Each backend maps a calibration split `(score, correct)`, a tolerance `α`, and (for the
high-probability methods) a confidence level `1−δ`, to a certified threshold `λ̂` — or to
**`None`**, an explicit refusal when no threshold is safe at the requested level.

- **`heuristic`** — lowest grid threshold whose *observed* error ≤ `α`. No guarantee; biased.
- **`crc`** — Conformal Risk Control (Angelopoulos et al., 2024). Controls the **marginal**
  risk `E[R] ≤ α`.
- **`rcps`** — Risk-Controlling Prediction Sets (Bates et al., 2021). A `(1−δ)`
  Hoeffding–Bentkus upper bound on the risk, exploiting monotonicity. **High-probability.**
- **`conformal.calibrate_threshold`** — Learn-then-Test (Angelopoulos et al., 2021). Treats each
  grid threshold as a hypothesis and controls the family-wise error rate by Bonferroni.
  **High-probability**, no monotonicity assumption.

## Reproducing the full feature pipeline

`extract_features.py` builds `dataset.csv` from raw model generations (it samples `K` candidate
queries per question, executes them, and computes the 24 features). Running it from scratch
requires API/model access and the benchmark databases; the already-extracted `results/*.csv`
are provided so every number and figure in the paper reproduces offline, without any model calls.

## Extending to more models or benchmarks

No code change is needed. Drop another `results/<name>/dataset.csv` with the same 24 feature
columns, a `label_correct` column (and optionally `difficulty`), and re-run
`python run_experiments.py`. The new cell is picked up automatically.

## Citation

If you use this code or data, please cite:

```bibtex
@article{ahmad_certified_abstention_text2sql,
  title   = {Certified abstention for text-to-SQL: a distribution-free risk-control
             comparison for trustworthy natural-language database interfaces},
  author  = {Ahmad, Tabish and Alam, Imtiaz},
  year    = {2026}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
