"""Collect the second-order runs of the studies and the ablation into CSV tables.

Walks benchmarks/<benchmark>/<runs_dir>/<rule>/n<n>/seed<k>/summary.json (the
tree train.run_study writes; the phase-1 hand-overs under <runs_dir>/phase1/
are not runs and are skipped) and writes, into --out_dir (default
benchmarks/results/):

    runs.csv                one row per run (the columns of RUN_COLUMNS)
    sampler_comparison.csv  median / min / max final_l2re per benchmark x rule
                            at n = --n_comparison (default 2^14), with the
                            run statuses counted
    batch_size.csv          the same per benchmark x rule x n, for every rule
                            found at each n (the batch-size study's five rules
                            uniform, sobol, rad, lev, sled at every n; the
                            comparison-only rules at 2^14); the a1 rules
                            sled_fl<phi>, sled_kappa, sled_iid only go to a1.csv
    a1.csv                  the ablation a1 per benchmark x rule at n = --n_a1
                            (default 2^14): the fixed floors sled_fl<phi>,
                            sled_kappa and sled_iid next
                            to the references sobol (phi = 1) and sled
                            (phi = "rho", the measured one), with the median /
                            min / max measured rho (rho_measured of
                            sled_fl<phi>, rho_floor of sled; both the raw
                            Spearman before sled's clamp to [0, 1])
    supplementary.csv       the supplementary study per benchmark x rule at
                            n = --n_supp (default 2^14): its protocols
                            pinnacle_s, pinnacle_k_r500, rad_r500_reset,
                            vrba_r100_reset next to the comparators sled,
                            pinnacle_k, rad_r500, vrba_r100 (its larger
                            batches are rows of batch_size.csv)

Statistics are taken over the runs whose final_l2re is finite: an "ok" run and
a "truncated" one (the quasi-Newton phase stopped early; its last error is a
result) both count, a "diverged" or "failed" run has no final error and is only
counted in its status column.

All runs of one (benchmark, seed) should start from ONE phase-1 hand-over, or
the rules of that seed are not compared on equal terms. When their
phase1_ckpt_hash values differ (e.g. a tree merged from two machines, or a
checkpoint retrained after some runs), the script warns on stderr, sets
phase1_hash_mixed = 1 on every run of that (benchmark, seed) in runs.csv and
counts those runs per table row (n_phase1_mixed). Needs only the Python
standard library.

    python collect_results.py                              # benchmarks/*/runs
    python collect_results.py --benchmarks burgers ks
    python collect_results.py --runs_dir_name runs_smoke --n_comparison 2048 \\
        --out_dir results/smoke                            # the --smoke trees
    python collect_results.py --runs_dir_name runs_shared --out_dir results/shared
                                                           # runs from checkpoints/
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# display order; anything else follows alphabetically
BENCHMARKS = ["burgers", "allen_cahn", "kdv", "ks", "ginzburg_landau", "rayleigh_taylor"]
RULES = ["uniform", "sobol", "rad", "kd", "lev", "sled",
         "sled_fl0", "sled_fl0.25", "sled_fl0.5", "sled_fl0.75", "sled_kappa", "sled_iid",
         "uniform_r500", "rad_r500", "vrba_r100", "pinnacle_k",
         "pinnacle_s", "pinnacle_k_r500", "rad_r500_reset", "vrba_r100_reset"]
# ablation a1: the fixed floor mass phi of each rule; sobol is phi = 1, sled
# the measured rho
A1_PHI = {"sled_fl0": 0.0, "sled_fl0.25": 0.25, "sled_fl0.5": 0.5,
          "sled_fl0.75": 0.75, "sobol": 1.0, "sled": "rho",
          "sled_kappa": "rho", "sled_iid": "rho"}
A1_ONLY = [r for r in A1_PHI if r not in ("sobol", "sled")]  # not in the study tables
# supplementary study: its protocols and their comparators at n = --n_supp
SUPP_RULES = ["pinnacle_s", "pinnacle_k_r500", "rad_r500_reset", "vrba_r100_reset"]
SUPP_COMPARATORS = ["sled", "pinnacle_k", "rad_r500", "vrba_r100"]
STATUSES = ["ok", "truncated", "diverged", "failed"]

RUN_COLUMNS = ["benchmark", "rule", "n", "seed", "status", "steps_done",
               "handover_l2re", "final_l2re", "best_l2re", "final_gap",
               "final_loss_frozen", "final_loss_heldout", "floor", "rho_floor",
               "n_floor", "ess", "batch_distinct", "pool_size", "optimizer",
               "wall_clock_s", "phase1_ckpt_hash", "phase1_hash_mixed"]

RUN_DIR = re.compile(r"^n(\d+)$")
SEED_DIR = re.compile(r"^seed(-?\d+)$")


def _order(value, preferred):
    return (preferred.index(value), "") if value in preferred else (len(preferred), value)


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def find_runs(root, runs_dir_name, benchmarks=None):
    """Yield (benchmark, rule, n, seed, path to summary.json)."""
    pattern = os.path.join(root, "*", runs_dir_name, "*", "n*", "seed*", "summary.json")
    for path in sorted(glob.glob(pattern)):
        seed_dir = os.path.dirname(path)
        n_dir = os.path.dirname(seed_dir)
        rule_dir = os.path.dirname(n_dir)
        bench = os.path.basename(os.path.dirname(os.path.dirname(rule_dir)))
        m_n = RUN_DIR.match(os.path.basename(n_dir))
        m_s = SEED_DIR.match(os.path.basename(seed_dir))
        if not (m_n and m_s):
            continue
        if benchmarks and bench not in benchmarks:
            continue
        yield bench, os.path.basename(rule_dir), int(m_n.group(1)), int(m_s.group(1)), path


def read_row(bench, rule, n, seed, path):
    """One runs.csv row. The directory is the run's identity; a summary that
    names a different benchmark / rule / n / seed is reported, not trusted."""
    row = {c: None for c in RUN_COLUMNS}
    row.update(benchmark=bench, rule=rule, n=n, seed=seed)
    try:
        with open(path) as fh:
            s = json.load(fh)
    except Exception as exc:
        print(f"warning: cannot read {path} ({exc})", file=sys.stderr)
        row["status"] = "unreadable"
        return row
    for key, want in (("benchmark", bench), ("rule", rule), ("n", n), ("seed", seed)):
        if key in s and s[key] is not None and s[key] != want:
            print(f"warning: {path}: summary says {key}={s[key]!r}, the directory "
                  f"{want!r}", file=sys.stderr)
    for c in RUN_COLUMNS[4:]:
        row[c] = s.get(c)
    if row["phase1_ckpt_hash"] is None:
        # a run that failed before restoring records only the checkpoint it
        # was given
        row["phase1_ckpt_hash"] = s.get("phase1_ckpt_hash_expected")
    # the measured Spearman(lev, K) for the a1 table (not a runs.csv column):
    # sled_fl<phi> records it as rho_measured, sled as its floor rho_floor
    row["_rho"] = s.get("rho_measured")
    if row["_rho"] is None:
        row["_rho"] = s.get("rho_floor")
    return row


def mark_mixed_handovers(rows):
    """Set phase1_hash_mixed (0/1) on every row: 1 when the runs of its
    (benchmark, seed) carry more than one phase1_ckpt_hash. Warns on stderr
    for each such group and returns {(benchmark, seed): {hash: n_runs}}."""
    hashes = {}
    for r in rows:
        if r["phase1_ckpt_hash"] is not None:
            per = hashes.setdefault((r["benchmark"], r["seed"]), {})
            per[r["phase1_ckpt_hash"]] = per.get(r["phase1_ckpt_hash"], 0) + 1
    mixed = {k: v for k, v in hashes.items() if len(v) > 1}
    for r in rows:
        r["phase1_hash_mixed"] = int((r["benchmark"], r["seed"]) in mixed)
    for (bench, seed), per in sorted(mixed.items(), key=lambda kv: (
            _order(kv[0][0], BENCHMARKS), kv[0][1])):
        by_hash = ", ".join(f"{h} ({k} runs)" for h, k in sorted(
            per.items(), key=lambda hk: -hk[1]))
        print(f"warning: {bench} seed {seed}: its runs started from {len(per)} "
              f"different phase-1 checkpoints: {by_hash}; they are flagged "
              "phase1_hash_mixed = 1", file=sys.stderr)
    return mixed


def aggregate(rows, keys):
    """Group rows by `keys`: run and status counts plus median/min/max of
    final_l2re over the runs where it is finite."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in groups.items():
        vals = [r["final_l2re"] for r in rs if _finite(r["final_l2re"])]
        rec = dict(zip(keys, key))
        rec["n_runs"] = len(rs)
        for st in STATUSES:
            rec[f"n_{st}"] = sum(r["status"] == st for r in rs)
        other = len(rs) - sum(rec[f"n_{st}"] for st in STATUSES)
        rec["n_other"] = other
        rec["n_used"] = len(vals)
        rec["n_phase1_mixed"] = sum(bool(r["phase1_hash_mixed"]) for r in rs)
        rec["final_l2re_median"] = statistics.median(vals) if vals else None
        rec["final_l2re_min"] = min(vals) if vals else None
        rec["final_l2re_max"] = max(vals) if vals else None
        rec["seeds"] = " ".join(str(s) for s in sorted(r["seed"] for r in rs))
        rhos = [r["_rho"] for r in rs if _finite(r.get("_rho"))]
        rec["n_rho"] = len(rhos)
        rec["rho_median"] = statistics.median(rhos) if rhos else None
        rec["rho_min"] = min(rhos) if rhos else None
        rec["rho_max"] = max(rhos) if rhos else None
        out.append(rec)
    return sorted(out, key=lambda r: tuple(
        _order(r[k], BENCHMARKS) if k == "benchmark" else
        _order(r[k], RULES) if k == "rule" else (r[k],) for k in keys))


def _cell(v, precise=False):
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v) if precise else f"{v:.6g}"
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


def write_csv(path, rows, columns, precise=False):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell(r.get(c), precise) for c in columns])


def print_table(title, rows, keys):
    print(f"\n{title}")
    if not rows:
        print("  (no runs)")
        return
    head = keys + ["runs", "ok/trunc/div/fail", "median", "min", "max"]
    lines = [head]
    for r in rows:
        lines.append([str(r[k]) for k in keys] + [
            str(r["n_runs"]) + ("*" if r.get("n_phase1_mixed") else ""),
            "/".join(str(r[f"n_{s}"]) for s in STATUSES),
            *(f"{r[c]:.3e}" if r[c] is not None else "-"
              for c in ("final_l2re_median", "final_l2re_min", "final_l2re_max"))])
    widths = [max(len(l[i]) for l in lines) for i in range(len(head))]
    for l in lines:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(l, widths)))
    if any(r.get("n_phase1_mixed") for r in rows):
        print("  * some runs of this row started from a different phase-1 "
              "checkpoint than other runs of their seed (n_phase1_mixed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=HERE,
                    help="directory holding the benchmark folders (default: this one)")
    ap.add_argument("--runs_dir_name", default="runs",
                    help="run tree inside each benchmark (runs, or runs_smoke for --smoke)")
    ap.add_argument("--benchmarks", nargs="+", default=None,
                    help="restrict to these benchmark folders")
    ap.add_argument("--n_comparison", type=int, default=2**14,
                    help="batch size of the sampler comparison table (default 16384)")
    ap.add_argument("--n_a1", type=int, default=2**14,
                    help="batch size of the ablation a1 table (default 16384)")
    ap.add_argument("--n_supp", type=int, default=2**14,
                    help="batch size of the supplementary table (default 16384)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    rows = [read_row(*r) for r in find_runs(args.root, args.runs_dir_name, args.benchmarks)]
    mixed = mark_mixed_handovers(rows)
    rows.sort(key=lambda r: (_order(r["benchmark"], BENCHMARKS), _order(r["rule"], RULES),
                             r["n"], r["seed"]))
    os.makedirs(args.out_dir, exist_ok=True)
    agg_cols = ["n_runs"] + [f"n_{s}" for s in STATUSES] + [
        "n_other", "n_used", "n_phase1_mixed", "final_l2re_median", "final_l2re_min",
        "final_l2re_max", "seeds"]

    write_csv(os.path.join(args.out_dir, "runs.csv"), rows, RUN_COLUMNS, precise=True)
    comp = aggregate([r for r in rows if r["n"] == args.n_comparison
                      and r["rule"] not in A1_ONLY], ["benchmark", "rule", "n"])
    write_csv(os.path.join(args.out_dir, "sampler_comparison.csv"), comp,
              ["benchmark", "rule", "n"] + agg_cols)
    bs = aggregate([r for r in rows if r["rule"] not in A1_ONLY],
                   ["benchmark", "rule", "n"])
    write_csv(os.path.join(args.out_dir, "batch_size.csv"), bs,
              ["benchmark", "rule", "n"] + agg_cols)
    a1 = aggregate([r for r in rows if r["n"] == args.n_a1 and r["rule"] in A1_PHI],
                   ["benchmark", "rule", "n"])
    for rec in a1:
        rec["phi"] = A1_PHI[rec["rule"]]
    write_csv(os.path.join(args.out_dir, "a1.csv"), a1,
              ["benchmark", "rule", "n", "phi"] + agg_cols
              + ["n_rho", "rho_median", "rho_min", "rho_max"])
    supp = aggregate([r for r in rows if r["n"] == args.n_supp
                      and r["rule"] in SUPP_RULES + SUPP_COMPARATORS], ["benchmark", "rule", "n"])
    write_csv(os.path.join(args.out_dir, "supplementary.csv"), supp,
              ["benchmark", "rule", "n"] + agg_cols)

    counts = {st: sum(r["status"] == st for r in rows) for st in STATUSES}
    print(f"{len(rows)} runs under {args.root}/*/{args.runs_dir_name}/ "
          + " ".join(f"{k}={v}" for k, v in counts.items()))
    print_table(f"sampler comparison, n = {args.n_comparison} (final_l2re)", comp,
                ["benchmark", "rule"])
    if any(r["rule"] not in ("sled", "sobol") for r in a1):
        print_table(f"ablation a1, n = {args.n_a1} (final_l2re)", a1,
                    ["benchmark", "rule"])
    if any(r["rule"] in SUPP_RULES for r in supp):
        print_table(f"supplementary study, n = {args.n_supp} (final_l2re)", supp,
                    ["benchmark", "rule"])
    print(f"\nwrote runs.csv ({len(rows)} rows), sampler_comparison.csv ({len(comp)}), "
          f"batch_size.csv ({len(bs)}), a1.csv ({len(a1)}), supplementary.csv ({len(supp)}) "
          f"to {args.out_dir}")
    if mixed:
        n_mixed = sum(r["phase1_hash_mixed"] for r in rows)
        print(f"warning: {n_mixed} runs in {len(mixed)} (benchmark, seed) groups "
              "come from mixed phase-1 checkpoints (phase1_hash_mixed = 1 in "
              "runs.csv)", file=sys.stderr)


if __name__ == "__main__":
    main()
