"""Check the shared phase-1 hand-overs against MANIFEST.json.

    python checkpoints/verify_checkpoints.py                   # file + parameter hashes
    python checkpoints/verify_checkpoints.py --errors          # + the hand-over error, re-evaluated
    python checkpoints/verify_checkpoints.py --errors --benchmarks burgers ks

Run from the repository root with `pipeline` importable (see README.md,
Installation). Every checkpoint is checked for its SHA-256, for the parameter
hash `phase1_ckpt_hash` (pipeline.train.params_hash, the hash every restore
re-checks) and its seed and first-order budget against the benchmark's
configs/base.py. `--errors` rebuilds each benchmark's model at the hand-over
(pipeline.train.build_from_checkpoint, one process per benchmark, run from its
directory) and re-evaluates the relative L2 error against the stored value;
CPU is enough (a few minutes in total).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ERR_RTOL = 1e-6        # the re-evaluated hand-over error against the recorded one


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(rows):
    from pipeline.train import load_phase1, params_hash
    bad = 0
    for r in rows:
        path = os.path.join(HERE, r["benchmark"], f"seed{r['seed']}", "phase1.pkl")
        ck = load_phase1(path)
        ok = (sha256(path) == r["sha256"] and params_hash(ck["params"]) == r["phase1_ckpt_hash"]
              == ck["phase1_ckpt_hash"] and int(ck["seed"]) == r["seed"]
              and int(ck["first_order_steps"]) == r["first_order_steps"])
        bad += not ok
        print(f"{r['benchmark']:16s} seed {r['seed']:2d}  {r['phase1_ckpt_hash']}  "
              f"{'ok' if ok else 'MISMATCH'}")
    return bad


def check_errors(benchmark, rows):
    """In benchmarks/<benchmark>/ (a child process): rebuild the model at every
    hand-over and re-evaluate its error."""
    os.environ["JAX_ENABLE_X64"] = "True"
    sys.path.insert(0, os.getcwd())
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    import pipeline.train as train
    import models
    from configs import base
    from utils import get_dataset
    classes = {"allen_cahn": ("AllenCahn", "AllenCahnEvaluator"),
               "burgers": ("Burgers", "BurgersEvaluator"),
               "ginzburg_landau": ("GinzburgLandau", "GinzburgLandauEvaluator"),
               "kdv": ("KDV", "KDVEvaluator"), "ks": ("KS", "KSEvaluator"),
               "rayleigh_taylor": ("NavierStokes", "NavierStokesEvaluator")}
    model_cls, evaluator_cls = (getattr(models, c) for c in classes[benchmark])
    bad = 0
    for r in (r for r in rows if r["benchmark"] == benchmark):
        config = base.get_config(seed=r["seed"])
        if int(config.optim.first_order_steps) != r["first_order_steps"]:
            print(f"{benchmark} seed {r['seed']}: configs/base.py asks for "
                  f"{config.optim.first_order_steps} first-order steps, the hand-over has "
                  f"{r['first_order_steps']}")
            bad += 1
            continue
        ck = train.load_phase1(os.path.join(HERE, benchmark, f"seed{r['seed']}", "phase1.pkl"))
        model, evaluator, _ = train.build_from_checkpoint(config, ck, get_dataset, model_cls,
                                                          evaluator_cls)
        err = train.l2re_of(evaluator, model.state.params)[0]
        rel = abs(err - r["handover_l2re"]) / r["handover_l2re"]
        ok = rel <= ERR_RTOL
        bad += not ok
        print(f"{benchmark:16s} seed {r['seed']:2d}  hand-over error {err:.6e} "
              f"(recorded {r['handover_l2re']:.6e}, rel. diff {rel:.1e})  {'ok' if ok else 'MISMATCH'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", action="store_true", help="re-evaluate the hand-over errors")
    ap.add_argument("--benchmarks", nargs="+", default=None)
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    rows = json.load(open(os.path.join(HERE, "MANIFEST.json")))["checkpoints"]
    if args._child:
        sys.exit(1 if check_errors(args._child, rows) else 0)
    benchmarks = args.benchmarks or list(dict.fromkeys(r["benchmark"] for r in rows))
    rows = [r for r in rows if r["benchmark"] in benchmarks]
    bad = check_hashes(rows)
    if args.errors:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            p for p in (ROOT, os.environ.get("PYTHONPATH")) if p)}
        for b in benchmarks:
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), "--_child", b],
                                 cwd=os.path.join(ROOT, "benchmarks", b), env=env)
            bad += rc != 0
    print("all checkpoints verified" if not bad else f"{bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
