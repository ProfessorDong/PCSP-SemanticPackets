"""
Turn the sweep CSVs into ready-to-paste LaTeX table fragments
and a matplotlib figure for the paper.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"


def fmt_mean_std(s):
    m, sd = s.mean(), s.std()
    if sd < 1e-4:
        return f"{m:.3f}"
    return f"{m:.3f}\\,$\\pm$\\,{sd:.3f}"


def render_attack_table():
    df = pd.read_csv(RES / "sweep_attack.csv")
    df = df[df.N == 128]                 # headline table at N=128
    rows = []
    for ds in ["nslkdd", "cicids"]:
        for bk in ["ML-DSA-65", "SLH-DSA-SHA2-128s"]:
            for atk in ["A1_replay", "A2_model_sub",
                         "A3_evidence_mismatch", "A4_stale_context"]:
                sub = df[(df.dataset == ds)
                         & (df.backend == bk)
                         & (df.attack == atk)
                         & (df.policy == "pcsp")]
                far = sub.FAR.mean() if len(sub) else float("nan")
                rows.append((ds, bk, atk, far))
    print("=== Per-(dataset, backend) pcsp FAR at N=128 (mean over 5 seeds) ===")
    for r in rows:
        print(f"  {r[0]:8s} {r[1]:20s} {r[2]:22s} FAR={r[3]:.4f}")

    print()
    print("=== Per-policy FAR collapse across both datasets, both backends ===")
    for atk in ["A1_replay", "A2_model_sub",
                 "A3_evidence_mismatch", "A4_stale_context"]:
        for pol in ["raw", "semantic", "signed", "pcsp"]:
            sub = df[(df.attack == atk) & (df.policy == pol)]
            far = sub.FAR.mean() if len(sub) else float("nan")
            sd  = sub.FAR.std()  if len(sub) else float("nan")
            print(f"  {atk:22s} {pol:9s} FAR={far:.4f} (std={sd:.4f})")


def render_timing_table():
    df = pd.read_csv(RES / "sweep_timing.csv")
    print()
    print("=== Per-(dataset, backend) measured timings + bytes (N=128) ===")
    df128 = df[df.N == 128]
    grp = df128.groupby(["dataset", "backend"]).agg({
        "t_sign_ms":      ["mean", "std"],
        "t_verify_ms":    ["mean", "std"],
        "bytes_per_claim":["mean"],
        "sig_bytes":      ["first"],
    })
    for (ds, bk), row in grp.iterrows():
        print(f"  {ds:8s} {bk:20s} "
              f"sign={row['t_sign_ms']['mean']:6.2f}ms "
              f"verify={row['t_verify_ms']['mean']:6.3f}ms "
              f"B/claim={row['bytes_per_claim']['mean']:6.1f} "
              f"sig_B={row['sig_bytes']['first']}")


def render_n_sweep():
    df = pd.read_csv(RES / "sweep_bytes.csv")
    print()
    print("=== Measured bytes/claim vs N (mean over 5 seeds per cell) ===")
    pivot = df.groupby(["dataset", "backend", "N"])\
              .bytes_per_claim.mean().reset_index()
    for ds in ["nslkdd", "cicids"]:
        print(f"  --- {ds} ---")
        for bk in pivot[pivot.dataset == ds].backend.unique():
            sub = pivot[(pivot.dataset == ds) & (pivot.backend == bk)]
            row = "    " + f"{bk:20s}"
            for _, r in sub.iterrows():
                row += f"  N={int(r.N):3d}: {r.bytes_per_claim:6.1f}"
            print(row)


def render_table_I_latex():
    """Emit the LaTeX Table I body (just the row content)."""
    df = pd.read_csv(RES / "sweep_attack.csv")
    timing = pd.read_csv(RES / "sweep_timing.csv")
    df = df[df.N == 128]
    timing = timing[timing.N == 128]

    print()
    print("=== LaTeX Table I body (paste between \\midrule and \\bottomrule) ===")
    for ds_human, ds_key in [("NSL-KDD", "nslkdd"),
                              ("CIC-IDS-2017", "cicids")]:
        # Find PCSP rows in this dataset and ML-DSA backend
        for bk_human, bk_key in [
            ("ML-DSA-65", "ML-DSA-65"),
            ("SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128s"),
        ]:
            cell_far = {}
            for atk in ["A1_replay", "A2_model_sub",
                         "A3_evidence_mismatch", "A4_stale_context"]:
                sub = df[(df.dataset == ds_key)
                         & (df.backend == bk_key)
                         & (df.attack == atk)
                         & (df.policy == "pcsp")]
                cell_far[atk] = sub.FAR.mean()
            tsub = timing[(timing.dataset == ds_key)
                          & (timing.backend == bk_key)]
            bpc = tsub.bytes_per_claim.mean()
            tv  = tsub.t_verify_ms.mean()
            row = (f"{ds_human} + {bk_human:18s} & "
                   f"{cell_far['A1_replay']:.3f} & "
                   f"{cell_far['A2_model_sub']:.3f} & "
                   f"{cell_far['A3_evidence_mismatch']:.3f} & "
                   f"{cell_far['A4_stale_context']:.3f} & "
                   f"{bpc:.1f} & {tv:.2f} \\\\")
            print(row)


if __name__ == "__main__":
    render_attack_table()
    render_timing_table()
    render_n_sweep()
    render_table_I_latex()
