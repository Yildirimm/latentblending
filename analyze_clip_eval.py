#!/usr/bin/env python3
"""
Make sense of CLIP-margin evaluation outputs.

Reads your evaluation artifacts from: ROOT/eval/
  - per_image.csv
  - summary_by_pair_alpha.csv
  - summary_by_category_alpha.csv
  - summary_best_alpha_per_pair.csv
  - summary_alpha_overall.csv
  - summary_overall.json

Writes report-friendly summaries to: ROOT/eval/report/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to results/ folder (the same --root you used before)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    eval_dir = root / "eval"
    out_dir = eval_dir / "report"
    ensure_dir(out_dir)

    # -----------------------------
    # Load files (fail fast if missing)
    # -----------------------------
    per_image_path = eval_dir / "per_image.csv"
    by_pair_alpha_path = eval_dir / "summary_by_pair_alpha.csv"
    by_cat_alpha_path = eval_dir / "summary_by_category_alpha.csv"
    best_alpha_path = eval_dir / "summary_best_alpha_per_pair.csv"
    alpha_overall_path = eval_dir / "summary_alpha_overall.csv"
    overall_json_path = eval_dir / "summary_overall.json"

    for p in [per_image_path, by_pair_alpha_path, by_cat_alpha_path, best_alpha_path, alpha_overall_path, overall_json_path]:
        if not p.exists():
            raise FileKeyError(f"Missing required file: {p}")

    per_image = pd.read_csv(per_image_path)
    by_pair_alpha = pd.read_csv(by_pair_alpha_path)
    by_cat_alpha = pd.read_csv(by_cat_alpha_path)
    best_alpha = pd.read_csv(best_alpha_path)
    alpha_overall = pd.read_csv(alpha_overall_path)
    with open(overall_json_path, "r", encoding="utf-8") as f:
        overall = json.load(f)

    # -----------------------------
    # Quick sanity prints (for you)
    # -----------------------------
    print("\n=== Overall (from summary_overall.json) ===")
    print(f"Num analogy images: {overall['num_analogy_images']}")
    print(f"Num items (pair,alpha): {overall['num_items_(pair,alpha)']}")
    print(f"Mean margin per item: {overall['mean_margin_per_item']:.6f} "
          f"[{overall['mean_margin_per_item_ci']['ci_low']:.6f}, {overall['mean_margin_per_item_ci']['ci_high']:.6f}]")
    print(f"Success rate per item: {overall['success_rate_per_item']:.3f} "
          f"[{overall['success_rate_per_item_ci']['ci_low']:.3f}, {overall['success_rate_per_item_ci']['ci_high']:.3f}]")
    print(f"Timing: {overall['timing_seconds']:.2f}s, throughput: {overall['throughput_images_per_second']:.2f} img/s")

    # -----------------------------
    # 1) Alpha win-rate (how often each alpha is best per pair)
    # -----------------------------
    winrate = (
        best_alpha.groupby("best_alpha")
        .size()
        .reset_index(name="num_pairs")
        .sort_values("num_pairs", ascending=False)
    )
    winrate["fraction"] = winrate["num_pairs"] / winrate["num_pairs"].sum()
    winrate.to_csv(out_dir / "alpha_winrate.csv", index=False)

    # -----------------------------
    # 2) Paired alpha deltas per pair (this is VERY report-friendly)
    #    Compute paired differences using seed-averaged margins per (pair,alpha).
    # -----------------------------
    # Pivot: rows = (category,pair), columns = alpha, values = mean_margin_over_seeds
    pivot = by_pair_alpha.pivot_table(
        index=["category", "pair"],
        columns="alpha",
        values="mean_margin_over_seeds",
        aggfunc="mean"
    )

    # Ensure consistent alpha columns if present
    # (Your typical is 0.5, 0.8, 1.0)
    deltas = []
    def add_delta(a_hi, a_lo, name):
        if a_hi in pivot.columns and a_lo in pivot.columns:
            d = (pivot[a_hi] - pivot[a_lo]).dropna()
            deltas.append({
                "comparison": name,
                "a_hi": a_hi,
                "a_lo": a_lo,
                "n_pairs": int(len(d)),
                "mean_delta": float(d.mean()),
                "median_delta": float(d.median()),
                "frac_positive": float((d > 0).mean())
            })

            # Save per-pair deltas too (for debugging / appendix)
            d_df = d.reset_index()
            d_df.rename(columns={0: "delta"}, inplace=True)
            d_df["comparison"] = name
            d_df.to_csv(out_dir / f"pairwise_delta_{name}.csv", index=False)

    add_delta(1.0, 0.8, "1.0_minus_0.8")
    add_delta(0.8, 0.5, "0.8_minus_0.5")
    add_delta(1.0, 0.5, "1.0_minus_0.5")

    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(out_dir / "alpha_pairwise_deltas.csv", index=False)

    # -----------------------------
    # 3) Category-level: "what works where?"
    #    For each category, choose alpha with best mean_margin (from by_cat_alpha)
    # -----------------------------
    # best alpha per category by mean_margin
    idx = by_cat_alpha.groupby("category")["mean_margin"].idxmax()
    cat_best = by_cat_alpha.loc[idx].copy()
    cat_best = cat_best.sort_values("mean_margin", ascending=False)
    cat_best.to_csv(out_dir / "category_overall.csv", index=False)

    # Also store a pivot table for easy “heatmap” in report tooling
    cat_alpha_pivot = by_cat_alpha.pivot_table(
        index="category", columns="alpha", values="mean_margin", aggfunc="mean"
    ).sort_index()
    cat_alpha_pivot.to_csv(out_dir / "category_alpha_margin_pivot.csv")

    cat_alpha_succ_pivot = by_cat_alpha.pivot_table(
        index="category", columns="alpha", values="mean_success_rate", aggfunc="mean"
    ).sort_index()
    cat_alpha_succ_pivot.to_csv(out_dir / "category_alpha_success_pivot.csv")

    # -----------------------------
    # 4) Top / bottom pairs by best alpha score
    # -----------------------------
    top20 = best_alpha.sort_values("best_mean_margin", ascending=False).head(20)
    bot20 = best_alpha.sort_values("best_mean_margin", ascending=True).head(20)
    top20.to_csv(out_dir / "top_20_pairs.csv", index=False)
    bot20.to_csv(out_dir / "bottom_20_pairs.csv", index=False)

    # -----------------------------
    # 5) Simple plots (matplotlib only)
    # -----------------------------
    # (a) margin distribution (per-image)
    margins = per_image.loc[per_image["kind"] == "analogy", "margin"].to_numpy(dtype=float)

    plt.figure()
    plt.hist(margins, bins=60)
    plt.title("CLIP margin distribution (per-image)")
    plt.xlabel("margin = sim(x,T) - sim(x,A)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_margin_distribution.png", dpi=200)
    plt.close()

    # (b) alpha overall mean margin (from summary_alpha_overall.csv)
    # Use the already seed-averaged item means.
    plt.figure()
    plt.plot(alpha_overall["alpha"], alpha_overall["mean_margin"], marker="o")
    plt.title("Mean margin by alpha (items averaged over seeds)")
    plt.xlabel("alpha")
    plt.ylabel("mean margin")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_alpha_overall.png", dpi=200)
    plt.close()

    # -----------------------------
    # Final message
    # -----------------------------
    print("\n[OK] Wrote report outputs to:", out_dir)
    print("Key files:")
    print(" - alpha_winrate.csv")
    print(" - alpha_pairwise_deltas.csv")
    print(" - category_overall.csv")
    print(" - top_20_pairs.csv / bottom_20_pairs.csv")
    print(" - plot_margin_distribution.png / plot_alpha_overall.png")


class FileKeyError(Exception):
    pass


if __name__ == "__main__":
    main()
