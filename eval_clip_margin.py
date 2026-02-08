#!/usr/bin/env python3
"""
CLIP-margin evaluation for your analogy-generation folder structure.

WHAT THIS SCRIPT DOES
---------------------
It reads your generated images from:

  ROOT/
    category_1/
      pair_0001/
        meta.json
        GeneratedImages/
          seed001_A.png
          seed001_T.png
          seed001_alpha0.50_analogy.png
          seed001_alpha0.80_analogy.png
          seed001_alpha1.00_analogy.png
          seed001_grid.png
          ...

and computes (for each evaluated image x):

  sim(x, A) = cosine similarity between CLIP(image x) and CLIP(text prompt A)
  sim(x, T) = cosine similarity between CLIP(image x) and CLIP(text prompt T)
  margin    = sim(x, T) - sim(x, A)
  success   = 1 if margin > 0 else 0

Similarity used:
  - CLIP embeddings (image encoder + text encoder)
  - L2-normalized vectors
  - cosine similarity (dot product of normalized vectors)

IMPORTANT: Your seeds are *replicates*.
---------------------------------------
You have 3 seeds per wordpair. Those are NOT independent “data points”.
So for report-ready stats we aggregate correctly:

  Item unit = (pair, alpha)   # average over the 3 seeds first
  Then compute overall mean margin / success rate over those items.

OUTPUTS (saved under ROOT/eval/)
--------------------------------
1) per_image.csv
   - One row per evaluated *image file*.

2) summary_overall.json
   - Report-ready:
     mean_margin_per_item (+ bootstrap CI), success_rate_per_item (+ CI),
     plus runtime timing_seconds and throughput_images_per_second.

3) summary_by_pair_alpha.csv
   - One row per (pair, alpha), averaged over seeds.

4) summary_by_category_alpha.csv
   - One row per (category, alpha), averaged over items.

5) summary_best_alpha_per_pair.csv
   - One row per pair: chooses best alpha by mean margin over seeds.

6) summary_alpha_overall.csv
   - Alpha leaderboard across pairs: mean margin per alpha (items averaged over seeds).

PERFORMANCE / TIMING
--------------------
- GPU (RTX 4060) should handle a few thousand images comfortably.
- We measure elapsed time and throughput (images/sec) and store it.

DEPENDENCIES
------------
Recommended:
  pip install open_clip_torch torch torchvision pillow numpy scipy

Fallback (if you prefer HuggingFace CLIP instead of open_clip):
  pip install transformers torch pillow numpy scipy

By default we try open_clip first (faster / common), and fallback to HF CLIP.

USAGE
-----
  python eval_clip_margin.py --root /path/to/results --device cuda --batch_size 64

Optional: also evaluate A/T control images (seed###_A.png and seed###_T.png):
  python eval_clip_margin.py --root /path/to/results --include_controls

Notes:
- Grid images (*_grid.png) are ignored.
- Only .png files are scanned (easy to expand).
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# ------------------------------------------------------------
# 1) CLIP loading
# ------------------------------------------------------------
def load_clip(model_name: str, pretrained: str, device: str):
    """
    Try to load CLIP via open_clip first.
    If that fails, try HuggingFace transformers CLIP.

    Returns:
      backend: "open_clip" or "hf_clip"
      model: CLIP model instance
      preprocess_or_processor: image preprocessing function/object
      tokenizer: only for open_clip (HF uses processor for both)
    """
    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
        return "open_clip", model, preprocess, tokenizer

    except Exception as e1:
        # Fallback to HuggingFace CLIP
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            # NOTE: for HF, model_name must be an HF model id
            # e.g. "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(model_name).to(device).eval()
            processor = CLIPProcessor.from_pretrained(model_name)
            return "hf_clip", model, processor, None

        except Exception as e2:
            raise RuntimeError(
                "Failed to load CLIP via open_clip and transformers.\n"
                f"open_clip error: {e1}\n"
                f"transformers error: {e2}\n"
                "Install one of: open_clip_torch or transformers."
            )


# ------------------------------------------------------------
# 2) Basic math helpers
# ------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    In this script, a and b are already L2-normalized, so this is ~dot product.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def bootstrap_ci(values: np.ndarray, fn, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """
    Generic bootstrap confidence interval for a statistic fn(values).
    Returns estimate, and [ci_low, ci_high].
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None}

    stats = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        stats.append(fn(sample))

    stats = np.array(stats, dtype=float)
    est = fn(values)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {"estimate": float(est), "ci_low": lo, "ci_high": hi}


def one_sample_ttest_gt0(values: np.ndarray):
    """
    One-sample t-test:
      H0: mean(values) = 0
      H1: mean(values) > 0
    Only computed if scipy is installed.
    """
    try:
        from scipy import stats
    except ImportError:
        return {"t": None, "p_one_sided": None, "note": "scipy not installed"}

    if len(values) < 2:
        return {"t": None, "p_one_sided": None, "note": "n<2"}

    t, p = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
    return {"t": float(t), "p_one_sided": float(p)}


# ------------------------------------------------------------
# 3) Encoding functions (open_clip and HF)
# ------------------------------------------------------------
def encode_texts_open_clip(model, tokenizer, device: str, texts: List[str]) -> np.ndarray:
    """
    Returns normalized text embeddings (N x D) for open_clip.
    """
    import torch
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.detach().cpu().float().numpy()


def encode_images_open_clip(model, preprocess, device: str, image_paths: List[str]) -> np.ndarray:
    """
    Returns normalized image embeddings (N x D) for open_clip.
    """
    import torch
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))
    batch = torch.stack(imgs, dim=0).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.detach().cpu().float().numpy()


def encode_texts_hf_clip(model, processor, device: str, texts: List[str]) -> np.ndarray:
    """
    Returns normalized text embeddings (N x D) for HuggingFace CLIP.
    """
    import torch
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        feats = model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.detach().cpu().float().numpy()


def encode_images_hf_clip(model, processor, device: str, image_paths: List[str]) -> np.ndarray:
    """
    Returns normalized image embeddings (N x D) for HuggingFace CLIP.
    """
    import torch
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.detach().cpu().float().numpy()


# ------------------------------------------------------------
# 4) Your filename patterns
# ------------------------------------------------------------
# Analogy outputs:
#   seed001_alpha0.50_analogy.png
#   seed002_alpha1.00_analogy.png
# Also supported (if you ever omit "_analogy"):
#   seed001_alpha0.50.png
RE_ANALOGY = re.compile(
    r"^seed(?P<seed>\d+)_alpha(?P<alpha>\d+\.\d+)(?:_analogy)?\.png$",
    re.IGNORECASE
)

# Controls:
#   seed001_A.png
#   seed001_T.png
RE_CTRL = re.compile(
    r"^seed(?P<seed>\d+)_(?P<kind>[AT])\.png$",
    re.IGNORECASE
)


def classify_image(filename: str):
    """
    Decide whether this image should be evaluated, and extract seed/alpha/kind.

    Returns dict like:
      {"kind": "analogy", "seed": 1, "alpha": 0.5}
    or
      {"kind": "control_A", "seed": 1, "alpha": None}
    or None (ignore)
    """
    fn = filename.lower()

    # Ignore grid images (they are composites, not a single generation)
    if fn.endswith("_grid.png"):
        return None

    m = RE_ANALOGY.match(filename)
    if m:
        return {
            "kind": "analogy",
            "seed": int(m.group("seed")),
            "alpha": float(m.group("alpha")),
        }

    m = RE_CTRL.match(filename)
    if m:
        return {
            "kind": f"control_{m.group('kind').upper()}",
            "seed": int(m.group("seed")),
            "alpha": None,
        }

    return None


# ------------------------------------------------------------
# 5) File IO helpers
# ------------------------------------------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_pairs(root: str) -> List[Tuple[str, str]]:
    """
    Finds all (category_folder, pair_folder) under root.
    Example: ("category_1", "pair_0001")
    """
    pairs = []
    for cat in sorted(os.listdir(root)):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path):
            continue
        for pair in sorted(os.listdir(cat_path)):
            pair_path = os.path.join(cat_path, pair)
            if os.path.isdir(pair_path):
                pairs.append((cat, pair))
    return pairs


def read_meta(pair_path: str) -> dict:
    """
    Reads pair_path/meta.json.
    Must contain keys:
      - A (source prompt)
      - T (target prompt)
    """
    meta_path = os.path.join(pair_path, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[dict]):
    """
    Write list-of-dicts to CSV.
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------
# 6) Main evaluation
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to results/ folder")
    ap.add_argument("--eval_dir", default=None, help="Default: ROOT/eval")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)

    # CLIP config (open_clip default)
    ap.add_argument("--model", default="ViT-B-32",
                    help="open_clip model name OR HF model id (if using HF fallback)")
    ap.add_argument("--pretrained", default="openai",
                    help="open_clip pretrained tag (ignored for HF)")

    # Whether to also evaluate controls seedXXX_A.png and seedXXX_T.png
    ap.add_argument("--include_controls", action="store_true",
                    help="Also evaluate seedXXX_A.png and seedXXX_T.png")

    args = ap.parse_args()

    # Start timing (for report + sanity)
    t0 = time.perf_counter()

    root = os.path.abspath(args.root)
    eval_dir = os.path.abspath(args.eval_dir) if args.eval_dir else os.path.join(root, "eval")
    ensure_dir(eval_dir)

    # Load CLIP model
    backend, model, preproc_or_proc, tokenizer = load_clip(args.model, args.pretrained, args.device)

    # Cache text embeddings so we don't recompute prompts constantly.
    # Keys are prompt strings; values are CLIP embeddings (numpy arrays).
    text_cache: Dict[str, np.ndarray] = {}

    def get_text_emb(prompt: str) -> np.ndarray:
        """
        Encode text prompt once and reuse.
        """
        if prompt in text_cache:
            return text_cache[prompt]
        if backend == "open_clip":
            emb = encode_texts_open_clip(model, tokenizer, args.device, [prompt])[0]
        else:
            emb = encode_texts_hf_clip(model, preproc_or_proc, args.device, [prompt])[0]
        text_cache[prompt] = emb
        return emb

    per_image: List[dict] = []
    bs = max(1, args.batch_size)

    # Iterate all (category, pair) folders
    for cat, pair in list_pairs(root):
        pair_path = os.path.join(root, cat, pair)
        gen_dir = os.path.join(pair_path, "GeneratedImages")
        if not os.path.isdir(gen_dir):
            continue  # skip if missing GeneratedImages

        # Read prompts from meta.json
        meta = read_meta(pair_path)
        A = meta.get("A")  # source prompt
        T = meta.get("T")  # target prompt
        if not A or not T:
            raise ValueError(f"{pair_path}/meta.json must contain keys 'A' and 'T'.")

        # Precompute text embeddings for this pair
        A_emb = get_text_emb(A)
        T_emb = get_text_emb(T)

        # Collect all png images in GeneratedImages/
        candidates = []
        for img_path in sorted(glob.glob(os.path.join(gen_dir, "*.png"))):
            info = classify_image(os.path.basename(img_path))
            if info is None:
                continue

            # Only include control images if user requested them
            if (not args.include_controls) and info["kind"].startswith("control_"):
                continue

            candidates.append((img_path, info))

        if not candidates:
            continue

        # Encode images in batches for speed on GPU
        for start in range(0, len(candidates), bs):
            batch = candidates[start:start + bs]
            paths = [x[0] for x in batch]

            if backend == "open_clip":
                img_embs = encode_images_open_clip(model, preproc_or_proc, args.device, paths)
            else:
                img_embs = encode_images_hf_clip(model, preproc_or_proc, args.device, paths)

            # Compute margin for each image
            for (img_path, info), img_emb in zip(batch, img_embs):
                # cosine similarity between normalized embeddings
                sA = cosine_sim(img_emb, A_emb)
                sT = cosine_sim(img_emb, T_emb)

                margin = sT - sA
                success = int(margin > 0)

                # "per_image" means: one row per image file
                per_image.append({
                    # folder identifiers
                    "category": cat,              # e.g. category_1
                    "pair": pair,                  # e.g. pair_0001

                    # meta identifiers (optional, but useful for traceability)
                    "category_id": meta.get("category_id"),
                    "pair_id": meta.get("pair_id"),
                    "model_id": meta.get("model_id"),

                    # file identifiers
                    "image_path": os.path.relpath(img_path, root),
                    "image_name": os.path.basename(img_path),

                    # parsed generation params
                    "kind": info["kind"],          # analogy / control_A / control_T
                    "seed": info["seed"],
                    "alpha": info["alpha"],        # float for analogy, None for controls

                    # prompts used for scoring
                    "A": A,
                    "T": T,

                    # raw similarities + margin
                    "sim_A": sA,
                    "sim_T": sT,
                    "margin": margin,
                    "success": success,
                })

    if not per_image:
        raise SystemExit("No evaluable images found. Check layout + filenames.")

    # Write raw log (one row per image)
    per_image_csv = os.path.join(eval_dir, "per_image.csv")
    write_csv(per_image_csv, per_image)

    # For report, focus on analogy images only
    analogy_rows = [r for r in per_image if r["kind"] == "analogy"]

    # ------------------------------------------------------------
    # Correct aggregation with seeds:
    #   Unit of analysis = (pair, alpha)
    #   First average over seeds, then compute stats over those items.
    # ------------------------------------------------------------
    # Group margins/successes by item key: (category, pair, alpha)
    item_margins_map: Dict[Tuple[str, str, float], List[float]] = {}
    item_success_map: Dict[Tuple[str, str, float], List[float]] = {}

    for r in analogy_rows:
        k = (r["category"], r["pair"], float(r["alpha"]))
        item_margins_map.setdefault(k, []).append(float(r["margin"]))
        item_success_map.setdefault(k, []).append(float(r["success"]))

    # Convert each item to a single number by averaging over seeds
    item_margins = np.array([np.mean(v) for v in item_margins_map.values()], dtype=float)
    item_success_rate = np.array([np.mean(v) for v in item_success_map.values()], dtype=float)

    # Also keep per-image stats (sometimes useful, but NOT the main stats)
    per_image_margins = np.array([r["margin"] for r in analogy_rows], dtype=float)
    per_image_success = np.array([r["success"] for r in analogy_rows], dtype=float)

    overall = {
        "root": root,
        "clip_backend_used": backend,
        "clip_model": args.model,
        "pretrained": args.pretrained if backend == "open_clip" else None,

        # counts
        "num_analogy_images": int(len(analogy_rows)),
        "num_items_(pair,alpha)": int(len(item_margins)),

        # per-image (not the main claim)
        "mean_margin_per_image": float(np.mean(per_image_margins)),
        "success_rate_per_image": float(np.mean(per_image_success)),

        # per-item (MAIN claim, report this)
        "mean_margin_per_item": float(np.mean(item_margins)),
        "success_rate_per_item": float(np.mean(item_success_rate)),

        # uncertainty (bootstrap over items)
        "mean_margin_per_item_ci": bootstrap_ci(
            item_margins, fn=lambda x: float(np.mean(x)),
            n_boot=args.n_boot, seed=args.seed
        ),
        "success_rate_per_item_ci": bootstrap_ci(
            item_success_rate, fn=lambda x: float(np.mean(x)),
            n_boot=args.n_boot, seed=args.seed
        ),

        # optional significance test (if scipy installed)
        "one_sample_ttest_mean_margin_per_item_gt0": one_sample_ttest_gt0(item_margins),
    }

    # ------------------------------------------------------------
    # summary_by_pair_alpha.csv
    # One row per (category, pair, alpha), averaged over seeds.
    # ------------------------------------------------------------
    by_pair_alpha = []
    for (cat, pair, alpha), margins_list in sorted(item_margins_map.items()):
        succ_list = item_success_map[(cat, pair, alpha)]
        by_pair_alpha.append({
            "category": cat,
            "pair": pair,
            "alpha": alpha,
            "n_seeds": len(margins_list),
            "mean_margin_over_seeds": float(np.mean(margins_list)),
            "success_rate_over_seeds": float(np.mean(succ_list)),
        })
    write_csv(os.path.join(eval_dir, "summary_by_pair_alpha.csv"), by_pair_alpha)

    # ------------------------------------------------------------
    # summary_by_category_alpha.csv
    # One row per (category, alpha), averaged over items (pairs).
    # Here each item is already seed-averaged.
    # ------------------------------------------------------------
    cat_alpha_map: Dict[Tuple[str, float], List[float]] = {}
    cat_alpha_succ: Dict[Tuple[str, float], List[float]] = {}

    for (cat, pair, alpha), margins_list in item_margins_map.items():
        k = (cat, alpha)
        cat_alpha_map.setdefault(k, []).append(float(np.mean(margins_list)))
        cat_alpha_succ.setdefault(k, []).append(float(np.mean(item_success_map[(cat, pair, alpha)])))

    by_category_alpha = []
    for (cat, alpha), vals in sorted(cat_alpha_map.items()):
        by_category_alpha.append({
            "category": cat,
            "alpha": alpha,
            "n_items": len(vals),
            "mean_margin": float(np.mean(vals)),
            "median_margin": float(np.median(vals)),
            "mean_success_rate": float(np.mean(cat_alpha_succ[(cat, alpha)])),
        })
    write_csv(os.path.join(eval_dir, "summary_by_category_alpha.csv"), by_category_alpha)

    # ------------------------------------------------------------
    # summary_best_alpha_per_pair.csv
    # For each pair, choose alpha that maximizes mean margin (over seeds).
    # This directly answers: "which alpha works best per pair?"
    # ------------------------------------------------------------
    per_pair_alpha: Dict[Tuple[str, str], Dict[float, Tuple[float, float]]] = {}
    for row in by_pair_alpha:
        k = (row["category"], row["pair"])
        per_pair_alpha.setdefault(k, {})[float(row["alpha"])] = (
            float(row["mean_margin_over_seeds"]),
            float(row["success_rate_over_seeds"]),
        )

    all_alphas = sorted({float(r["alpha"]) for r in by_pair_alpha})

    best_rows = []
    for (cat, pair), a2stats in sorted(per_pair_alpha.items()):
        best_alpha = max(a2stats.keys(), key=lambda a: a2stats[a][0])  # argmax mean margin
        best_margin, best_succ = a2stats[best_alpha]

        out = {
            "category": cat,
            "pair": pair,
            "best_alpha": best_alpha,
            "best_mean_margin": best_margin,
            "best_success_rate": best_succ,
        }

        # Include each alpha's mean margin (helpful for later analysis / tables)
        for a in all_alphas:
            mm = a2stats.get(a, (None, None))[0]
            out[f"mean_margin_alpha_{a:.2f}"] = mm

        best_rows.append(out)

    write_csv(os.path.join(eval_dir, "summary_best_alpha_per_pair.csv"), best_rows)

    # ------------------------------------------------------------
    # summary_alpha_overall.csv
    # Alpha leaderboard across pairs:
    # For each alpha, average the seed-averaged (pair,alpha) margins across pairs.
    # ------------------------------------------------------------
    alpha_rows = []
    for a in all_alphas:
        vals = []
        succs = []
        for (cat, pair), a2stats in per_pair_alpha.items():
            if a in a2stats:
                vals.append(a2stats[a][0])   # mean margin over seeds for that pair,alpha
                succs.append(a2stats[a][1])  # success rate over seeds for that pair,alpha

        vals = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        succs = np.array(succs, dtype=float) if succs else np.array([], dtype=float)

        alpha_rows.append({
            "alpha": a,
            "n_items": int(len(vals)),
            "mean_margin": float(np.mean(vals)) if len(vals) else None,
            "median_margin": float(np.median(vals)) if len(vals) else None,
            "mean_success_rate": float(np.mean(succs)) if len(succs) else None,
        })

    write_csv(os.path.join(eval_dir, "summary_alpha_overall.csv"), alpha_rows)

    # ------------------------------------------------------------
    # Timing: save runtime and throughput for your report
    # ------------------------------------------------------------
    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    overall["timing_seconds"] = float(elapsed_s)
    overall["throughput_images_per_second"] = float(len(per_image) / elapsed_s) if elapsed_s > 0 else None
    overall["num_images_scored_total_including_controls"] = int(len(per_image))

    # Write summary_overall.json (final, including timing + all computed stats)
    with open(os.path.join(eval_dir, "summary_overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(f"[OK] wrote {per_image_csv} and summaries to {eval_dir}")
    print(f"[TIME] {elapsed_s:.2f}s total, {overall['throughput_images_per_second']:.2f} images/s")


if __name__ == "__main__":
    main()
