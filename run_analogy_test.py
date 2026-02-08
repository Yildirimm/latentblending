import os
import re
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from docx import Document
from diffusers import StableDiffusionXLPipeline


# -----------------------------
# Config
# -----------------------------
DOCX_PATH = "data/word_pairs.docx"  # your uploaded file
MODEL_ID = "segmind/SSD-1B"

OUT_ROOT = "results"  # will create results/category_<id>/pair_<id>/GeneratedImages

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

WIDTH = 768
HEIGHT = 768
STEPS = 25
GUIDANCE = 6.0

# For robust testing: sweep a few seeds and alphas
SEEDS = [1, 2, 3]
ALPHAS = [0.5, 0.8, 1.0]

NEGATIVE = "blurry, low quality, deformed, distorted"


# -----------------------------
# Parsing the DOCX
# -----------------------------
QUOTE_RE = r"[\"“”]"  # support "..." and “...”
PAIR_RE = re.compile(
    rf"{QUOTE_RE}(?P<A>.+?){QUOTE_RE}\s*[-−]\s*{QUOTE_RE}(?P<B>.+?){QUOTE_RE}\s*\+\s*{QUOTE_RE}(?P<C>.+?){QUOTE_RE}\s*[≈~]\s*{QUOTE_RE}(?P<T>.+?){QUOTE_RE}"
)
CLASS_RE = re.compile(r"Class\s+(?P<id>\d+)\s*—\s*(?P<name>.+)$")


def read_pairs_from_docx(docx_path: str) -> List[Dict[str, Any]]:
    doc = Document(docx_path)
    current_class_id = None
    current_class_name = None

    pairs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        m_class = CLASS_RE.match(text)
        if m_class:
            current_class_id = int(m_class.group("id"))
            current_class_name = m_class.group("name").strip()
            continue

        m_pair = PAIR_RE.search(text)
        if m_pair and current_class_id is not None:
            pairs.append(
                {
                    "category_id": current_class_id,
                    "category_name": current_class_name,
                    "A": m_pair.group("A").strip(),
                    "B": m_pair.group("B").strip(),
                    "C": m_pair.group("C").strip(),
                    "T": m_pair.group("T").strip(),
                    "raw": text,
                }
            )

    return pairs


# -----------------------------
# Image helpers
# -----------------------------
def make_grid(images: List[Image.Image], cols: int) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


# -----------------------------
# Diffusers encode/generate
# -----------------------------
@torch.inference_mode()
def encode(pipe, prompt: str, negative_prompt: str):
    # Works in diffusers 0.25.0 for SDXL pipelines
    pe, ne, ppe, nppe = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
    )
    return pe, ne, ppe, nppe


@torch.inference_mode()
def gen_text(pipe, prompt: str, negative: str, seed: int) -> Image.Image:
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=g,
    ).images[0]


@torch.inference_mode()
def gen_embeds(pipe, pe, ppe, ne, nppe, seed: int) -> Image.Image:
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    return pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=pe,
        pooled_prompt_embeds=ppe,
        negative_prompt_embeds=ne,
        negative_pooled_prompt_embeds=nppe,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=g,
    ).images[0]


def safe_name(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s[:maxlen].strip("_") or "item"


def main():
    pairs = read_pairs_from_docx(DOCX_PATH)
    if not pairs:
        raise RuntimeError("No pairs parsed. Check DOCX formatting / quotes and 'Class N — ...' headers.")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16",
    ).to(DEVICE)

    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # Process
    cat_counters = {}  # category_id -> running index (pair id)
    for item in pairs:
        cid = item["category_id"]
        cat_counters[cid] = cat_counters.get(cid, 0) + 1
        pair_id = cat_counters[cid]

        # Directory: /category_ID/wordPairID/GeneratedImages
        pair_dir = out_root / f"category_{cid}" / f"pair_{pair_id:04d}"
        img_dir = pair_dir / "GeneratedImages"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            **item,
            "pair_id": pair_id,
            "model_id": MODEL_ID,
            "width": WIDTH,
            "height": HEIGHT,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "negative": NEGATIVE,
            "seeds": SEEDS,
            "alphas": ALPHAS,
        }
        (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        A, B, C, T = item["A"], item["B"], item["C"], item["T"]

        # Encode embeddings (A, B, C)
        pe_a, ne_a, ppe_a, nppe_a = encode(pipe, A, NEGATIVE)
        pe_b, _,    ppe_b, _      = encode(pipe, B, NEGATIVE)
        pe_c, _,    ppe_c, _      = encode(pipe, C, NEGATIVE)

        # Generate baselines + analogy variations
        for seed in SEEDS:
            images_for_grid = []

            img_A = gen_text(pipe, A, NEGATIVE, seed)
            img_T = gen_text(pipe, T, NEGATIVE, seed)

            fn_A = img_dir / f"seed{seed:03d}_A.png"
            fn_T = img_dir / f"seed{seed:03d}_T.png"
            img_A.save(fn_A)
            img_T.save(fn_T)

            images_for_grid.extend([img_A, img_T])

            for alpha in ALPHAS:
                pe = pe_a + alpha * (pe_c - pe_b)
                ppe = ppe_a + alpha * (ppe_c - ppe_b)

                img_X = gen_embeds(pipe, pe, ppe, ne_a, nppe_a, seed)
                fn_X = img_dir / f"seed{seed:03d}_alpha{alpha:.2f}_analogy.png"
                img_X.save(fn_X)
                images_for_grid.append(img_X)

            # Save one grid per seed: [A | T | analogy(alpha...)]
            grid = make_grid(images_for_grid, cols=2 + len(ALPHAS))
            grid.save(img_dir / f"seed{seed:03d}_grid.png")

        print(f"[OK] category {cid} pair {pair_id:04d}: {A}  - {B} + {C}  ≈  {T}")

    print(f"\nDone. Results under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
