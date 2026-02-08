import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# -----------------------------
# Config (tune these)
# -----------------------------
MODEL_ID = "segmind/SSD-1B"  # SSD-1B is SDXL-family (2 text encoders) :contentReference[oaicite:1]{index=1}
OUTDIR = "analogy_out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIDTH = 768
HEIGHT = 768
STEPS = 25
GUIDANCE = 6.0
SEED = 42

# Prompt template matters a lot.
# For "king-man+woman≈queen", a consistent template often helps.
TEMPLATE = "a high quality portrait photo of a {w}"
NEGATIVE = "low quality, blurry, distorted, deformed"

# Analogy terms
A = "king"
B = "man"
C = "woman"
TARGET_TEXT = "queen"

# Arithmetic scaling: E = E(A) + alpha*(E(C)-E(B))
ALPHA = 1.0


def make_grid(images, cols=3):
    """Simple image grid."""
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


@torch.inference_mode()
def encode(pipe, prompt: str, negative_prompt: str):
    """
    Returns:
      prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    StableDiffusionXLPipeline requires pooled embeds when passing prompt_embeds. :contentReference[oaicite:2]{index=2}
    """
    # Diffusers SDXL pipeline has encode_prompt() in recent versions.
    if hasattr(pipe, "encode_prompt"):
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

    # Fallback: if encode_prompt is missing (older diffusers), bail clearly.
    raise RuntimeError(
        "This diffusers version/pipeline doesn't expose encode_prompt(). "
        "Upgrade diffusers or switch to a pipeline version that supports SDXL encode_prompt."
    )


@torch.inference_mode()
def generate_from_text(pipe, prompt: str, negative: str, seed: int, name: str):
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=g,
    ).images[0]
    img.save(os.path.join(OUTDIR, f"{name}.png"))
    return img


@torch.inference_mode()
def generate_from_embeds(pipe, prompt_embeds, pooled_prompt_embeds,
                         negative_prompt_embeds, negative_pooled_prompt_embeds,
                         seed: int, name: str):
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    img = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=g,
    ).images[0]
    img.save(os.path.join(OUTDIR, f"{name}.png"))
    return img


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # Memory-friendly settings
    # For 4070 you can usually do pipe.to("cuda"), but offload is safer if you push resolution.
    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.vae.enable_tiling()

    # Baselines
    img_a = generate_from_text(pipe, TEMPLATE.format(w=A), NEGATIVE, SEED, "A_king_text")
    img_target = generate_from_text(pipe, TEMPLATE.format(w=TARGET_TEXT), NEGATIVE, SEED, "target_queen_text")

    # Encode embeddings for A,B,C
    pe_a, ne_a, ppe_a, nppe_a = encode(pipe, TEMPLATE.format(w=A), NEGATIVE)
    pe_b, ne_b, ppe_b, nppe_b = encode(pipe, TEMPLATE.format(w=B), NEGATIVE)
    pe_c, ne_c, ppe_c, nppe_c = encode(pipe, TEMPLATE.format(w=C), NEGATIVE)

    # Arithmetic (prompt embeds are token-wise hidden states; pooled embeds are global).
    # We'll do the same arithmetic on both.
    pe_analogy = pe_a + ALPHA * (pe_c - pe_b)
    ppe_analogy = ppe_a + ALPHA * (ppe_c - ppe_b)

    # For negatives: simplest is "use A's negative", but you can also do arithmetic.
    ne = ne_a
    nppe = nppe_a

    img_analogy = generate_from_embeds(
        pipe,
        prompt_embeds=pe_analogy,
        pooled_prompt_embeds=ppe_analogy,
        negative_prompt_embeds=ne,
        negative_pooled_prompt_embeds=nppe,
        seed=SEED,
        name="analogy_king_minus_man_plus_woman",
    )

    grid = make_grid([img_a, img_analogy, img_target], cols=3)
    grid.save(os.path.join(OUTDIR, "grid.png"))

    print("Saved:")
    print(" -", os.path.join(OUTDIR, "A_king_text.png"))
    print(" -", os.path.join(OUTDIR, "analogy_king_minus_man_plus_woman.png"))
    print(" -", os.path.join(OUTDIR, "target_queen_text.png"))
    print(" -", os.path.join(OUTDIR, "grid.png"))


if __name__ == "__main__":
    main()
