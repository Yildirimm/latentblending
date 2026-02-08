import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import gc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import gradio as gr
import uuid
from diffusers import AutoPipelineForText2Image
from blending_engine import BlendingEngine
import datetime
import tempfile
import json
from lunar_tools import concatenate_movies
import argparse


# =========================
# MultiUserRouter
# =========================

class MultiUserRouter:
    def __init__(self, do_compile=False):
        self.do_compile = do_compile

        self.list_models = [
            "segmind/SSD-1B",             # Distilled SDXL (Recommended)
            "stabilityai/sdxl-turbo",     # Fast SDXL
            "runwayml/stable-diffusion-v1-5" # Original small model
        ]

        self.user_blendingvariableholder = {}

        # single-GPU state
        self.current_pipe = None
        self.current_model = None
        self.current_be = None

    # ---------- GPU model management ----------
    def load_model(self, model):
        if self.current_model == model:
            return self.current_be

        # unload previous model
        if self.current_pipe is not None:
            del self.current_pipe
            del self.current_be
            self.current_pipe = None
            self.current_be = None
            gc.collect()           # Force Python to clear RAM
            torch.cuda.empty_cache() # Force GPU to clear VRAM
            
        pipe = AutoPipelineForText2Image.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16", # Use the smaller fp16 weights
            use_safetensors=True
        )

        # Optimization: Move model parts to CPU when not in use
        # This is the "OOM Killer" fix. 
        # Do NOT use pipe.to("cuda") if you use this.

        # REPLACE enable_model_cpu_offload() with this:
        pipe.enable_sequential_cpu_offload()
        
        pipe.enable_attention_slicing()
        
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
            
        pipe.vae.enable_tiling()

        be = BlendingEngine(pipe, do_compile=self.do_compile)

        self.current_pipe = pipe
        self.current_model = model
        self.current_be = be

        return be

    # ---------- helpers ----------
    def _get_holder_and_be(self, user_id):
        holder = self.user_blendingvariableholder[user_id]
        be = self.load_model(holder.model)
        holder.be = be
        return holder

    # ---------- user ----------
    def register_new_user(self, model, width, height):
        user_id = uuid.uuid4().hex[:8].upper()

        be = self.load_model(model)
        be.set_dimensions((width, height))

        holder = BlendingVariableHolder(model)
        holder.be = be
        self.user_blendingvariableholder[user_id] = holder

        return user_id

    # ---------- UI callbacks ----------
    def preview_img_selected(self, data, user_id):
        self.user_blendingvariableholder[user_id].preview_img_selected(data)
        return None

    def movie_img_selected(self, data, user_id):
        self.user_blendingvariableholder[user_id].movie_img_selected(data)
        return None


    def compute_imgs(self, user_id, prompt, negative_prompt):
        holder = self._get_holder_and_be(user_id)
        return holder.compute_imgs(prompt, negative_prompt)

    def add_image_to_video(self, user_id):
        holder = self._get_holder_and_be(user_id)
        return holder.add_image_to_video()

    def img_movie_delete(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_delete()

    def img_movie_later(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_later()

    def img_movie_earlier(self, user_id):
        return self.user_blendingvariableholder[user_id].img_movie_earlier()

    def generate_movie(self, user_id, t_per_segment):
        holder = self._get_holder_and_be(user_id)
        return holder.generate_movie(t_per_segment)


# =========================
# BlendingVariableHolder
# =========================

class BlendingVariableHolder:
    def __init__(self, model):
        self.model = model
        self.be = None

        self.prompt = None
        self.negative_prompt = None

        self.nmb_preview_images = 4
        self.list_seeds = []
        self.list_images_preview = []

        self.idx_img_preview_selected = None
        self.idx_img_movie_selected = None

        self.data = []
        self.idx_movie = 0

        self.jpg_quality = 80
        self.fp_movie = ""
        self.fp_json = ""

    # ---------- selection ----------
    def preview_img_selected(self, data):
        self.idx_img_preview_selected = data.index

    def movie_img_selected(self, data):
        self.idx_img_movie_selected = data.index

    # ---------- generation ----------
    def compute_imgs(self, prompt, negative_prompt):
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.be.set_prompt1(prompt)
        self.be.set_prompt2(prompt)
        self.be.set_negative_prompt(negative_prompt)

        self.list_seeds.clear()
        self.list_images_preview.clear()
        self.idx_img_preview_selected = None

        for _ in range(self.nmb_preview_images):
            seed = np.random.randint(0, np.iinfo(np.int32).max)
            self.be.seed1 = seed
            self.list_seeds.append(seed)

            img = self.be.compute_latents1(return_image=True)

            fn = f"image_{uuid.uuid4().hex}.jpg"
            path = os.path.join(tempfile.gettempdir(), fn)
            img.save(path, quality=self.jpg_quality, optimize=True)

            self.list_images_preview.append(path)

        return self.list_images_preview

    # ---------- movie ----------
    def init_new_movie(self):
        t = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.fp_movie = f"movie_{t}.mp4"
        self.fp_json = f"movie_{t}.json"

    def write_json(self):
        data_copy = self.data.copy()
        data_copy.insert(
            0,
            {
                "settings": "sdxl",
                "width": self.be.dh.width_img,
                "height": self.be.dh.height_img,
                "num_inference_steps": self.be.dh.num_inference_steps,
            },
        )
        with open(self.fp_json, "w") as f:
            json.dump(data_copy, f, indent=2)

    def add_image_to_video(self):
        if self.prompt is None or self.idx_img_preview_selected is None:
            return self.get_list_images_movie()

        if self.idx_movie == 0:
            self.init_new_movie()

        self.data.append(
            {
                "iteration": self.idx_movie,
                "seed": self.list_seeds[self.idx_img_preview_selected],
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "preview_image": self.list_images_preview[self.idx_img_preview_selected],
            }
        )

        self.write_json()
        self.idx_movie += 1
        return self.get_list_images_movie()

    def get_list_images_movie(self):
        return [d["preview_image"] for d in self.data]

    def img_movie_delete(self):
        if (
            self.idx_img_movie_selected is not None
            and 0 <= self.idx_img_movie_selected < len(self.data)
        ):
            del self.data[self.idx_img_movie_selected]
        self.idx_img_movie_selected = None
        return self.get_list_images_movie()

    def img_movie_later(self):
        i = self.idx_img_movie_selected
        if i is not None and i + 1 < len(self.data):
            self.data[i], self.data[i + 1] = self.data[i + 1], self.data[i]
        self.idx_img_movie_selected = None
        return self.get_list_images_movie()

    def img_movie_earlier(self):
        i = self.idx_img_movie_selected
        if i is not None and i > 0:
            self.data[i], self.data[i - 1] = self.data[i - 1], self.data[i]
        self.idx_img_movie_selected = None
        return self.get_list_images_movie()

    def generate_movie(self, t_per_segment=10):
        prompts = [d["prompt"] for d in self.data]
        negs = [d["negative_prompt"] for d in self.data]
        seeds = [d["seed"] for d in self.data]

        parts = []

        for i in range(len(prompts) - 1):
            if i == 0:
                self.be.set_prompt1(prompts[i])
                self.be.set_negative_prompt(negs[i])
                self.be.set_prompt2(prompts[i + 1])
                recycle = False
            else:
                self.be.swap_forward()
                self.be.set_negative_prompt(negs[i + 1])
                self.be.set_prompt2(prompts[i + 1])
                recycle = True

            fp = f"tmp_part_{i:03d}.mp4"
            self.be.run_transition(
                recycle_img1=recycle,
                fixed_seeds=seeds[i : i + 2],
            )
            self.be.write_movie_transition(fp, t_per_segment)
            parts.append(fp)

        concatenate_movies(self.fp_movie, parts)
        return self.fp_movie


# =========================
# Runtime
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_compile", action="store_true")
    parser.add_argument("--server_name", type=str, default=None)
    args = parser.parse_args()

    mur = MultiUserRouter(do_compile=args.do_compile)

    with gr.Blocks() as demo:
        with gr.Accordion("Setup", open=True):
            model = gr.Dropdown(mur.list_models, value=mur.list_models[0])
            width = gr.Slider(256, 2048, 512, step=128)
            height = gr.Slider(256, 2048, 512, step=128)
            user_id = gr.Textbox(interactive=False)
            b_start = gr.Button("start session", variant="primary")

        with gr.Accordion("Latent Blending", open=False):
            prompt = gr.Textbox(label="prompt")
            negative_prompt = gr.Textbox(label="negative prompt")

            b_compute = gr.Button("generate preview", variant="primary")
            gallery_preview = gr.Gallery(columns=4, allow_preview=False)

            b_select = gr.Button("add selected image", variant="primary")
            gallery_movie = gr.Gallery(columns=20, allow_preview=False)

            b_delete = gr.Button("delete")
            b_earlier = gr.Button("earlier")
            b_later = gr.Button("later")

            t_per_segment = gr.Slider(1, 30, 10, step=0.1)
            b_movie = gr.Button("generate movie", variant="primary")
            movie = gr.Video()

        b_start.click(mur.register_new_user, [model, width, height], user_id)
        b_compute.click(mur.compute_imgs, [user_id, prompt, negative_prompt], gallery_preview)
        gallery_preview.select(
            mur.preview_img_selected,
            inputs=[user_id],
        )

        gallery_movie.select(
            mur.movie_img_selected,
            inputs=[user_id],
        )

        b_select.click(
            mur.add_image_to_video,
            inputs=[user_id],
            outputs=gallery_movie,
        )

        b_delete.click(
            mur.img_movie_delete,
            inputs=[user_id],
            outputs=gallery_movie,
        )

        b_earlier.click(
            mur.img_movie_earlier,
            inputs=[user_id],
            outputs=gallery_movie,
        )

        b_later.click(
            mur.img_movie_later,
            inputs=[user_id],
            outputs=gallery_movie,
        )

        b_movie.click(
        mur.generate_movie,
        inputs=[user_id, t_per_segment],
        outputs=movie,
        )


    demo.launch(share=True)

