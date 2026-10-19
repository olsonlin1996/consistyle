#!/usr/bin/env python3
"""純 SDXL 對照 — 不含 ConsiStyle 任何機制（無共享注意力、無特徵注入）。

設定刻意對齊 run.py 的 load_pipeline()，使差異只剩「有無 ConsiStyle 機制」：
  SDXL base 1.0 / DDIMScheduler / fp16 / FreeU(0.6,0.4,1.1,1.2) / vae tiling
  n_steps=50 / guidance_scale=5.0（pipeline.py:62 的預設值）
"""
import os
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

SD_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUT = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"), "D1_4_vanilla")
SETTINGS = [
    "in a neon city, cinematic photo",
    "in a ruined factory, cinematic photo",
    "in a desert canyon, cinematic photo",
]
SUBJECTS = {
    "spaceship": "saucer-shaped spaceship with two swept wings",
    "dog": "floppy-eared dog with a long muzzle",
}

device = torch.device("cuda:0")
scheduler = DDIMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
pipe = StableDiffusionXLPipeline.from_pretrained(
    SD_ID, torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True, scheduler=scheduler,
).to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)

for name, subj in SUBJECTS.items():
    for seed in (0, 1, 2):
        d = f"{OUT}/{name}_seed{seed}"
        os.makedirs(d, exist_ok=True)
        # 比照 create_latents：同一 seed 下一次產生 batch 份 latent，再逐張取用
        g = torch.Generator(device).manual_seed(seed)
        shape = (len(SETTINGS), pipe.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=torch.float16)
        for i, st in enumerate(SETTINGS):
            prompt = f"a {subj} {st}"
            img = pipe(prompt=prompt, latents=latents[i:i + 1],
                       num_inference_steps=50, guidance_scale=5.0).images[0]
            img.save(f"{d}/image_{i}.png")
        print(f"{name} seed {seed} 完成", flush=True)
print("ALL DONE")
