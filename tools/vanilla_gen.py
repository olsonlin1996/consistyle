#!/usr/bin/env python3
"""純 SDXL 生成（參數化版）—— 不含 ConsiStyle 任何機制。

生成邏輯逐行沿用 vanilla_sdxl.py（D1-4），只把寫死的路徑、主體、場景改成參數，
使 D1-8 的 prompt 篩選（Phase B）與純 SDXL 對照（Phase D）可共用。
與 D1-4 的位元一致性以 `--verify-d14` 檢查（重跑 D1-4 spaceship seed 0 並比對 md5）。

用法：
    python tools/vanilla_gen.py --subject "sport motorcycle" --out <目錄> \
        --scenes "in a desert canyon, cinematic photo" --seeds 0 1 2
    python tools/vanilla_gen.py --verify-d14 --out <暫存目錄>

輸出：<out>/seed<N>/image_<i>.png（i 對應 --scenes 的順序）與 <out>/prompt.txt
"""
import argparse
import hashlib
import os

import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

SD_ID = "stabilityai/stable-diffusion-xl-base-1.0"
D14_SCENES = [
    "in a neon city, cinematic photo",
    "in a ruined factory, cinematic photo",
    "in a desert canyon, cinematic photo",
]
D14_DIR = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"), "D1_4_vanilla")


def load_pipe(device):
    # 設定對齊 run.py 的 load_pipeline()，與 vanilla_sdxl.py 相同
    scheduler = DDIMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SD_ID, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, scheduler=scheduler,
    ).to(device)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, device, subject, scenes, seeds, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "prompt.txt"), "w") as f:
        f.write("\n".join(f"a {subject} {st}" for st in scenes) + "\n")
    for seed in seeds:
        d = os.path.join(out, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        # 比照 create_latents：同一 seed 下一次產生 len(scenes) 份 latent，再逐張取用
        g = torch.Generator(device).manual_seed(seed)
        shape = (len(scenes), pipe.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=g, device=device, dtype=torch.float16)
        for i, st in enumerate(scenes):
            img = pipe(prompt=f"a {subject} {st}", latents=latents[i:i + 1],
                       num_inference_steps=50, guidance_scale=5.0).images[0]
            img.save(os.path.join(d, f"image_{i}.png"))
        print(f"[vanilla_gen] {subject} seed {seed} 完成", flush=True)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--scenes", nargs="+", default=D14_SCENES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-d14", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    pipe = load_pipe(device)

    if args.verify_d14:
        generate(pipe, device, "saucer-shaped spaceship with two swept wings", D14_SCENES, [0], args.out)
        ok = True
        for i in range(3):
            a = md5(os.path.join(args.out, "seed0", f"image_{i}.png"))
            b = md5(os.path.join(D14_DIR, "spaceship_seed0", f"image_{i}.png"))
            print(f"image_{i}: {'SAME' if a == b else 'DIFF'} {a} {b}")
            ok &= a == b
        print("VERIFY_D14", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    if not args.subject:
        ap.error("--subject 必填（除非用 --verify-d14）")
    generate(pipe, device, args.subject, args.scenes, args.seeds, args.out)
    print("ALL DONE")


if __name__ == "__main__":
    main()
