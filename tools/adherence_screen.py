#!/usr/bin/env python3
"""形容詞遵從性篩選 — 純 SDXL，不含 ConsiStyle 機制。

用途：把形容詞用進正式實驗**之前**，先確認基礎模型真的會渲染它。
起因：D1-3B 指定 `floppy-eared dog`，18 張全數未被遵從，且事後才發現，
      該輪的 dog 對照組因此半廢。完整記錄見 docs/experiments/D1_5_adherence_screen.md

設定對齊 run.py 的 load_pipeline()：
  SDXL base 1.0 / DDIM / fp16 / FreeU(0.6,0.4,1.1,1.2) / vae tiling / 50 步 / guidance 5.0
場景固定為沙漠峽谷（光線中性，最好判讀）。

用法：
    python adherence_screen.py              # 跑全部候選
    python adherence_screen.py 2 7 17       # 只跑指定編號

判定標準（事前定）：3/3 採用、2/3 備用、≤1/3 淘汰；且必須一眼可判讀。

五條已驗證的經驗法則（見 D1_5 記錄）：
  1. 屬性難易：色彩 > 質感 > 形狀（與 T2I-CompBench 量化排序一致）
  2. 含數字的屬性一律不用（`three landing legs` 數量 0/3）
  3. `and` 會觸發第二個實體 → 改用「前置修飾 + 單一 with 片語」
  4. 遵從度高 ≠ 可用（還要排除污染構圖、姿勢遮蔽）
  5. 組合必須單獨測，leakage 只在兩屬性並存時顯現
"""
import os
import sys

import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

SD_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUT = os.environ.get(
    "ADHERENCE_OUT",
    "/home/olson/ai_workspace/academic_research/projects/consistyle/results/D1_5_adherence",
)
SCENE = "in a desert canyon, cinematic photo"
SEEDS = (0, 1, 2)

# (編號, subject 片段, 已測結果)  —— 結果欄為 2026-09-27 的實測判定
CANDIDATES = [
    ("1a_ear_pre",     "floppy-eared dog",                                      "1/3 ❌"),
    ("1b_ear_post",    "dog with long floppy ears",                             "1/3 ❌"),
    ("2_bandana",      "dog with a red bandana",                                "3/3 ✅"),
    ("3_chestpatch",   "dog with a white chest patch",                          "2/3 ⚠️"),
    ("4_shaggy",       "dog with shaggy fur",                                   "1-2/3 ⚠️"),
    ("5_redstripes",   "spaceship with glowing red stripes",                    "3/3 ✅"),
    ("6_threelegs",    "spaceship with three landing legs",                     "數量 0/3 ❌"),
    ("7_copper",       "spaceship with rusty copper plating",                   "3/3 ✅"),
    ("8_roundwindow",  "spaceship with a large round window",                   "污染構圖 ❌"),
    ("9_blackcollar",  "dog with a black studded collar",                       "3/3 ✅"),
    ("10_blueeye",     "dog with bright blue eyes",                             "2/3 ⚠️"),
    ("11_curledtail",  "dog with a curled tail",                                "1/3 ❌"),
    ("12_darkmask",    "dog with a dark face mask",                             "2/3 ⚠️"),
    ("13_fluffyfur",   "dog with thick fluffy fur",                             "2.5/3 ⚠️"),
    ("14_bluedome",    "spaceship with a glowing blue dome",                    "3/3 ✅"),
    ("15_dog_pair",    "dog with thick fluffy fur and a red bandana",           "兩隻狗 ⚠️"),
    ("16_ship_pair",   "spaceship with rusty copper plating and a glowing blue dome", "3/3 ✅"),
    ("17_dog_pair_v2", "fluffy dog with a red bandana",                         "3/3 ✅ 採用"),
    ("18_ship_pair_v2","copper-plated spaceship with a glowing blue dome",      "3/3 ✅ 採用"),
]


def main():
    wanted = set(sys.argv[1:])
    todo = [c for c in CANDIDATES if not wanted or c[0].split("_")[0] in wanted]
    if not todo:
        sys.exit(f"沒有符合的候選。可用編號：{[c[0] for c in CANDIDATES]}")

    device = torch.device("cuda:0")
    scheduler = DDIMScheduler.from_pretrained(SD_ID, subfolder="scheduler")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SD_ID, torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True, scheduler=scheduler,
    ).to(device)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    for tag, subj, prev in todo:
        d = f"{OUT}/{tag}"
        os.makedirs(d, exist_ok=True)
        prompt = f"a {subj} {SCENE}"
        for seed in SEEDS:
            g = torch.Generator(device).manual_seed(seed)
            latents = randn_tensor((1, pipe.unet.config.in_channels, 128, 128),
                                   generator=g, device=device, dtype=torch.float16)
            img = pipe(prompt=prompt, latents=latents,
                       num_inference_steps=50, guidance_scale=5.0).images[0]
            img.save(f"{d}/seed{seed}.png")
        with open(f"{d}/prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt + "\n")
        print(f"{tag:17} 完成（前次：{prev}）| {prompt}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
