#!/usr/bin/env python3
"""lab meeting 投影片用對照圖（16:9，1920x1080）。

只重新拼接 results/ 底下既有的輸出，不重跑生成。
起因：舊對照圖的中文標籤缺字型而成方框，且多為直長版，不適合投影片。

用法：python tools/make_slide_figs.py
輸出：../labmeeting_20261002/figs/*.png
"""
import glob
import os

import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "labmeeting_20261002", "figs")

W, H = 1920, 1080
BG, FG, SUB, ACCENT = (20, 21, 26), (235, 235, 240), (160, 165, 180), (120, 180, 255)
MARGIN, GAP, GROUP_GAP, LABEL_W = 30, 8, 28, 110
FONT_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SCENES = ["neon city", "ruined factory", "desert canyon"]
CACHED_COLS = ["anchor_image_0", "anchor_image_1", "extra_image_0"]


def _font(size):
    # .ttc 內含多個語系，挑繁中（TC）那一份，避免字形是日文或簡中
    for idx in range(10):
        try:
            f = ImageFont.truetype(FONT_TTC, size, index=idx)
        except OSError:
            break
        if "TC" in f.getname()[0]:
            return f
    return ImageFont.truetype(FONT_TTC, size)


F_TITLE, F_SUB, F_HEAD, F_SMALL = _font(40), _font(22), _font(22), _font(18)


def cached_run(exp_dir, subject, seed):
    """cached 模式一個 run 的三張圖，依場景順序。"""
    d = glob.glob(f"{RES}/{exp_dir}/{subject}_seed{seed}/*_seed{seed}")
    assert len(d) == 1, (exp_dir, subject, seed, d)
    return [os.path.join(d[0], f"{c}.png") for c in CACHED_COLS]


def batch_run(exp_dir, seed, branch="consistyle"):
    d = glob.glob(f"{RES}/{exp_dir}/v2_seed{seed}/*_seed{seed}/{branch}")
    assert len(d) == 1, (exp_dir, seed, d)
    return [os.path.join(d[0], f"image_{i}.png") for i in range(3)]


def vanilla_run(subject, seed):
    return [f"{RES}/D1_4_vanilla/{subject}_seed{seed}/image_{i}.png" for i in range(3)]


def grid(name, title, subtitle, groups, rows, col_heads=None, row_notes=None, label_w=LABEL_W):
    """groups: [組名, ...]；rows: [(列標籤, [每組的圖片路徑或 PIL.Image 清單]), ...]"""
    n_groups = len(groups)
    n_cols = len(rows[0][1][0])
    top = MARGIN + 60 + (34 if subtitle else 0) + 16
    head_h = 34 + (30 if col_heads else 0) + (24 if col_heads and any("\n" in h for h in col_heads) else 0)
    avail_w = W - 2 * MARGIN - label_w - (n_groups - 1) * GROUP_GAP - n_groups * (n_cols - 1) * GAP
    avail_h = H - top - head_h - MARGIN - (len(rows) - 1) * GAP
    cell = int(min(avail_w / (n_groups * n_cols), avail_h / len(rows)))

    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas)
    dr.text((MARGIN, MARGIN), title, font=F_TITLE, fill=FG)
    if subtitle:
        dr.text((MARGIN, MARGIN + 60), subtitle, font=F_SUB, fill=SUB)

    group_w = n_cols * cell + (n_cols - 1) * GAP
    # 內容比畫面窄時（例如 D1-2 只有一組）水平置中，標籤欄跟著移
    content_w = label_w + n_groups * group_w + (n_groups - 1) * GROUP_GAP
    left = max(MARGIN, (W - content_w) // 2)
    x0 = left + label_w
    for g, gname in enumerate(groups):
        gx = x0 + g * (group_w + GROUP_GAP)
        dr.text((gx, top), gname, font=F_HEAD, fill=ACCENT)
        if col_heads:
            for c, ch in enumerate(col_heads):
                dr.multiline_text((gx + c * (cell + GAP), top + 32), ch, font=F_SMALL, fill=SUB, spacing=2)

    y = top + head_h
    for r, (rlabel, per_group) in enumerate(rows):
        dr.text((left, y + cell // 2 - 14), rlabel, font=F_HEAD, fill=FG)
        if row_notes and row_notes[r]:
            dr.text((left, y + cell // 2 + 14), row_notes[r], font=F_SMALL, fill=SUB)
        for g, imgs in enumerate(per_group):
            gx = x0 + g * (group_w + GROUP_GAP)
            for c, im in enumerate(imgs):
                im = Image.open(im) if isinstance(im, str) else im
                canvas.paste(im.convert("RGB").resize((cell, cell), Image.LANCZOS),
                             (gx + c * (cell + GAP), y))
        y += cell + GAP

    os.makedirs(OUT, exist_ok=True)
    canvas.save(os.path.join(OUT, name + ".png"), optimize=True)
    print("ok", name, "cell", cell)


def scene_heads():
    return [f"{c}\n{s}" for c, s in zip(CACHED_COLS, SCENES)]


def mask_panels(exp_dir, subject, seed):
    """anchor 圖疊 64x64 遮罩（紅），加原始 cross-attention 圖；並回傳兩張 anchor 的 CV。"""
    d = torch.load(f"{RES}/{exp_dir}/{subject}_seed{seed}/probe/anchor_masks.pt", map_location="cpu")
    imgs = cached_run(exp_dir, subject, seed)[:2]
    panels, cvs = [], []
    for i in range(2):
        base = Image.open(imgs[i]).convert("RGB").resize((512, 512))
        m = d["last_mask"][64][i].reshape(64, 64).float().numpy()
        mask = Image.fromarray((m * 255).astype("uint8")).resize((512, 512), Image.NEAREST)
        red = Image.new("RGB", (512, 512), (220, 30, 30))
        overlay = Image.composite(Image.blend(base, red, 0.55), base, mask)
        a = d["last_attn_raw"][64][i, 0].float()
        cvs.append(float(a.std() / a.mean()))
        an = ((a - a.min()) / (a.max() - a.min()) * 255).byte().numpy()
        panels += [overlay, Image.fromarray(an).resize((512, 512), Image.BILINEAR)]
    return panels, cvs


def main():
    seeds = [0, 1, 2]
    heads = scene_heads()

    grid("D1-0_failure_case", "D1-0　failure case 復現",
         "prompt：a spaceship / a dog + 三個場景（裸名詞）　每列同一 seed　ConsiStyle cached 模式",
         ["spaceship", "dog"],
         [(f"seed {s}", [cached_run("D1_0", "spaceship", s), cached_run("D1_0", "dog", s)]) for s in seeds],
         col_heads=heads)

    for subj in ["spaceship", "dog"]:
        grid(f"D1-1_same_latent_{subj}", f"D1-1　同一起始雜訊（{subj}）",
             "唯一變數：--same_latent（所有圖共用第一份雜訊）　其餘同 D1-0",
             ["same_latent = False（D1-0）", "same_latent = True（D1-1）"],
             [(f"seed {s}", [cached_run("D1_0", subj, s), cached_run("D1_1_samelatent", subj, s)]) for s in seeds],
             col_heads=heads)

        rows, notes = [], []
        for s in seeds:
            p, cv = mask_panels("D1_2_maskprobe", subj, s)
            rows.append((f"seed {s}", [p]))
            notes.append(f"CV {cv[0]:.3f} / {cv[1]:.3f}")
        grid(f"D1-2_mask_probe_{subj}", f"D1-2　anchor 遮罩探針（{subj}）",
             "紅色＝二值化遮罩（64×64）　灰階＝二值化前的 cross-attention（逐張正規化）　CV＝σ/μ，越低越平坦",
             ["anchor_image_0 / anchor_image_1"], rows,
             col_heads=["anchor_image_0 + 遮罩", "anchor_image_0 注意力", "anchor_image_1 + 遮罩", "anchor_image_1 注意力"],
             row_notes=notes, label_w=190)

        shape = {"spaceship": "saucer-shaped spaceship with two swept wings",
                 "dog": "floppy-eared dog with a long muzzle"}[subj]
        grid(f"D1-3B_shape_{subj}", f"D1-3B　在 prompt 指定形狀（{subj}）",
             f"右組 subject：{shape}　其餘同 D1-0",
             [f"裸名詞 a {subj}（D1-0）", "指定形狀（D1-3B）"],
             [(f"seed {s}", [cached_run("D1_0", subj, s), cached_run("D1_3B_shape", subj, s)]) for s in seeds],
             col_heads=heads)

        grid(f"D1-4_vanilla_{subj}", f"D1-4　純 SDXL 對照（{subj}）",
             f"兩組 prompt 相同：{shape}　左組不含任何 ConsiStyle 機制、逐張獨立生成",
             ["純 SDXL（D1-4）", "ConsiStyle（D1-3B）"],
             [(f"seed {s}", [vanilla_run(subj, s), cached_run("D1_3B_shape", subj, s)]) for s in seeds],
             col_heads=SCENES)

    grid("D1-5_adherence_final", "D1-5　形容詞遵從性篩選：採用的兩條 prompt",
         "純 SDXL × in a desert canyon × 3 seed　兩屬性皆 3/3、無 attribute leakage、單一主體",
         ["seed 0 / seed 1 / seed 2"],
         [("spaceship", [[f"{RES}/D1_5_adherence/18_ship_pair_v2/seed{s}.png" for s in seeds]]),
          ("dog", [[f"{RES}/D1_5_adherence/17_dog_pair_v2/seed{s}.png" for s in seeds]])],
         row_notes=["copper-plated", "fluffy"], label_w=150)

    grid("D1-6_batch_vs_cached", "D1-6　batch 與 cached 執行模式（spaceship）",
         "prompt：a spaceship + 三個場景　batch B=3（v2：perform_original_sdxl + perform_styled_injection）",
         ["cached（D1-0）", "batch 的 consistyle 輸出（D1-6 v2）"],
         [(f"seed {s}", [cached_run("D1_0", "spaceship", s), batch_run("D1_6_batch", s)]) for s in seeds],
         col_heads=SCENES)


if __name__ == "__main__":
    main()
