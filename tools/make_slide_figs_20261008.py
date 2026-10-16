#!/usr/bin/env python3
"""lab meeting 投影片用圖（續篇：2026-10-02 之後的進度，D1-8）。

版面沿用 make_slide_figs.py 的 grid()（1920x1080、深色底、繁中字型），只換輸出目錄。
圖表（× 數、CV）用 matplotlib，配色經 dataviz 驗證器檢查（深色底 #14151a，三色全過）。

用法：python tools/make_slide_figs_20261008.py
輸出：../labmeeting_20261008/figs/*.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

import make_slide_figs as base

base.OUT = os.path.join(base.ROOT, "labmeeting_20261008", "figs")
RES = base.RES
SCENES = base.SCENES

# 深色投影片配色（與 make_slide_figs 一致）；類別色依序取用，驗證器全過
BG, FG, SUB, GRID = "#14151a", "#ebebf0", "#a0a5b4", "#2c2f38"
C_CONSI, C_VAN, C_THIRD = "#3987e5", "#d95926", "#199e70"
font_manager.fontManager.addfont(base.FONT_TTC)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=base.FONT_TTC).get_name()


def cached_run(exp_dir, prefix, seed):
    d = glob.glob(f"{RES}/{exp_dir}/{prefix}_seed{seed}/*_seed{seed}")
    assert len(d) == 1, (exp_dir, prefix, seed, d)
    return [os.path.join(d[0], f"{c}.png") for c in base.CACHED_COLS]


def vanilla_moto(seed):
    return [f"{RES}/D1_8_motorcycle/vanilla/seed{seed}/image_{i}.png" for i in range(3)]


def style_axes(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SUB, labelsize=15)


def save(fig, name):
    os.makedirs(base.OUT, exist_ok=True)
    fig.savefig(os.path.join(base.OUT, name + ".png"), dpi=100, facecolor=BG)
    plt.close(fig)
    print("ok", name)


def chart_counts():
    """三主體 × 兩條件的細節不一致格數（人工判讀，滿分 15）。"""
    subjects = ["太空船", "狗", "motorcycle"]
    van = [15, 11, 9]
    consi = [9, 3, 7]
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    fig.text(0.03, 0.93, "D1-8　三個主體的細節不一致格數（人工判讀，滿分 15）", color=FG, fontsize=30)
    fig.text(0.03, 0.885, "每格 = 一個事先登記的細節 × 一個 seed；三張圖有任一不同即算 ×。越少越一致",
             color=SUB, fontsize=18)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.66])
    style_axes(ax)
    x = range(len(subjects))
    w = 0.34
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], van, w, color=C_VAN, label="純 SDXL（無機制）")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], consi, w, color=C_CONSI, label="ConsiStyle")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.25, f"{int(r.get_height())}",
                    ha="center", va="bottom", color=FG, fontsize=20)
    ax.set_xticks(list(x))
    ax.set_xticklabels(subjects, fontsize=22, color=FG)
    ax.set_ylim(0, 16.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", fontsize=18, frameon=False)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.text(0.03, 0.05, "motorcycle 的 ConsiStyle：初判 3，經指出後放大重看更正為 7。"
             "太空船、狗、motorcycle 純 SDXL 是在較小的圖上判的，判讀標準不完全一致",
             color=SUB, fontsize=16)
    save(fig, "D1-8_detail_counts")


def anchor_cv(path):
    d = torch.load(path, map_location="cpu")
    out = []
    for i in range(2):
        a = d["last_attn_raw"][64][i, 0].float()
        out.append(float(a.std() / a.mean()))
    return out


def chart_cv():
    """各條件下兩張 anchor 的原始注意力 CV（每點一張 anchor，3 seed × 2 = 6 點）。"""
    rows = [
        ("太空船｜裸名詞（D1-2）", "D1_2_maskprobe", "spaceship_seed{}", C_VAN),
        ("太空船｜指定形狀（D1-3B）", "D1_3B_shape", "spaceship_seed{}", C_VAN),
        ("motorcycle｜cruiser（D1-8）", "D1_8_motorcycle", "consistyle_seed{}", C_THIRD),
        ("狗｜裸名詞（D1-2）", "D1_2_maskprobe", "dog_seed{}", C_CONSI),
        ("狗｜指定形狀（D1-3B）", "D1_3B_shape", "dog_seed{}", C_CONSI),
    ]
    fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)
    fig.text(0.03, 0.93, "注意力的平坦程度（CV = σ/μ，越低越平坦）", color=FG, fontsize=30)
    fig.text(0.03, 0.885, "每點一張 anchor 的二值化前 cross-attention（64×64）；短豎線為平均",
             color=SUB, fontsize=18)
    ax = fig.add_axes([0.30, 0.14, 0.64, 0.68])
    style_axes(ax)
    for y, (label, exp, pat, col) in enumerate(reversed(rows)):
        vals = []
        for s in range(3):
            vals += anchor_cv(f"{RES}/{exp}/{pat.format(s)}/probe/anchor_masks.pt")
        m = sum(vals) / len(vals)
        ax.scatter(vals, [y] * len(vals), s=140, color=col, edgecolors=BG, linewidths=2, zorder=3)
        ax.plot([m, m], [y - 0.28, y + 0.28], color=FG, linewidth=2.5, zorder=4)
        ax.text(0.205, y, f"平均 {m:.3f}", va="center", color=FG, fontsize=17)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=19, color=FG)
    ax.set_xlim(0.06, 0.235)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    fig.text(0.03, 0.05, "太空船與 motorcycle 同樣平坦，狗明顯較集中；prompt 是否指定形狀幾乎不影響",
             color=SUB, fontsize=16)
    save(fig, "D1-8_attention_cv")


def main():
    seeds = [0, 1, 2]
    heads = base.scene_heads()

    base.grid("D1-8_phaseB_screening", "D1-8 Phase B　motorcycle prompt 篩選（純 SDXL）",
              "in a desert canyon × 3 seed　四條件：車型 3/3、單一主體、結構合理、三張細節彼此不同 → 依序取第一個全過者：cruiser",
              ["seed 0 / seed 1 / seed 2"],
              [(t, [[f"{RES}/D1_8_phaseB/{d}/seed{s}/image_0.png" for s in seeds]])
               for t, d in [("sport", "1_sport"), ("cruiser", "2_cruiser"), ("vintage", "3_vintage")]],
              row_notes=["淘汰：車型不一", "採用", "備用"], label_w=170)

    base.grid("D1-8_motorcycle_vanilla_vs_consistyle", "D1-8　motorcycle：純 SDXL vs ConsiStyle",
              "兩組 prompt 相同：a cruiser motorcycle + 三個場景　--concept_token motorcycle，其餘參數逐字同 D1-3B",
              ["純 SDXL", "ConsiStyle"],
              [(f"seed {s}", [vanilla_moto(s), cached_run("D1_8_motorcycle", "consistyle", s)]) for s in seeds],
              col_heads=SCENES)

    base.grid("D1-8_motorcycle_rejudge", "D1-8　放大重看：外形一樣，細節不同",
              "ConsiStyle 輸出。初判在小圖上給 × 3，放大重看更正為 × 7",
              ["anchor_image_0 / anchor_image_1 / extra_image_0"],
              [(f"seed {s}", [cached_run("D1_8_motorcycle", "consistyle", s)]) for s in (0, 1)],
              col_heads=heads,
              row_notes=["鏽斑、油箱標誌、排氣管", "油箱圖紋、排氣管、行李箱"], label_w=230)

    chart_counts()
    chart_cv()


if __name__ == "__main__":
    main()
