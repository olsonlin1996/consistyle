#!/usr/bin/env python3
"""D1-7 未解矛盾的檢查：四個輸出變體的兩兩差異，整張圖 vs 只算主體區域。

D1-6 §10 記載「注入規模極大，但四階段輸出的兩兩差異只有 16–24/255」。
懷疑整張圖平均把主體區域的差異稀釋了（主體只佔畫面約兩成）。本腳本改用獨立遮罩
（tools/subject_mask.py，OWLv2＋SAM）分開計算主體與背景的平均絕對差。

用法：python tools/d17_masked_diff.py
輸出：results/D1_7b_maskeddiff/{masks_*.png, masked_diff.json}
"""
import glob
import json
import os

import numpy as np
from PIL import Image

from subject_mask import SubjectMasker, overlay

R = "/home/olson/ai_workspace/academic_research/projects/consistyle/results"
OUT = f"{R}/D1_7b_maskeddiff"
VARIANTS = ["original sdxl", "first pass", "consistory", "consistyle"]


def main():
    os.makedirs(OUT, exist_ok=True)
    sm = SubjectMasker()
    # D1-7 batch_seed0 與 D1-6 v3 seed 0 為同一設定，輸出相同，只算 D1-6 v3 三個 seed
    runs = {f"D1-6v3_seed{s}": glob.glob(f"{R}/D1_6_batch/v3_seed{s}/*_seed{s}")[0] for s in range(3)}
    summary = {}
    for name, d in runs.items():
        data, tiles = {}, []
        for v in VARIANTS:
            for i in range(3):
                img = Image.open(f"{d}/{v}/image_{i}.png").convert("RGB")
                m, _, sc = sm.mask(img, "a spaceship")
                data[(v, i)] = (np.asarray(img).astype(np.float32), m, sc)
                tiles.append(overlay(img, m).resize((256, 256)))
        sheet = Image.new("RGB", (3 * 256, 4 * 256))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % 3) * 256, (k // 3) * 256))
        sheet.save(f"{OUT}/masks_{name}.png")  # 列 = 變體（依 VARIANTS 順序），欄 = 三個場景

        rows = []
        for a in range(4):
            for b in range(a + 1, 4):
                for i in range(3):
                    xa, ma, _ = data[(VARIANTS[a], i)]
                    xb, mb, _ = data[(VARIANTS[b], i)]
                    ad = np.abs(xa - xb).mean(axis=2)
                    u = ma | mb  # 兩張的主體聯集
                    rows.append(dict(pair=f"{VARIANTS[a]} vs {VARIANTS[b]}", img=i,
                                     whole=float(ad.mean()), subject=float(ad[u].mean()),
                                     background=float(ad[~u].mean()), coverage=float(u.mean())))
        summary[name] = rows
        f = lambda k: np.array([r[k] for r in rows])
        print(f"{name:14s} whole {f('whole').min():5.1f}-{f('whole').max():5.1f} (mean {f('whole').mean():5.1f})"
              f" | subject {f('subject').min():5.1f}-{f('subject').max():5.1f} (mean {f('subject').mean():5.1f})"
              f" | background mean {f('background').mean():5.1f} | coverage {f('coverage').mean():.2f}"
              f" | min box score {min(x[2] for x in data.values()):.2f}")
    json.dump(summary, open(f"{OUT}/masked_diff.json", "w"), indent=1)


if __name__ == "__main__":
    main()
