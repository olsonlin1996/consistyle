#!/usr/bin/env python3
"""獨立的主體遮罩：OWLv2 依文字找框 → SAM 依框切遮罩。

用途：評估與診斷時需要主體區域，但**不能用 ConsiStyle 自己的遮罩**（用機制內部的遮罩
評估機制本身是循環論證；D1-2 也已知它在 spaceship 上較散）。見 docs/metric_design_draft_20261008.md §3。

模型（transformers 4.35 內建，權重在 HF 快取）：
  google/owlv2-base-patch16-ensemble、facebook/sam-vit-base

限制：只取分數最高的一個框；畫面中有多個同類物件時（如 D1-3B seed 1 有兩艘船），
只會切出其中最顯著的一個。產出的遮罩須目視檢查。
"""
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor

OWL_ID = "google/owlv2-base-patch16-ensemble"
SAM_ID = "facebook/sam-vit-base"


class SubjectMasker:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.owl_proc = Owlv2Processor.from_pretrained(OWL_ID)
        self.owl = Owlv2ForObjectDetection.from_pretrained(OWL_ID).to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained(SAM_ID)
        self.sam = SamModel.from_pretrained(SAM_ID).to(device).eval()

    @torch.no_grad()
    def box(self, img, query):
        """回傳 (x0, y0, x1, y1, score)，座標為原圖像素。"""
        inputs = self.owl_proc(text=[[query]], images=img, return_tensors="pt").to(self.device)
        out = self.owl(**inputs)
        # OWLv2 會先把圖補成正方形再縮放；原圖本來就是正方形，target_sizes 直接用原尺寸
        res = self.owl_proc.post_process_object_detection(
            out, threshold=0.0, target_sizes=torch.tensor([img.size[::-1]], device=self.device))[0]
        # 分數最高的框常是旁邊的小型同類物件；改在「分數 ≥ 最高分一半」的候選中取面積最大者
        scores, boxes = res["scores"], res["boxes"]
        cand = (scores >= 0.5 * scores.max()).nonzero().flatten()
        areas = (boxes[cand, 2] - boxes[cand, 0]) * (boxes[cand, 3] - boxes[cand, 1])
        i = int(cand[areas.argmax()])
        x0, y0, x1, y1 = [float(v) for v in boxes[i]]
        w, h = img.size
        return max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1), float(res["scores"][i])

    @torch.no_grad()
    def mask(self, img, query):
        """回傳 (bool 遮罩 HxW, 框, 框分數)。"""
        img = img.convert("RGB")
        x0, y0, x1, y1, score = self.box(img, query)
        inputs = self.sam_proc(img, input_boxes=[[[x0, y0, x1, y1]]], return_tensors="pt").to(self.device)
        out = self.sam(**inputs, multimask_output=True)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
        # SAM 自評 IoU 最高的常只是零件（半個船身）；改取框內面積最大的候選，並裁到框內
        m = masks.numpy().astype(bool)
        inbox = np.zeros_like(m[0])
        inbox[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
        best = int(np.argmax([(mk & inbox).sum() for mk in m]))
        return m[best] & inbox, (x0, y0, x1, y1), score


def overlay(img, m, color=(220, 30, 30), alpha=0.55):
    """遮罩疊圖，供目視檢查。"""
    base = img.convert("RGB")
    red = Image.new("RGB", base.size, color)
    mk = Image.fromarray((m * 255).astype(np.uint8))
    return Image.composite(Image.blend(base, red, alpha), base, mk)
