# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import argparse
import datetime
import json
import os

import torch

from run import (
    load_pipeline,
    run_batch_generation,
    run_anchor_generation,
    run_extra_generation,
)


def str2bool(v):
    """Robust bool parser for argparse (avoids bool('False') == True)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def run_batch(
    gpu: int,
    float_type,
    seed: int = 100,
    n_steps: int = 50,
    mask_dropout: float = 0.5,
    same_latent: bool = False,
    concept_token=None,
    prompts=None,
    out_dir: str | None = None,
    record_queries: bool = False,
    invert: bool = False,
    perform_sdsa: bool = True,
    perform_consistory_injection: bool = False,
    perform_styled_injection: bool = False,
    perform_original_sdxl: bool = False,
    attn_v_range=None,
    attn_qk_range=None,
    perform_ablations: bool = False,
    perform_adain_ablation: bool = False,
    perform_no_dift_ablation: bool = False,
    background_adain_mode: str = "none",  # none | pre-subject | pre-subject-background | post
):
    if concept_token is None:
        concept_token = ["dog"]
    if prompts is None:
        prompts = [
            "portrait of a happy girl wearing headphones, anime drawing",
            "portrait of a happy girl having a picnic, realistic photo",
            "portrait a happy girl in the snow, black and white sketch",
        ]
    if attn_v_range is None:
        attn_v_range = [3, 10]
    if attn_qk_range is None:
        attn_qk_range = [5, 15]

    print("Torch CUDA is available: ", torch.cuda.is_available())
    story_pipeline = load_pipeline(gpu, float_type)

    # FeatureInjector expects: None or one of {'pre-subject','pre-subject-background','post'}
    background_adain = None if background_adain_mode == "none" else background_adain_mode

    results = run_batch_generation(
        story_pipeline,
        prompts=prompts,
        concept_token=concept_token,
        seed=seed,
        n_steps=n_steps,
        mask_dropout=mask_dropout,
        same_latent=same_latent,
        record_queries=record_queries,
        invert=invert,
        perform_sdsa=perform_sdsa,
        perform_consistory_injection=perform_consistory_injection,
        perform_styled_injection=perform_styled_injection,
        background_adain=background_adain,
        perform_original_sdxl=perform_original_sdxl,
        attn_v_range=attn_v_range,
        attn_qk_range=attn_qk_range,
        perform_ablations=perform_ablations,
        perform_adain_ablation=perform_adain_ablation,
        perform_no_dift_ablation=perform_no_dift_ablation,
    )

    if out_dir is not None:
        for result in results:
            result.save(out_dir)

    return results


def run_cached_anchors(
    gpu: int,
    float_type,
    seed: int = 40,
    mask_dropout: float = 0.5,
    same_latent: bool = False,
    style: str = "A photo of ",
    subject: str = "a cute dog",
    concept_token=None,
    settings=None,
    cache_cpu_offloading: bool = False,
    out_dir: str | None = None,
):
    if concept_token is None:
        concept_token = ["dog"]
    if settings is None:
        settings = ["sitting in the beach", "standing in the snow"]

    story_pipeline = load_pipeline(gpu, float_type)
    prompts = [f"{style}{subject} {setting}" for setting in settings]
    anchor_prompts = prompts[:2]
    extra_prompts = prompts[2:]

    anchor_out_images, _anchor_image_all, anchor_cache_first_stage, anchor_cache_second_stage = run_anchor_generation(
        story_pipeline,
        anchor_prompts,
        concept_token,
        seed=seed,
        mask_dropout=mask_dropout,
        same_latent=same_latent,
        cache_cpu_offloading=cache_cpu_offloading,
    )

    if out_dir is not None:
        for i, image in enumerate(anchor_out_images):
            image.save(f"{out_dir}/anchor_image_{i}.png")

    for i, extra_prompt in enumerate(extra_prompts):
        extra_out_images, _extra_image_all = run_extra_generation(
            story_pipeline,
            [extra_prompt],
            concept_token,
            anchor_cache_first_stage,
            anchor_cache_second_stage,
            seed=seed,
            mask_dropout=mask_dropout,
            same_latent=same_latent,
            cache_cpu_offloading=cache_cpu_offloading,
        )
        if out_dir is not None:
            extra_out_images[0].save(f"{out_dir}/extra_image_{i}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", default="batch", type=str, required=False)  # batch, cached

    parser.add_argument("--gpu", default=0, type=int, required=False)
    parser.add_argument("--float_type", default=16, type=int, required=False)
    parser.add_argument("--seed", default=40, type=int, required=False)
    parser.add_argument("--n_steps", default=50, type=int, required=False)
    parser.add_argument("--perform_sdsa", default=True, type=str2bool, required=False)
    parser.add_argument("--mask_dropout", default=0.5, type=float, required=False)
    parser.add_argument("--same_latent", default=False, type=str2bool, required=False)

    parser.add_argument("--style", default="A photo of ", type=str, required=False)
    parser.add_argument("--subject", default="a cute dog", type=str, required=False)
    parser.add_argument("--concept_token", default=["dog"], type=str, nargs="*", required=False)
    parser.add_argument(
        "--prompts",
        default=[
            "portrait of a happy girl wearing headphones, anime drawing",
            "portrait of a happy girl having a picnic, realistic photo",
            "portrait a happy girl in the snow, black and white sketch",
        ],
        type=str,
        nargs="*",
        required=False,
    )
    parser.add_argument(
        "--settings",
        default=["sitting in the beach", "standing in the snow"],
        type=str,
        nargs="*",
        required=False,
    )

    parser.add_argument("--cache_cpu_offloading", default=False, type=str2bool, required=False)
    parser.add_argument("--perform_consistory_injection", default=False, type=str2bool, required=False)

    parser.add_argument("--attn_v_range", default=[3, 10], type=int, nargs="*", required=False)
    parser.add_argument("--attn_qk_range", default=[5, 15], type=int, nargs="*", required=False)

    parser.add_argument("--record_queries", default=False, type=str2bool, required=False)
    parser.add_argument("--invert", default=False, type=str2bool, required=False)

    # Experiment options (aligned with run.py)
    parser.add_argument("--perform_original_sdxl", default=False, type=str2bool, required=False)
    parser.add_argument("--perform_styled_injection", default=False, type=str2bool, required=False)
    parser.add_argument("--perform_ablations", default=False, type=str2bool, required=False)
    parser.add_argument("--perform_adain_ablation", default=False, type=str2bool, required=False)
    parser.add_argument("--perform_no_dift_ablation", default=False, type=str2bool, required=False)

    parser.add_argument("--out_dir", default=None, type=str, required=False)
    parser.add_argument(
        "--background_adain_mode",
        default="none",
        type=str,
        choices=["none", "pre-subject", "pre-subject-background", "post"],
        required=False,
    )

    args = parser.parse_args()

    float_type = torch.float16 if args.float_type == 16 else torch.float32

    # Safety check (original author intent)
    for concept in args.concept_token:
        if concept not in args.subject:
            print("Concept token should be part of the subject")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = args.out_dir or "outputs"
    args.out_dir = f"{base_out}/{'-'.join(args.concept_token)}_{timestamp}_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, mode="w", newline="") as f:
        json.dump(vars(args), f, indent=4)

    if args.run_type == "batch":
        run_batch(
            gpu=args.gpu,
            float_type=float_type,
            seed=args.seed,
            n_steps=args.n_steps,
            mask_dropout=args.mask_dropout,
            same_latent=args.same_latent,
            concept_token=args.concept_token,
            prompts=args.prompts,
            out_dir=args.out_dir,
            record_queries=args.record_queries,
            invert=args.invert,
            perform_sdsa=args.perform_sdsa,
            perform_consistory_injection=args.perform_consistory_injection,
            perform_styled_injection=args.perform_styled_injection,
            perform_original_sdxl=args.perform_original_sdxl,
            attn_v_range=args.attn_v_range,
            attn_qk_range=args.attn_qk_range,
            perform_ablations=args.perform_ablations,
            perform_adain_ablation=args.perform_adain_ablation,
            perform_no_dift_ablation=args.perform_no_dift_ablation,
            background_adain_mode=args.background_adain_mode,
        )
    elif args.run_type == "cached":
        run_cached_anchors(
            gpu=args.gpu,
            float_type=float_type,
            seed=args.seed,
            mask_dropout=args.mask_dropout,
            same_latent=args.same_latent,
            style=args.style,
            subject=args.subject,
            concept_token=args.concept_token,
            settings=args.settings,
            cache_cpu_offloading=args.cache_cpu_offloading,
            out_dir=args.out_dir,
        )
    else:
        print("Invalid run type")
