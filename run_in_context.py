#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

from utils.utils import (
    box2squre,
    crop_back,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    pad_to_square,
)


DEFAULT_LORA_WEIGHT = "20250321_steps5000_pytorch_lora_weights.safetensors"
DEFAULT_NUNCHAKU_TRANSFORMER = "mit-han-lab/svdq-int4-flux.1-fill-dev"
DEFAULT_NUNCHAKU_LORA = (
    "aha2023/insert-anything-lora-for-nunchaku/"
    "insert-anything_extracted_lora_rank_64-bf16.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run In-Context editing on a source/reference image-mask pair.")
    parser.add_argument("--source-image", "--source_image", required=True)
    parser.add_argument("--source-mask", "--source_mask", required=True)
    parser.add_argument("--ref-image", "--ref_image", "--reference-image", "--reference_image", required=True)
    parser.add_argument("--ref-mask", "--ref_mask", "--reference-mask", "--reference_mask", required=True)
    parser.add_argument("--out-dir", "--out_dir", default="result")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", "--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance-scale", "--guidance_scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", "--max_sequence_length", type=int, default=512)
    parser.add_argument("--prepare-only", "--prepare_only", action="store_true")
    parser.add_argument("--cpu-offload", "--cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sequential-cpu-offload",
        "--sequential_cpu_offload",
        action="store_true",
        help="Use lower-VRAM sequential offload instead of model-level offload.",
    )
    parser.add_argument("--local-files-only", "--local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--flux-fill-path",
        "--flux_fill_path",
        default=os.environ.get("FLUX_FILL_PATH", "black-forest-labs/FLUX.1-Fill-dev"),
    )
    parser.add_argument(
        "--flux-redux-path",
        "--flux_redux_path",
        default=os.environ.get("FLUX_REDUX_PATH", "black-forest-labs/FLUX.1-Redux-dev"),
    )
    parser.add_argument(
        "--lora-path",
        "--lora_path",
        default=os.environ.get("IN_CONTEXT_LORA_PATH", "WensongSong/Insert-Anything"),
    )
    parser.add_argument(
        "--lora-weight-name",
        "--lora_weight_name",
        default=os.environ.get("IN_CONTEXT_LORA_WEIGHT", DEFAULT_LORA_WEIGHT),
    )
    parser.add_argument("--nunchaku", action="store_true", help="Use the Nunchaku int4 FLUX transformer.")
    parser.add_argument(
        "--nunchaku-transformer-path",
        "--nunchaku_transformer_path",
        default=os.environ.get("NUNCHAKU_TRANSFORMER_PATH", DEFAULT_NUNCHAKU_TRANSFORMER),
    )
    parser.add_argument(
        "--nunchaku-lora-path",
        "--nunchaku_lora_path",
        default=os.environ.get("NUNCHAKU_LORA_PATH", DEFAULT_NUNCHAKU_LORA),
    )
    parser.add_argument("--nunchaku-lora-strength", "--nunchaku_lora_strength", type=float, default=1.0)
    parser.add_argument("--nunchaku-precision", "--nunchaku_precision", choices=["auto", "int4", "fp4"], default=os.environ.get("NUNCHAKU_PRECISION", "auto"), help="Quantization precision passed to Nunchaku.")
    parser.add_argument(
        "--full-flux-quantize",
        "--full_flux_quantize",
        choices=["none", "int8", "int4"],
        default=os.environ.get("FULL_FLUX_QUANTIZE", "none"),
        help="Optional bitsandbytes quantization for the full diffusers FLUX transformer.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def enable_vae_slicing(pipe) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    else:
        pipe.enable_vae_slicing()


def cuda_runtime_status() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    try:
        torch.cuda.current_device()
        torch.empty(1, device="cuda")
    except Exception as exc:
        return False, f"CUDA runtime initialization failed: {type(exc).__name__}: {exc}"
    return True, "ok"


def validate_pipeline_runtime(args: argparse.Namespace) -> None:
    device = str(getattr(args, "device", "cuda") or "cuda")
    full_flux_quantize = getattr(args, "full_flux_quantize", "none")
    cpu_offload = bool(getattr(args, "cpu_offload", False))
    sequential_cpu_offload = bool(getattr(args, "sequential_cpu_offload", False))
    needs_cuda = full_flux_quantize != "none" or device.startswith("cuda") or cpu_offload or sequential_cpu_offload
    cuda_ready, cuda_reason = cuda_runtime_status() if needs_cuda else (False, "not requested")

    if full_flux_quantize != "none" and not cuda_ready:
        raise RuntimeError(
            f"--full-flux-quantize {full_flux_quantize} requires a working PyTorch CUDA runtime. "
            f"{cuda_reason}. nvidia-smi can see the driver/GPU, but PyTorch must also be able "
            "to initialize CUDA. Run this in a CUDA-capable PyTorch environment, or use "
            "--full-flux-quantize none only for CPU smoke tests."
        )
    if device.startswith("cuda") and not cuda_ready:
        raise RuntimeError(
            f"--device {device} was requested, but PyTorch CUDA is not usable. {cuda_reason}. "
            "Run this in a CUDA-capable PyTorch environment or pass --device cpu for CPU-only smoke tests."
        )
    if (cpu_offload or sequential_cpu_offload) and not cuda_ready:
        raise RuntimeError(
            "CPU offload still needs CUDA as the execution device, but PyTorch CUDA is not usable. "
            f"{cuda_reason}. Disable offload and use --device cpu only for CPU-only smoke tests."
        )


def bitsandbytes_quantization_config(quantize: str, dtype: torch.dtype):
    if quantize == "none":
        return None
    from diffusers import BitsAndBytesConfig

    if quantize == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantize == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported full FLUX quantization: {quantize}")


def load_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 128).astype(np.uint8)
    if int(mask.sum()) == 0:
        raise ValueError(f"Mask is empty after thresholding: {path}")
    return mask


def prepare_diptych(
    source_image_path: str | Path,
    source_mask_path: str | Path,
    ref_image_path: str | Path,
    ref_mask_path: str | Path,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    ref_image = load_rgb(ref_image_path)
    tar_image = load_rgb(source_image_path)
    ref_mask = load_mask(ref_mask_path)
    tar_mask = load_mask(source_mask_path, size=(tar_image.shape[1], tar_image.shape[0]))

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=1.3)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)

    kernel = np.ones((7, 7), np.uint8)
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=2)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2)
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]
    h1, w1 = tar_image.shape[:2]

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size, interpolation=cv2.INTER_NEAREST)

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    tar_image = pad_to_square(tar_image, pad_value=255)
    h2, w2 = tar_image.shape[:2]
    tar_image = cv2.resize(tar_image, size)

    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.zeros_like(tar_image)
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
    mask_diptych[mask_diptych == 1] = 255

    return (
        Image.fromarray(diptych_ref_tar),
        Image.fromarray(mask_diptych.astype(np.uint8)),
        masked_ref_image,
        old_tar_image,
        np.array([h1, w1, h2, w2]),
        np.array(tar_box_yyxx_crop),
    )


def resolve_hf_file(path_or_repo_file: str, local_files_only: bool) -> str:
    path = Path(path_or_repo_file).expanduser()
    if path.is_file():
        return str(path)

    repo_id = os.path.dirname(path_or_repo_file)
    filename = os.path.basename(path_or_repo_file)
    if not repo_id or repo_id == ".":
        raise FileNotFoundError(f"File not found: {path_or_repo_file}")
    return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=local_files_only)


def resolve_hf_repo_dir(path_or_repo: str, local_files_only: bool) -> str:
    path = Path(path_or_repo).expanduser()
    if path.exists():
        return str(path)
    return snapshot_download(repo_id=path_or_repo, local_files_only=local_files_only)


def load_pipelines(args: argparse.Namespace) -> tuple[FluxFillPipeline, FluxPriorReduxPipeline]:
    validate_pipeline_runtime(args)
    dtype = dtype_from_name(args.dtype)
    if args.nunchaku:
        from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

        nunchaku_transformer_path = resolve_hf_repo_dir(args.nunchaku_transformer_path, args.local_files_only)
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            nunchaku_transformer_path,
            torch_dtype=dtype,
            device=args.device,
            precision=getattr(args, "nunchaku_precision", "auto"),
        )
        pipe = FluxFillPipeline.from_pretrained(
            args.flux_fill_path,
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        )
        nunchaku_lora_path = resolve_hf_file(args.nunchaku_lora_path, args.local_files_only)
        transformer.update_lora_params(nunchaku_lora_path)
        transformer.set_lora_strength(args.nunchaku_lora_strength)
    else:
        full_flux_quantize = getattr(args, "full_flux_quantize", "none")
        quantization_config = bitsandbytes_quantization_config(full_flux_quantize, dtype)
        if quantization_config is None:
            pipe = FluxFillPipeline.from_pretrained(
                args.flux_fill_path,
                torch_dtype=dtype,
                local_files_only=args.local_files_only,
            )
        else:
            from diffusers import FluxTransformer2DModel

            transformer = FluxTransformer2DModel.from_pretrained(
                args.flux_fill_path,
                subfolder="transformer",
                torch_dtype=dtype,
                quantization_config=quantization_config,
                local_files_only=args.local_files_only,
            )
            pipe = FluxFillPipeline.from_pretrained(
                args.flux_fill_path,
                transformer=transformer,
                torch_dtype=dtype,
                local_files_only=args.local_files_only,
            )
        pipe.load_lora_weights(
            args.lora_path,
            weight_name=args.lora_weight_name,
            local_files_only=args.local_files_only,
        )
        # Diffusers/PEFT adapter naming has varied across releases, especially
        # when the transformer is wrapped for BitsAndBytes quantization. Align
        # the active selection with the adapters that were actually registered
        # and fail closed: LoRA is part of this method's runtime contract.
        peft_config = getattr(pipe.transformer, "peft_config", None)
        available_adapters = list(peft_config.keys()) if isinstance(peft_config, dict) else []
        initial_active_adapters = (
            list(pipe.get_active_adapters())
            if hasattr(pipe, "get_active_adapters")
            else []
        )
        active_adapters = list(initial_active_adapters)
        realigned = False
        if available_adapters and not set(active_adapters).intersection(available_adapters):
            pipe.set_adapters(available_adapters)
            active_adapters = list(pipe.get_active_adapters())
            realigned = True
        lora_runtime_audit = {
            "available_adapters": available_adapters,
            "initial_active_adapters": initial_active_adapters,
            "active_adapters": active_adapters,
            "realigned": realigned,
            "verified": bool(set(active_adapters).intersection(available_adapters)),
        }
        if not lora_runtime_audit["verified"]:
            raise RuntimeError(
                "In-Context LoRA is not active after load: "
                f"available={available_adapters}, active={active_adapters}"
            )
        pipe._in_context_lora_audit = lora_runtime_audit
        print(
            "In-Context LoRA adapters: "
            f"available={available_adapters}, initial={initial_active_adapters}, "
            f"active={active_adapters}, realigned={realigned}, verified=True",
            flush=True,
        )
    redux = FluxPriorReduxPipeline.from_pretrained(
        args.flux_redux_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )

    if args.sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        enable_vae_slicing(pipe)
        redux.enable_model_cpu_offload()
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload()
        enable_vae_slicing(pipe)
        redux.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
        redux.to(args.device)
    return pipe, redux


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (args.size, args.size)
    diptych_ref_tar, mask_diptych, masked_ref_image, old_tar_image, extra_sizes, crop_box = prepare_diptych(
        args.source_image,
        args.source_mask,
        args.ref_image,
        args.ref_mask,
        size,
    )
    Image.fromarray(masked_ref_image).save(out_dir / "debug_masked_reference.png")
    diptych_ref_tar.save(out_dir / "debug_diptych_input.png")
    mask_diptych.save(out_dir / "debug_diptych_mask.png")
    if args.prepare_only:
        print(out_dir)
        return

    pipe, redux = load_pipelines(args)
    lora_runtime_audit = getattr(pipe, "_in_context_lora_audit", None)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))

    generator_device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(generator_device).manual_seed(args.seed)
    pipe_kwargs = {
        "image": diptych_ref_tar,
        "mask_image": mask_diptych,
        "height": mask_diptych.size[1],
        "width": mask_diptych.size[0],
        "max_sequence_length": args.max_sequence_length,
        "generator": generator,
        **pipe_prior_output,
    }
    if args.num_inference_steps is not None:
        pipe_kwargs["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        pipe_kwargs["guidance_scale"] = args.guidance_scale

    edited_image = pipe(**pipe_kwargs).images[0]
    width, height = edited_image.size
    edited_image = edited_image.crop((width // 2, 0, width, height))

    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, extra_sizes, crop_box)
    edited_image = Image.fromarray(edited_image)

    source_stem = Path(args.source_image).stem
    ref_stem = Path(args.ref_image).stem
    output_path = out_dir / f"{source_stem}_{ref_stem}_seed{args.seed}.png"
    edited_image.save(output_path)

    metadata = {
        "source_image": args.source_image,
        "source_mask": args.source_mask,
        "ref_image": args.ref_image,
        "ref_mask": args.ref_mask,
        "seed": args.seed,
        "nunchaku": args.nunchaku,
        "nunchaku_transformer_path": args.nunchaku_transformer_path if args.nunchaku else None,
        "nunchaku_lora_path": args.nunchaku_lora_path if args.nunchaku else None,
        "nunchaku_lora_strength": args.nunchaku_lora_strength if args.nunchaku else None,
        "flux_fill_path": args.flux_fill_path,
        "flux_redux_path": args.flux_redux_path,
        "lora_path": args.lora_path,
        "lora_weight_name": args.lora_weight_name,
        "lora_runtime_audit": lora_runtime_audit,
        "output_path": str(output_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
