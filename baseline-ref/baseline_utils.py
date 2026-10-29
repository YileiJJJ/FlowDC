"""Dataset/CLI helpers; numerical algorithms live in the bundled pipelines."""

import argparse
from pathlib import Path
import random


def dataset_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--image_root", type=Path, help="Prefix for relative init_img paths.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_num", type=int, default=1000)
    parser.add_argument("--generate_from_head", action="store_true", help="Overwrite existing images.")
    return parser


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(args):
    import yaml

    if args.max_num < 0:
        raise ValueError("max_num must be nonnegative")
    with open(args.dataset_path, encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)
    if not isinstance(dataset, list):
        raise ValueError("Dataset YAML must contain a list of samples")
    return dataset[:args.max_num]


def sample_paths(data, args):
    image_path = Path(data["init_img"])
    if not image_path.is_absolute() and args.image_root is not None:
        image_path = args.image_root / image_path
    prompts = data["target_prompts"]
    if not prompts:
        raise ValueError(f"No target prompts for {image_path}")
    # Released YAMLs omit target_codes/init_img_name; use the evaluator's naming.
    target_codes = data.get("target_codes")
    target_name = str(target_codes[-1]) if target_codes else f"0{len(prompts) - 1}"
    image_name = str(data.get("init_img_name") or image_path.stem)
    output_path = args.output_dir / image_name / f"{target_name}.png"
    return image_path, output_path, prompts[-1]


def save_parameters(args, parameters):
    import yaml

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.output_dir / "Edit_parameters.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({**config, **parameters}, handle, sort_keys=False)


def run_inversion_baseline(method, default_invert_tmax):
    parser = dataset_parser(f"{method}: original FlowDC baseline reference")
    parser.add_argument("--model_path", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--num_inversion_steps", type=int, default=28)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--invert_tmax", type=int, default=default_invert_tmax)
    parser.add_argument("--tar_guidance_scale", type=float, default=5.5)
    args = parser.parse_args()
    if not 1 <= args.invert_tmax <= min(args.num_inversion_steps, args.num_inference_steps):
        parser.error("invert_tmax must be positive and no greater than either step count")

    import torch
    from PIL import Image
    from pipeline_flux_baseline_inversion_base import RFInversionFluxPipeline

    dataset = load_dataset(args)
    pipe = RFInversionFluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    parameters = dict(
        method=method, gamma=0, eta=0, sde_choose="default", enable_sde=False,
        start_timestep=0, stop_timestep=7 / 28, decay_eta=False, eta_decay_power=1.0,
        n_max=args.invert_tmax, height=1024, width=1024, vae_encoding="sample",
    )
    save_parameters(args, parameters)
    invert = pipe.invert if method == "ODE-Inv" else pipe.sde_invert

    for index, data in enumerate(dataset):
        image_path, output_path, target_prompt = sample_paths(data, args)
        if output_path.exists() and not args.generate_from_head:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{index + 1}/{len(dataset)}: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((1024, 1024))
        seed_everything(args.seed)
        inverted_latents, image_latents, latent_image_ids = invert(
            image=image, num_inversion_steps=args.num_inversion_steps,
            gamma=0, invert_tmax=args.invert_tmax,
        )
        seed_everything(args.seed)
        edited_image = pipe(
            enable_sde=False, prompt=target_prompt, inverted_latents=inverted_latents,
            image_latents=image_latents, latent_image_ids=latent_image_ids,
            sde_choose="default", start_timestep=0, stop_timestep=7 / 28,
            num_inference_steps=args.num_inference_steps, eta=0, decay_eta=False,
            eta_decay_power=1.0, guidance_scale=args.tar_guidance_scale,
            n_max=args.invert_tmax,
        ).images[0]
        edited_image.save(output_path)
        del edited_image, inverted_latents, image_latents, latent_image_ids
        torch.cuda.empty_cache()
