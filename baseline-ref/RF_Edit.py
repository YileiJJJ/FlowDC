"""RF-Edit, adapted from RF-Solver-Edit's local rf_solver_edit_baseline.py."""

from baseline_utils import dataset_parser, load_dataset, sample_paths, save_parameters, seed_everything


def main():
    parser = dataset_parser("RF-Edit: original FlowDC baseline reference")
    parser.add_argument("--model_path", default="black-forest-labs/FLUX.1-dev",
                        help="Hugging Face repo for original single-file FLUX weights.")
    parser.add_argument("--flux_checkpoint", help="Local flux1-dev.safetensors (or FLUX_DEV env var).")
    parser.add_argument("--ae_checkpoint", help="Local ae.safetensors (or AE env var).")
    parser.add_argument("--t5_path", default="google/t5-v1_1-xxl")
    parser.add_argument("--clip_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--num_steps", type=int, default=30, help="Intervals per direction; 2 model calls per interval.")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--inject", type=int, default=2)
    parser.add_argument("--resize", type=int, help="Optional square resize; default preserves native image dimensions before cropping.")
    args = parser.parse_args()
    if args.num_steps < 1 or not 0 <= args.inject <= args.num_steps:
        parser.error("num_steps must be positive; inject must lie in [0, num_steps]")
    if args.resize is not None and (args.resize < 16 or args.resize % 16):
        parser.error("resize must be a positive multiple of 16")

    import numpy as np
    import torch
    from PIL import Image
    from einops import rearrange
    from rf_edit_flux.sampling import denoise, get_schedule, prepare, unpack
    from rf_edit_flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

    dataset = load_dataset(args)
    config = configs["flux-dev"]
    config.repo_id = args.model_path
    if args.flux_checkpoint:
        config.ckpt_path = args.flux_checkpoint
    if args.ae_checkpoint:
        config.ae_path = args.ae_checkpoint
    device = torch.device(args.device)
    with torch.inference_mode():
        t5 = load_t5(device, max_length=512, model_path=args.t5_path)
        clip = load_clip(device, model_path=args.clip_path)
        model = load_flow_model("flux-dev", device=device)
        ae = load_ae("flux-dev", device=device)
        save_parameters(args, dict(
            method="RF-Edit", inversion_guidance=1, nfe_inversion=2 * args.num_steps,
            nfe_editing=2 * args.num_steps, nfe_total=4 * args.num_steps,
            vae_encoding="sample", watermark=True,
            flux_checkpoint=config.ckpt_path, ae_checkpoint=config.ae_path,
        ))
        for index, data in enumerate(dataset):
            image_path, output_path, target_prompt = sample_paths(data, args)
            if output_path.exists() and not args.generate_from_head:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{index + 1}/{len(dataset)}: {image_path}")
            seed_everything(args.seed)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                if args.resize is not None:
                    image = image.resize((args.resize, args.resize))
                pixels = np.array(image)
            height, width = pixels.shape[:2]
            height, width = height - height % 16, width - width % 16
            if min(height, width) < 16:
                raise ValueError(f"Image must be at least 16 x 16: {image_path}")
            pixels = pixels[:height, :width]
            image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1
            image_latents = ae.encode(image_tensor.unsqueeze(0).to(device)).to(torch.bfloat16)
            inp = prepare(t5, clip, image_latents, prompt=data["source_prompt"])
            inp_target = prepare(t5, clip, image_latents, prompt=target_prompt)
            timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=True)
            info = {"feature": {}, "inject_step": args.inject}
            seed_everything(args.seed)
            z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            inp_target["img"] = z
            timesteps = get_schedule(args.num_steps, inp_target["img"].shape[1], shift=True)
            x, _ = denoise(model, **inp_target, timesteps=timesteps,
                           guidance=args.guidance, inverse=False, info=info)
            # Use actual height/width here; the original runner swapped these for non-square images.
            batch_x = unpack(x.float(), height, width)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                decoded = ae.decode(batch_x)
            decoded = embed_watermark(decoded.clamp(-1, 1).float())
            pixels = rearrange(decoded[0], "c h w -> h w c")
            Image.fromarray((127.5 * (pixels + 1)).cpu().byte().numpy()).save(output_path)
            del inp, inp_target, info, z, x, batch_x, decoded, image_tensor, image_latents
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
