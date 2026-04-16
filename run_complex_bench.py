
import os
from PIL import Image
import yaml
import torch
import argparse
from diffusers import FluxPipeline
from inference import FlowDC



def parse_args():
    parser = argparse.ArgumentParser(description="FlowDC Inference Script")
    
    # model and pipeline
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="black-forest-labs/FLUX.1-dev",
        help="Path to the pretrained FLUX model."
    )
    parser.add_argument(
        "--custom_pipeline", 
        type=str, 
        default="./pipeline_flowdc.py",
        help="Path to the custom pipeline file."
    )

    # Datasets input
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/Complex_PIE_Bench.yaml",
        help="Path to the input dataset for inference."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results/Complex_PIE_Bench",
        help="Directory to save the generated images."
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=1000,
        help="Maximum number of samples to process from the dataset."
    )
    parser.add_argument(
        "--generate_from_head",
        action='store_true',
        help="Whether to generate images from the head of the dataset."
    )
    
    
    # FlowDC Parameters
    parser.add_argument(
        "--total_step", 
        type=int, 
        default=27,
        help="Hyperparameter for FlowDC total timestep."
    )
    parser.add_argument(
        "--guide_step", 
        type=int, 
        default=22,
        help="Hyperparameter for FlowDC guidance timestep."
    )
    parser.add_argument(
        "--orthogonal_step", 
        type=int, 
        default=1,
        help="Hyperparameter for FlowDC orthogonal timestep."
    )
    
    parser.add_argument(
        "--decay_step", 
        type=int, 
        default=20,
        help="Hyperparameter for FlowDC decay timestep."
    )
    parser.add_argument(
        "--lambda_1", 
        type=float, 
        default=0.1,
        help="Hyperparameter for FlowDC decay lambda."
    )
    parser.add_argument(
        "--lambda_d", 
        type=float, 
        default=0.6,
        help="Hyperparameter for FlowDC decay lambda."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility."
    )
    
    args, unknown = parser.parse_known_args()
    return args

def run_dataset(pipe, FlowDC_param, save_root, datasets, max_num=1000, generate_from_head=False):

    for i, data in enumerate(datasets):
        if i >= max_num:
            break
        image_path = data['init_img']
        src_prompt=data['source_prompt']
        tar_prompts = data['target_prompts']
        
        FlowDC(pipe, image_path, src_prompt, tar_prompts, save_root, 
               FlowDC_param['seed'], FlowDC_param, generate_from_head)
    
if __name__ == "__main__":
    args = parse_args()
    
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        custom_pipeline=args.custom_pipeline)
    pipe.to("cuda:0")
    FlowDC_param = {
        'total_step': args.total_step,
        'guide_step': args.guide_step,
        'orthogonal_step': args.orthogonal_step,
        'decay_step': args.decay_step,
        'lambda_1': args.lambda_1,
        'lambda_d': args.lambda_d,
        'seed': args.seed,
        'num_inference_steps': 28,
        'src_guidance_scale': 1.5,
        'tar_guidance_scale': 5.5,
    }
    
    # datasets
    with open(args.dataset_path, 'r') as f:
        dataset = yaml.safe_load(f)
    run_dataset(pipe, FlowDC_param, args.output_dir, dataset, 
                max_num=args.max_num, 
                generate_from_head=args.generate_from_head)
