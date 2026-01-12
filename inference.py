import torch
import argparse
import requests
import PIL
import os
import yaml
import random 
import numpy as np
from tqdm import tqdm
from io import BytesIO
from diffusers import FluxPipeline


# set seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def FlowDC(pipe, image_path, src_prompt, tar_prompts, save_root, seed, FlowDC_param):
    image = PIL.Image.open(image_path).convert("RGB").resize((1024,1024))
    
    target_names = [ f'0{k}' for k in range(len(tar_prompts))]
    init_img_name = image_path.split('/')[-1].split('.')[0]
    prompt_choose = [ list(range(k+1)) for k in range(len(tar_prompts))]

    save_folder = os.path.join(save_root, init_img_name)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_root, 'FlowDC_parameters.yaml'), 'w') as f:
        yaml.dump(FlowDC_param, f)
        
    target_name = target_names[-1]
    image_save_path = os.path.join(save_folder, target_name+'.png')
    print('-' * 50)
    print(f'Editing for: {init_img_name}')

    with torch.no_grad():
        (   
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipe.encode_prompt(
            prompt=[src_prompt]*len(tar_prompts),
            prompt_2=None,
            device=pipe._execution_device,
        )
        src_prompt_dict = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }
        
    seed_everything(seed)
    edited_image = pipe(
        src_image=image,
        prompt=tar_prompts,
        prompt_choose = prompt_choose,
        source_prompt_dict=src_prompt_dict,
        num_inference_steps=FlowDC_param['num_inference_steps'],
        total_step=FlowDC_param['total_step'],
        guide_step=FlowDC_param['guide_step'],
        orthogonal_step=FlowDC_param['orthogonal_step'],
        decay_step=FlowDC_param['decay_step'],
        lambda_1=FlowDC_param['lambda_1'],
        lambda_d=FlowDC_param['lambda_d'],
        src_guidance_scale=FlowDC_param['src_guidance_scale'],
        tar_guidance_scale=FlowDC_param['tar_guidance_scale'],
        return_dict=False,
    )[-1]
        
    target_name = target_names[-1]
    image_save_path = os.path.join(save_folder, target_name+'.png')

    edited_image.save(image_save_path)
    print('Results saved here: ', image_save_path)
    del edited_image, prompt_embeds, pooled_prompt_embeds, text_ids, src_prompt_dict
    torch.cuda.empty_cache()



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

    # Example inputs and outputs
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to the source image to edit."
    )
    parser.add_argument(
        "--src_prompt", 
        type=str, 
        required=True,
        help="Text prompt describing the source image."
    )
    parser.add_argument(
        "--prompts", 
        type=str,           
        nargs='+',          
        required=True,
        help="Text prompt(s) describing the target edit."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Directory to save the results."
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility."
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
        default=0.64,
        help="Hyperparameter for FlowDC decay lambda."
    )
    
    args, unknown = parser.parse_known_args()
    return args

def main():
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
        'num_inference_steps': 28,
        'src_guidance_scale': 1.5,
        'tar_guidance_scale': 5.5,
    }
    
    FlowDC(pipe, args.image_path, args.src_prompt, 
           args.prompts, args.output_dir, args.seed, 
           FlowDC_param)
    

if __name__ == "__main__":
    main()
