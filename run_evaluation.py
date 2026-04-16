import os
import yaml
import lpips
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

# -------- CLIP Evaluation --------
@torch.no_grad()
def extract_clip_features(clip_model, image_path=None, text_prompt=None):
    if text_prompt is None:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        image_features = clip_model.get_image_features(**inputs)# torch.Size([1, 768])
        
        return image_features
    else:
        inputs = clip_tokenizer(text_prompt, padding=True, return_tensors="pt").to(device)
        text_features = clip_model.get_text_features(**inputs)
        
        return text_features

def load_clip_embeds(datasets, dataset_root, generated_img_root, index=[-1], max_num=1000):
    print("loading embeddings...")
    clip_t_score = 0
    clip_i_score = 0
    clip_dic = {
        "generated_img_root": generated_img_root,
        "clip_t_score": 0,
        "clip_i_score": 0,
    }
    num = 0
    for i in tqdm(range(len(datasets))):
        if i >= max_num:
            break
        data = datasets[i]
        target_prompts = data['target_prompts']
        target_names = [ f'0{k}' for k in range(len(target_prompts))]
        init_img_name = data['init_img'].split('/')[-1].split('.')[0]
        save_folder = os.path.join(generated_img_root, init_img_name)
        
        ori_clip_i_save_path = f"{dataset_root}/CLIP_Features/{init_img_name}/{init_img_name}_i.pt"
        ori_clip_i_feature = torch.load(ori_clip_i_save_path, map_location=device)
        ori_clip_t_save_path = f"{dataset_root}/CLIP_Features/{init_img_name}/{init_img_name}_t.pt"
        ori_clip_t_feature = torch.load(ori_clip_t_save_path, map_location=device)
        for j in index:
            target_prompt, target_name = target_prompts[j], target_names[j]
            tar_clip_t_save_path = f"{dataset_root}/CLIP_Features/{init_img_name}/{target_name}_t.pt"
            tar_clip_t_feature = torch.load(tar_clip_t_save_path, map_location=device)
            tar_clip_i_save_path = os.path.join(save_folder, target_name+'_clip.pt')
            tar_clip_i_feature = torch.load(tar_clip_i_save_path, map_location=device)

            cos_sim_t = torch.nn.functional.cosine_similarity(tar_clip_t_feature, tar_clip_i_feature, dim=1).cpu().item()
            cos_sim_i = torch.nn.functional.cosine_similarity(ori_clip_i_feature, tar_clip_i_feature, dim=1).cpu().item()
            clip_t_score += cos_sim_t
            clip_i_score += cos_sim_i
            num += 1

    clip_t_score /= num
    clip_i_score /= num
    clip_dic['clip_t_score'] = clip_t_score
    clip_dic['clip_i_score'] = clip_i_score
    print(f"clip_t_score: {clip_t_score}, clip_i_score: {clip_i_score}")
    with open(os.path.join(generated_img_root, f'clip_score.yaml'), 'w') as f:
        yaml.dump(clip_dic, f)
    return clip_t_score, clip_i_score
    
def generate_clip_embeds(datasets, dataset_root, generated_img_root, generate_from_head=False, index = [-1], max_num=1000):
    print("generating embeddings...")
    os.makedirs(f"{dataset_root}/CLIP_Features", exist_ok=True)
    for i in tqdm(range(len(datasets))):
        if i >= max_num:
            break
        data = datasets[i]
        src_prompt = data['source_prompt']
        target_prompts = data['target_prompts']
        target_names = [ f'0{k}' for k in range(len(target_prompts))]
        init_img_name = data['init_img'].split('/')[-1].split('.')[0]
        save_folder = os.path.join(generated_img_root, init_img_name)
        
        # generate clip features of the original image
        ori_clip_i_save_path = f"{dataset_root}/CLIP_Features/{init_img_name}/{init_img_name}_i.pt"
        os.makedirs(f"{dataset_root}/CLIP_Features/{init_img_name}", exist_ok=True)
        if not os.path.exists(ori_clip_i_save_path) or generate_from_head:
            ori_clip_i_features = extract_clip_features(clip_model, image_path=data['init_img'])
            torch.save(ori_clip_i_features, ori_clip_i_save_path)
        
        # generate clip features of the original text prompt
        ori_clip_t_save_path =f"{dataset_root}/CLIP_Features/{init_img_name}/{init_img_name}_t.pt"
        if not os.path.exists(ori_clip_t_save_path) or generate_from_head:
            ori_clip_t_features = extract_clip_features(clip_model, text_prompt=src_prompt)
            torch.save(ori_clip_t_features, ori_clip_t_save_path)
            
        
        # generate clip features of the target text prompts and generated images
        for j in index:
            target_prompt, target_name = target_prompts[j], target_names[j]
            tar_clip_t_save_path =f"{dataset_root}/CLIP_Features/{init_img_name}/{target_name}_t.pt"
            if not os.path.exists(tar_clip_t_save_path) or generate_from_head:
                tar_clip_t_features = extract_clip_features(clip_model, text_prompt=target_prompt)
                torch.save(tar_clip_t_features, tar_clip_t_save_path)
            
            tar_clip_i_save_path = os.path.join(save_folder, target_name+'_clip.pt')
            if not os.path.exists(tar_clip_i_save_path) or generate_from_head:
                tar_clip_i_features = extract_clip_features(clip_model, image_path=os.path.join(save_folder, target_name+'.png'))
                torch.save(tar_clip_i_features, tar_clip_i_save_path)
        
def clip_evaluation(datasets, dataset_root="dataset/Complex_PIE_Bench", generated_img_root="results/Complex_PIE_Bench", generate_from_head=False, index=[-1], max_num=1000):
    print("-"*30)
    print(f"CLIP Evaluating generated images under {generated_img_root}...")
    generate_clip_embeds(datasets, dataset_root, generated_img_root, generate_from_head, index, max_num)
    clip_t_score, clip_i_score = load_clip_embeds(datasets, dataset_root, generated_img_root, index, max_num)
    return clip_t_score, clip_i_score



# -------- LPIPS Evaluation --------
def load_image_lpips(path):
    target_size = (512, 512)
    
    if path[-3:].lower() == 'dng':
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif path[-3:].lower() in ['bmp', 'jpg', 'png'] or path[-4:].lower() == 'jpeg':
        import cv2
        img = cv2.imread(path)[:,:,::-1]
    else:
        import matplotlib.pyplot as plt
        img = (255 * plt.imread(path)[:,:,:3]).astype('uint8')
    
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return resized_img

def lpips_evaluation(datasets, generated_img_root="results/Complex_PIE_Bench", index=[-1], max_num=1000):
    print("-"*30)
    print(f"LPIPS Evaluating generated images under {generated_img_root}...")
    lpips_score = 0
    lpips_dic = {
        "generated_img_root": generated_img_root,
        "lpips_score": 0,
    }
    num = 0
    for i in tqdm(range(len(datasets))):
        if i >= max_num:
            break
        data = datasets[i]
        target_prompts = data['target_prompts']
        target_names = [ f'0{k}' for k in range(len(target_prompts))]
        init_img_name = data['init_img'].split('/')[-1].split('.')[0]
        save_folder = os.path.join(generated_img_root, init_img_name)
        
        ori_img = lpips.im2tensor(load_image_lpips(data['init_img'])).to(device)
        for j in index:
            target_prompt, target_name = target_prompts[j], target_names[j]
            tar_img_path = os.path.join(save_folder, target_name+'.png')
            tar_img = lpips.im2tensor(load_image_lpips(tar_img_path)).to(device)
            d = loss_fn_alex(ori_img, tar_img).cpu().item()
            lpips_score += d
            num += 1
            
    
    print("average lpips: ", lpips_score/num)
    lpips_dic['lpips_score'] = lpips_score/num
    with open(os.path.join(generated_img_root, f'lpips_score.yaml'), 'w') as f:
        yaml.dump(lpips_dic, f)
    return lpips_dic['lpips_score']


# -------- DINO Evaluation --------
@torch.no_grad()
def extract_dino_features(model, samples):
    feats = model(samples)
    return feats

  
def load_dino_embeds(datasets, dataset_root, generated_img_root, index = [-1], max_num=1000):
    print("loading embeddings...")
    dino_score = 0
    dino_dic = {
        "generated_img_root": generated_img_root,
        "dino_score": 0,
    }
    num = 0
    for i in tqdm(range(len(datasets))):
        if i >= max_num:
            break
        data = datasets[i]
        target_prompts = data['target_prompts']
        target_names = [ f'0{k}' for k in range(len(target_prompts))]
        init_img_name = data['init_img'].split('/')[-1].split('.')[0]
        save_folder = os.path.join(generated_img_root, init_img_name)
        
        ori_dino_save_path = f"{dataset_root}/DINO_Features/{init_img_name}.pt"
        ori_dino_feature = torch.load(ori_dino_save_path).to(device)

        for j in index:
            target_prompt, target_name = target_prompts[j], target_names[j]
            tar_dino_save_path = os.path.join(save_folder, target_name+'_dino.pt')
            tar_dino_feature = torch.load(tar_dino_save_path).to(device)
            cos_sim = torch.nn.functional.cosine_similarity(ori_dino_feature, tar_dino_feature, dim=1).cpu().item()
            dino_score += cos_sim
            num += 1
            
    dino_score /= num
    dino_dic['dino_score'] = dino_score
    print(f"dino_score: {dino_score}")
    with open(os.path.join(generated_img_root, f'dino_score.yaml'), 'w') as f:
        yaml.dump(dino_dic, f)
    return dino_score

def generate_dino_embeds(datasets, dataset_root, generated_img_root, generate_from_head=False, index = [-1], max_num=1000):
    print("generating embeddings...")
    os.makedirs(f"{dataset_root}/DINO_Features", exist_ok=True)
    for i in tqdm(range(len(datasets))):
        if i >= max_num:
            break
        data = datasets[i]
        target_prompts = data['target_prompts']
        target_names = [ f'0{k}' for k in range(len(target_prompts))]
        init_img_name = data['init_img'].split('/')[-1].split('.')[0]
        save_folder = os.path.join(generated_img_root, init_img_name)
        
        # generate dino features of the original image
        ori_dino_save_path = f"{dataset_root}/DINO_Features/{init_img_name}.pt"
        if not os.path.exists(ori_dino_save_path) or generate_from_head:
            ori_img_path = data['init_img']
            ori_img = Image.open(ori_img_path).convert('RGB')
            ori_tensor = dino_preprocess(ori_img).unsqueeze(dim=0).to(device)
            ori_dino_features = extract_dino_features(dinov2_vitl14_reg, ori_tensor)
            torch.save(ori_dino_features, ori_dino_save_path)
        
        # generate dino features of the target generated images
        for j in index:
            target_prompt, target_name = target_prompts[j], target_names[j]
            tar_dino_save_path = os.path.join(save_folder, target_name+'_dino.pt')
            if not os.path.exists(tar_dino_save_path) or generate_from_head:
                tar_img_path = os.path.join(save_folder, target_name+'.png')
                tar_image = Image.open(tar_img_path).convert('RGB')
                tar_tensor = dino_preprocess(tar_image).unsqueeze(dim=0).to(device)
                tar_dino_features = extract_dino_features(dinov2_vitl14_reg, tar_tensor)
                torch.save(tar_dino_features, tar_dino_save_path)

def dino_evaluation(datasets, dataset_root="dataset/Complex_PIE_Bench", generated_img_root="results/Complex_PIE_Bench", generate_from_head=False, index=[-1], max_num=1000):
    print("-"*30)
    print(f"DINO Evaluating generated images under {generated_img_root}...")
    generate_dino_embeds(datasets, dataset_root, generated_img_root, generate_from_head, index, max_num)
    dino_score = load_dino_embeds(datasets, dataset_root, generated_img_root, index, max_num)
    return dino_score


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation for Generated Images")
    # load evaluation model
    parser.add_argument("--clip_model_path", type=str,
        default="clip-vit-large-patch14",
        help="Path to the pre-trained CLIP model."
    )
    parser.add_argument("--dino_model_dir", type=str,
        default="dinov2",
        help="Dir of the pre-trained DINO model."
    )
    parser.add_argument("--dataset_path", type=str, 
        default="dataset/Complex_PIE_Bench.yaml",
        help="Path to the input dataset for inference."
    )
    parser.add_argument("--dataset_root", type=str, 
        default="dataset/Complex_PIE_Bench", 
        help="root path for generated features of the original images and text prompts"
    )
    parser.add_argument("--generated_img_root", type=str, 
        default="results/Complex_PIE_Bench", 
        help="root path of the generated images"
    )
    parser.add_argument("--generate_from_head", action='store_true', 
        help="whether to generate embeddings from head"
    )
    parser.add_argument("--index", nargs='+', type=int, 
        default=[-1], 
        help="the index of target prompts to evaluate"
    )
    parser.add_argument("--max_num", type=int, 
        default=1000, 
        help="the max number of samples to evaluate"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda:0")

    # load CLIP model and processor
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path)

    # load LPIPS model
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores

    # load DINO model
    dinov2_vitl14_reg = torch.hub.load(args.dino_model_dir, 'dinov2_vitl14_reg', source='local')
    dinov2_vitl14_reg.eval()
    dinov2_vitl14_reg.to(device)

    dino_preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    
    with open(args.dataset_path, 'r') as f:
        datasets = yaml.safe_load(f)
    
    if args.dataset_root.split('/')[-1] != args.generated_img_root.split('/')[-1]:
        print("Warning: the name of dataset_root and generated_img_root is different, please make sure you know which one you are evaluating.")     
        
    clip_t_score, clip_i_score = clip_evaluation(datasets, args.dataset_root, args.generated_img_root, args.generate_from_head, args.index, args.max_num)
    lpips_score = lpips_evaluation(datasets, args.generated_img_root, args.index, args.max_num)
    dino_score = dino_evaluation(datasets, args.dataset_root, args.generated_img_root, args.generate_from_head, args.index, args.max_num)
    
            