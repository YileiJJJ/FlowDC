
# [CVPR 2026 Highlight] FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing


[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.11395) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation of the paper: "FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing"**

</div>

---

## 📖 Methodology

![Teaser Image](./assets/teaser.png)


## 📅 TODO & Roadmap

- [x] Release inference code.
- [x] Release Complex-PIE-Bench.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YileiJJJ/FlowDC.git
cd FlowDC
```

### 2. Create a Conda Environment
We recommend using **Anaconda** or **Miniconda** to manage dependencies.

```bash
conda create -n flowdc python=3.10
conda activate flowdc
```

### 3. Install Dependencies
Install PyTorch (adjust the CUDA version according to your hardware):
```bash
# Example for CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

Install or upgrade diffusers:
```bash
pip install -r requirements.txt
```



## 🚀 Usage

### Inference (Image Editing)

To edit an image, run the following command:

```bash
python inference.py \
  --image_path "test/cat_dog_car.png" \
  --src_prompt "A blue-gray car parked in a grassy area. A white dog sitting on the grass, next to the car. A cat laying on the hood of the car." \
  --prompts "A blue-gray car parked in a grassy area. A white dog sitting on the grass, next to the car." \
            "A blue-gray car parked in a grassy area. A white dog sitting on the grass, next to the car. A ball is on the grass." \
            "A blue-gray car parked in a grassy area. A white wolf sitting on the grass, next to the car. A ball is on the grass." \
            "A red car parked in a grassy area. A white wolf sitting on the grass, next to the car. A ball is on the grass." \
  --output_dir "results/test" \
  --seed 42
```

**Arguments:**
- `--image_path`: Path to the source image.
- `--src_prompt`: Text prompt describing the source image.
- `--prompts`: Text prompt(s) describing the target edit.
- `--output_dir`: Saving dir for editing image .
- `--seed`: Seed for image editing.

More examples in `run_script.sh`

### Evaluation of Complex PIE-Bench Dataset

**1. Preparation** First, download the original PIE-Bench dataset from [here](https://github.com/cure-lab/PnPInversion) and update the image directory path in `dataset/Complex_PIE_Bench.yaml`.

**2. Generation** To generate the edited results, run the following command:

```bash
python run_complex_bench.py \
    --model_path black-forest-labs/FLUX.1-dev
```

**3. Evaluation** To evaluate the generated results, run the evaluation script. *(Note: Please ensure the model paths for CLIP and DINOv2 match your local environment.)*

```bash
python run_evaluation.py \
    --dataset_path dataset/Complex_PIE_Bench.yaml \
    --dataset_root dataset/Complex_PIE_Bench \
    --generated_img_root results/Complex_PIE_Bench \
    --clip_model_path /path/to/your/clip-vit-large-patch14 \
    --dino_model_dir /path/to/your/dinov2
```

## 🔗 Citation

If you find our code or paper useful for your research, please consider citing:

```bibtex
@misc{jiang2025flowdcflowbaseddecouplingdecaycomplex,
      title={FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing}, 
      author={Yilei Jiang and Zhen Wang and Yanghao Wang and Jun Yu and Yueting Zhuang and Jun Xiao and Long Chen},
      year={2025},
      eprint={2512.11395},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.11395}, 
}
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
