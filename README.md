
# FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.11395)
<!-- [![Project Page](https://img.shields.io/badge/Project-Website-blue)](YOUR_PROJECT_PAGE_LINK)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-yellow)](YOUR_HF_DEMO_LINK) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation of the paper: "FlowDC: Flow-Based Decoupling-Decay for Complex Image Editing"**

</div>

---

## 📖 Methodology

![Teaser Image](./assets/teaser.png)


## 📅 TODO & Roadmap

- [x] Release inference code.
- [ ] Release Complex-PIE-Bench.
- [ ] Add Project Page.

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
