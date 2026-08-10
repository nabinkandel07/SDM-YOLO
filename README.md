# SDM-YOLO: An Improved YOLO Model with Multiscale Attention for Steel Surface Defect Detection
[![Paper](https://img.shields.io/badge/Paper-Engineering%20Research%20Express-blue)](https://iopscience.iop.org/article/10.1088/2631-8695/ae9671)
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2631--8695%2Fae9671-green)](https://doi.org/10.1088/2631-8695/ae9671)
[![Python](https://img.shields.io/badge/Python-3.12.4-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a6158b3-06b8-4893-a246-cc8e9e903fb9" alt="SDM-YOLO" width="800"/>
</p>

---

## 📖 Overview

**SDM-YOLO** is a lightweight and stability-enhanced detection framework for real-time steel surface defect inspection, built upon the YOLO11n architecture. Rather than introducing entirely new architectural primitives, the contribution of this work lies in the **systematic integration and task-specific adaptation of complementary modules** within a unified lightweight framework.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **C2PSA-SEAM** | Self-Ensembling Attention Module for enhanced global-local feature interaction |
| 📐 **DySample** | Content-aware dynamic upsampling for adaptive multi-scale feature alignment |
| 🎯 **MSCA** | Multi-scale convolutional attention for scale-sensitive feature refinement |
| 📊 **NWD Loss** | Normalized Wasserstein Distance loss for improved localization stability |

### Performance Highlights

| Dataset | mAP50 | mAP50:95 | Params | FLOPs | FPS |
|---------|-------|----------|--------|-------|-----|
| **NEU-DET** | **81.0%** | 46.7% | 2.68M | 6.6G | 94.5 |
| **GC10-DET** | **72.3%** | 38.0% | 2.68M | 6.6G | 94.5 |

---

## 📁 Repository Structure

```
SDM-YOLO/
├── configs/                      # Model configuration files
│   ├── sdm_yolo.yaml            # SDM-YOLO architecture config
│   └── yolov11n.yaml            # Baseline YOLO11n config
├── data/                        # Dataset files
│   ├── neudet/                  # NEU-DET dataset
│   │   ├── images/              # Image files
│   │   ├── labels/              # Annotation files
│   │   └── data.yaml            # Dataset configuration
│   └── gc10det/                 # GC10-DET dataset
│       ├── images/              # Image files
│       ├── labels/              # Annotation files
│       └── data.yaml            # Dataset configuration
├── scripts/                     # Training and evaluation scripts
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   └── multi_seed_train.py      # Multi-run training script
├── weights/                     # Pretrained model weights
│   ├── sdm_yolo_best.pt         # Best SDM-YOLO weights
│   └── yolov11n_best.pt         # Baseline YOLO11n weights
├── runs/                        # Training logs and outputs
├── requirements.txt             # Python dependencies
├── LICENSE.txt                  # MIT License
└── README.md                    # This file
```

---

## 📊 Datasets

### 1. NEU-DET Dataset

Steel surface defect benchmark released by Northeastern University.

| Category | Abbreviation | Images |
|----------|--------------|--------|
| Crazing | Cr | 300 |
| Inclusion | In | 300 |
| Patches | Pa | 300 |
| Pitted Surface | Ps | 300 |
| Rolled-in Scale | Rs | 300 |
| Scratches | Sc | 300 |

**Dataset Statistics:**
- Total Images: **1,800**
- Image Resolution: **200 × 200**
- Training/Validation Split: **90% / 10%** (1,620 / 180)

**Official Source:** [NEU-DET Dataset](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)

**Citation:**
```bibtex
@article{he2020end,
  title={An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features},
  author={He, Yongfeng and Song, Kechen and Meng, Qingguang and Yan, Yunhui},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={69},
  number={4},
  pages={1493--1504},
  year={2020}
}
```

---

### 2. GC10-DET Dataset

Large-scale industrial steel defect dataset.

| Category | Abbreviation |
|----------|--------------|
| Punching Hole | Pu |
| Welding Line | Wl |
| Crescent Gap | Cg |
| Water Spot | Ws |
| Oil Spot | Os |
| Silk Spot | Ss |
| Inclusion | In |
| Rolling Pit | Rp |
| Crease | Cr |
| Waist Folding | Wf |

**Dataset Statistics:**
- Total Images: **3,563**
- Original Resolution: **2048 × 1000**
- Training Resolution: **256 × 256**
- Training/Validation Split: **80% / 20%** (2,850 / 713)

**Official Source:** [GC10-DET Dataset](https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Dataset)

**Citation:**
```bibtex
@article{lv2020deep,
  title={Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network},
  author={Lv, Xiaoming and Duan, Fuchang and Jiang, Jinjun and Fu, Xiang and Gan, Lin},
  journal={Sensors},
  volume={20},
  number={6},
  pages={1562},
  year={2020}
}
```

---

### Dataset Split Files

Dataset split information is available in the repository:

| Dataset | Training | Validation | File |
|---------|----------|------------|------|
| NEU-DET | 1,620 | 180 | `data/neudet/train.txt`, `data/neudet/val.txt` |
| GC10-DET | 2,850 | 713 | `data/gc10det/train.txt`, `data/gc10det/val.txt` |

---

## 🚀 Installation

### Requirements

- Python 3.12.4+
- PyTorch 2.7.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/nabinkandel07/SDM-YOLO.git
cd SDM-YOLO

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.7.0
torchvision>=0.18.0
ultralytics>=8.3.0
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## 🏋️ Training

### Single Run Training

```bash
python scripts/train.py --config configs/sdm_yolo.yaml --data data/neudet/data.yaml --epochs 300 --batch 16 --imgsz 256 --seed 42
```

### Multi-Run Training (5 runs with different seeds)

```bash
python scripts/multi_seed_train.py --config configs/sdm_yolo.yaml --data data/neudet/data.yaml --epochs 300 --batch 16 --imgsz 256
```

### Using Ultralytics CLI

```bash
yolo detect train model=configs/sdm_yolo.yaml data=data/neudet/data.yaml epochs=300 imgsz=256 batch=16 seed=42
```

---

## 📊 Evaluation

### Evaluate on Validation Set

```bash
python scripts/eval.py --weights weights/sdm_yolo_best.pt --data data/neudet/data.yaml --imgsz 256
```

### Using Ultralytics CLI

```bash
yolo detect val model=weights/sdm_yolo_best.pt data=data/neudet/data.yaml imgsz=256
```

---

## 📁 Data Availability

The complete implementation code, model configuration files, dataset partition details, training settings, and evaluation scripts are publicly available.

### Baidu Netdisk

```
Link: https://pan.baidu.com/s/1ONCQLBAEUMKJbXW1PSF13A
Password: nk26
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a6158b3-06b8-4893-a246-cc8e9e903fb9" alt="Baidu Netdisk QR Code" width="300"/>
</p>

### Google Drive (Alternative)

```
[Link will be added after publication]
```

---

## 📈 Results

### NEU-DET Dataset

| Method | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) | FPS |
|--------|-----------|--------------|------------|-----------|-----|
| SSD | 73.6 | 32.2 | 25.1 | 88.2 | 37.7 |
| RetinaNet | 73.8 | 33.8 | 31.6 | 10.8 | 41.2 |
| Faster R-CNN | 72.9 | 44.6 | 138.4 | 368.2 | 37.0 |
| CenterNet | 75.2 | 39.9 | 32.7 | 109.3 | 60.5 |
| YOLOv3n | 73.3 | 45.3 | 61.5 | 154.6 | 49.0 |
| YOLOv5n | 73.3 | 41.9 | 1.76 | 4.2 | 204.7 |
| YOLOv6n | 74.6 | 42.9 | 16.3 | 44.2 | 134.4 |
| YOLOv8n | 78.6 | 48.1 | 3.00 | 8.1 | 154.9 |
| YOLOv9s | 78.8 | 44.4 | 7.1 | 26.2 | 43.0 |
| YOLOX | 76.8 | 44.7 | 54.2 | 155.7 | 58.1 |
| YOLOv10n | 75.7 | 45.8 | 2.69 | 8.2 | 93.2 |
| YOLOv11n | 72.7 | 42.0 | 2.58 | 8.4 | 107.3 |
| YOLOv12n | 79.0 | 40.6 | 9.1 | 19.2 | 66.5 |
| YOLOv13n | 78.5 | 41.5 | 9.0 | 20.7 | 29.8 |
| **SDM-YOLO (Ours)** | **81.0** | **46.7** | **2.68** | **6.6** | **94.5** |

### GC10-DET Dataset

| Method | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) |
|--------|-----------|--------------|------------|-----------|
| SSD | 67.4 | 30.3 | 25.1 | 88.2 |
| RetinaNet | 67.5 | 34.1 | 31.6 | 10.8 |
| Faster R-CNN | 69.1 | 22.7 | 138.4 | 368.2 |
| CenterNet | 68.2 | 33.5 | 32.7 | 109.3 |
| YOLOv3n | 60.6 | 31.5 | 61.5 | 154.6 |
| YOLOv5n | 68.5 | 32.1 | 1.76 | 4.2 |
| YOLOv6n | 69.5 | 31.6 | 16.3 | 44.2 |
| YOLOv8n | 68.5 | 30.9 | 3.00 | 8.1 |
| YOLOv9s | 70.6 | 31.0 | 7.1 | 26.2 |
| YOLOv10n | 67.7 | 31.6 | 2.69 | 8.2 |
| YOLOv11n | 66.2 | 32.5 | 2.58 | 8.4 |
| YOLOv12n | 70.6 | 32.4 | 9.1 | 19.2 |
| ARYOLO | 71.2 | - | - | - |
| FPDNet | 66.8 | - | - | - |
| **SDM-YOLO (Ours)** | **72.3** | **38.0** | **2.68** | **6.6** |

### Ablation Study (NEU-DET)

| Method | Params (M) | GFLOPs | mAP50 (%) |
|--------|------------|--------|-----------|
| Baseline (YOLOv11n) | 2.58 | 8.40 | 72.7 |
| Baseline + NWD | 2.58 | 8.40 | 73.9 |
| Baseline + C2PSA-SEAM | 4.87 | 7.60 | 75.0 |
| Baseline + DySample | 4.91 | 7.00 | 76.5 |
| Baseline + C2PSA-SEAM + DySample | 5.87 | 6.60 | 77.3 |
| Baseline + MSCA | 5.56 | 7.88 | 78.8 |
| Naive Stacking (All modules) | 7.92 | 11.26 | 79.0 |
| **SDM-YOLO (Coordinated Integration)** | **2.68** | **6.6** | **81.0** |

### Multi-Run Validation

| Dataset | Seeds | Mean mAP50 | Std Dev |
|---------|-------|------------|---------|
| NEU-DET | 0, 42, 100, 2025, 3407 | **81.00%** | 0.11% |
| GC10-DET | 0, 42, 100, 2025, 3407 | **72.30%** | 0.16% |

### Resolution Comparison (GC10-DET)

| Input Resolution | mAP50 (%) | mAP50:95 (%) |
|------------------|-----------|--------------|
| 256 × 256 | **72.3** | 38.0 |
| 640 × 640 | 72.2 | **38.1** |

---

## 🖼️ Sample Detection Results

### NEU-DET Dataset

<p align="center">
  <img width="3413" height="1920" alt="image" src="https://github.com/user-attachments/assets/8020219e-d14b-4b0d-b1cc-afdb9641a696" />

</p>

### GC10-DET Dataset

<p align="center">
  <img <img width="1470" height="660" alt="gc10-det" src="https://github.com/user-attachments/assets/e90c7381-a48a-4d5a-a8a7-f0c971d051a6" />
</p>

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{kandel2026sdmyolo,
  author = {Kandel, Nabin and Wu, Ping},
  title = {SDM-YOLO: An Improved YOLO Model with Multiscale Attention for Steel Surface Defect Detection},
  journal = {Engineering Research Express},
  year = {2026},
  doi = {10.1088/2631-8695/ae9671},
  url = {https://iopscience.iop.org/article/10.1088/2631-8695/ae9671}
}
```

---

## 📧 Contact

| Author | Role | Email |
|--------|------|-------|
| **Ping Wu** | Corresponding Author | pingwu@zstu.edu.cn |
| **Nabin Kandel** | First Author | nabinkandel60@gmail.com |

**Affiliation:**
School of Information Science and Engineering
Zhejiang Sci-Tech University
Hangzhou, China

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🙏 Acknowledgments

This work was supported by:

- "Pioneer" and "Leading Goose" R\&D Program of Zhejiang (Grant **2026C02A3004**)
- National Natural Science Foundation of China (Grant **62573387**)
- Natural Science Foundation of Zhejiang Province (Grant **LY24F030004**)
- Fundamental Research Funds of Zhejiang Sci-Tech University (**25222139-Y**)

---

## ⭐ Star History

If you find this repository useful, please consider giving it a star ⭐

---

## 📚 References

1. He, Y., Song, K., Meng, Q., \& Yan, Y. (2020). An End-to-End Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features. *IEEE Transactions on Instrumentation and Measurement*, 69(4), 1493-1504.
2. Lv, X., Duan, F., Jiang, J., Fu, X., \& Gan, L. (2020). Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network. *Sensors*, 20(6), 1562.
3. Khanam, R., \& Hussain, M. (2024). YOLOv11: An Overview of the Key Architectural Enhancements. *arXiv preprint*.
4. Liu, W., et al. (2016). SSD: Single Shot MultiBox Detector. *ECCV*.
5. Ren, S., et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *IEEE TPAMI*.

---

## 🔄 Updates

- **2026**: Initial release with paper publication
- Code, weights, and datasets will be continuously updated

---

**Note:** This repository will be made fully public upon paper acceptance. For review purposes, please contact the authors for access.

---

*Built with ❤️ by Nabin Kandel*
```
