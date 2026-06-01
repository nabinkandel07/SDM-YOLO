# SDM-YOLO
# Dataset

The proposed SDM-YOLO model was evaluated on two publicly available steel surface defect datasets.

## 1. NEU-DET Dataset

The NEU-DET dataset is a benchmark steel surface defect dataset released by Northeastern University. It contains six common defect categories:

* Crazing (Cr)
* Inclusion (In)
* Patches (Pa)
* Pitted Surface (Ps)
* Rolled-in Scale (Rs)
* Scratches (Sc)

Dataset statistics:

| Item              | Value     |
| ----------------- | --------- |
| Number of Classes | 6         |
| Images per Class  | 300       |
| Total Images      | 1800      |
| Image Resolution  | 200 × 200 |

Official Dataset Source:

http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

Citation:

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

## 2. GC10-DET Dataset

GC10-DET is a large-scale industrial steel defect dataset containing ten categories of metallic surface defects.

Defect Categories:

* Punching Hole (Pu)
* Welding Line (Wl)
* Crescent Gap (Cg)
* Water Spot (Ws)
* Oil Spot (Os)
* Silk Spot (Ss)
* Inclusion (In)
* Rolling Pit (Rp)
* Crease (Cr)
* Waist Folding (Wf)

Dataset Statistics:

| Item                | Value       |
| ------------------- | ----------- |
| Number of Classes   | 10          |
| Total Images        | 3563        |
| Original Resolution | 2048 × 1000 |

Official Dataset Source:

https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Dataset

Citation:

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

## Dataset Split

### NEU-DET

| Set        | Images |
| ---------- | ------ |
| Training   | 1620   |
| Validation | 180    |

Training / Validation Ratio:

```text
90% / 10%
```

### GC10-DET

| Set        | Images |
| ---------- | ------ |
| Training   | 2850   |
| Validation | 713    |

Training / Validation Ratio:

```text
80% / 20%
```

---

## Data Preparation

All images were resized to:

```text
256 × 256
```

before training.

This preprocessing strategy follows the lightweight design objective of SDM-YOLO and enables fair comparison with all baseline detectors under identical experimental conditions.

---

## Reproducibility

All experiments were conducted using:

* Python 3.12.4
* PyTorch 2.7.0
* CUDA 11.8
* NVIDIA RTX 3060 (12 GB)

The complete training configuration, model architecture files, and pretrained weights are provided in this repository.
