# NWD Loss Implementation for SDM-YOLO

This document describes the implementation of the **Normalized Wasserstein Distance (NWD)** loss used in the SDM-YOLO framework, as proposed in:

> **SDM-YOLO: An Improved YOLO Model with Multiscale Attention for Steel Surface Defect Detection**
> Nabin Kandel, Ping Wu
> *Engineering Research Express*, 2026.

---

## 1. Overview

The NWD loss is integrated into the bounding-box regression stage to improve localization stability for **small, overlapping, and low-contrast defects**. Unlike the conventional IoU-based losses, NWD models bounding boxes as 2D Gaussian distributions and measures their similarity in a continuous metric space, which provides smoother gradients for small defects.

The final regression loss combines **CIoU** (Complete IoU) and **NWD** in an equal-weight formulation:

$$
\mathcal{L}_{\mathrm{reg}} = \frac{1}{2N}\sum_{i=1}^{N}\left[(1 - \mathrm{NWD}_i) + (1 - \mathrm{CIoU}_i)\right]
$$

---

## 2. Mapping Between Paper and Code

| Paper Formula | Description | Code Component |
|---------------|-------------|----------------|
| **Equation (7)** | Squared Wasserstein distance $W_2^2$ | `Wasserstein()` function |
| **Equation (8)** | Normalized Wasserstein Distance (NWD) | `torch.exp(-torch.pow(Wasserstein(...), 1/2) / λ)` |
| **Equation (9)** | Combined regression loss | `0.5 * CIoU_loss + 0.5 * NWD_loss` |

---

## 3. File to Modify

All changes should be made in:

```
ultralytics/utils/loss.py
```

---

## 4. Step 1 — Add the `Wasserstein` Function

Add the following function **at the top of `loss.py`**, before the `BboxLoss` class definition:

```python
def Wasserstein(box1, box2, xywh=True):
    """
    Compute the squared Wasserstein distance between two sets of bounding boxes.

    Args:
        box1 (Tensor): Predicted boxes, shape (4, N).
        box2 (Tensor): Target boxes, shape (N, 4).
        xywh (bool): If True, boxes are in (x1, y1, x2, y2) format.

    Returns:
        Tensor: Squared Wasserstein distance for each matched box pair.
    """
    box2 = box2.T
    if xywh:
        b1_cx, b1_cy = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        b1_w, b1_h = box1[2] - box1[0], box1[3] - box1[1]
        b2_cx, b2_cy = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        b2_w, b2_h = box2[2] - box2[0], box2[3] - box2[1]
    else:
        b1_cx, b1_cy, b1_w, b1_h = box1[0], box1[1], box1[2], box1[3]
        b2_cx, b2_cy, b2_w, b2_h = box2[0], box2[1], box2[2], box2[3]

    cx_L2Norm = torch.pow((b1_cx - b2_cx), 2)
    cy_L2Norm = torch.pow((b1_cy - b2_cy), 2)
    p1 = cx_L2Norm + cy_L2Norm

    w_FroNorm = torch.pow((b1_w - b2_w) / 2, 2)
    h_FroNorm = torch.pow((b1_h - b2_h) / 2, 2)
    p2 = w_FroNorm + h_FroNorm

    return p1 + p2
```

---

## 5. Step 2 — Modify the `BboxLoss` Class

Inside the `BboxLoss` class, locate the regression loss computation. It typically looks like this:

**Original code:**

```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
```

**Modified code (with NWD integrated):**

```python
loss_iou = 0

iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

# λ = 5.2 (as reported in the paper)
lambda_nwd = 5.2

nwd = torch.exp(
    -torch.pow(Wasserstein(pred_bboxes[fg_mask].T, target_bboxes[fg_mask], xywh=False), 1 / 2) / lambda_nwd
)

# Weighted combination matching Equation (9) in the paper
loss_iou1 = (((1.0 - iou) * weight).sum() / target_scores_sum) * 0.5 + \
            (((1.0 - nwd) * weight).sum() / target_scores_sum) * 0.5

loss_iou = loss_iou + loss_iou1
```

---

## 6. Important Note on λ (Normalization Factor)

### Paper vs. Early Prototype Code

| Source | Value of λ |
|--------|------------|
| **Published paper (Section 3.4)** | **λ = 5.2** |
| **Early prototype code (internal)** | λ = 1.0 (placeholder) |

The value **λ = 5.2** is the value used to produce the results reported in the paper. This value corresponds to the average absolute scale of defects in the NEU-DET dataset.

The same value (λ = 5.2) was retained for GC10-DET for consistency, because the defect scales in both datasets are of the same order of magnitude after resizing to 256×256.

**To reproduce the paper's results, use `lambda_nwd = 5.2`** as shown in the modified code above.

---

## 7. Coordinate Units

The bounding-box coordinates $(c_x, c_y, w, h)$ used in Equation (7) are in **image pixel units** at the network input resolution (256 × 256). The Wasserstein distance is therefore computed in pixel space, and λ is calibrated accordingly.

---

## 8. Weighting by Target Score

The NWD term is combined with the CIoU term using the same target-score weighting scheme as the standard YOLO regression loss. The loss is computed only for positive-matched anchors, and each matched box is weighted by the corresponding objectness/target score.

Equation (9) in the paper presents a simplified averaged form for clarity, but the actual implementation follows the standard YOLO regression loss pipeline, which is weighted by target score and normalized by the number of matched boxes $N$.

---

## 9. Formula-to-Code Summary

| Paper | Code |
|-------|------|
| $W_2^2(\mathbf{r}_p, \mathbf{r}_g)$ (Eq. 7) | `Wasserstein(pred.T, target, xywh=False)` |
| $\mathrm{NWD} = \exp(-\sqrt{W_2^2}/\lambda)$ (Eq. 8) | `torch.exp(-torch.pow(Wasserstein(...), 1/2) / lambda_nwd)` |
| $\mathcal{L}_{\mathrm{reg}} = \frac{1}{2N}\sum[(1-\mathrm{NWD}) + (1-\mathrm{CIoU})]$ (Eq. 9) | `0.5 * CIoU_loss + 0.5 * NWD_loss` |

---

## 10. Verification Tips

If you are reproducing the results, please verify the following:

1. The `Wasserstein()` function is called with `xywh=False` because the YOLO pipeline passes boxes in `(x1, y1, x2, y2)` format at that stage.
2. The divisor in the exponential uses **λ = 5.2**, not 1.0.
3. The combined loss uses **0.5 * CIoU + 0.5 * NWD**, matching Equation (9).
4. Coordinate units are pixel-based at 256 × 256 resolution.
5. The loss is applied only to positive-matched anchors, weighted by target score.

---

## 11. Citation

If you use this implementation in your research, please cite:

```bibtex
@article{kandel2026sdmyolo,
  title   = {SDM-YOLO: An Improved YOLO Model with Multiscale Attention for Steel Surface Defect Detection},
  author  = {Kandel, Nabin and Wu, Ping},
  journal = {Engineering Research Express},
  year    = {2026}
}
```

---

## 12. Contact

For questions or issues regarding this implementation, please contact:nabinkandel60@gmail.com

**Nabin Kandel**
School of Information Science and Engineering
Zhejiang Sci-Tech University, Hangzhou, China
Email: nabinkandel60@gmail.com

---

*Last updated: October 2026*
