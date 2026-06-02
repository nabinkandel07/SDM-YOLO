import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 然后再 import numpy, torch, onnxruntime ...
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"E:\NEU-DETSSDD\runs\nabi\Nabin\weights\best.pt") 
    model.val(data='data.yaml',
              split='val', 
              imgsz=256,
              batch=2,
              # iou=0.7,
              #rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/results'
              )
