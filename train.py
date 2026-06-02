import os
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 然后再 import numpy, torch, onnxruntime ...
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolo11-sdm.yaml')
    
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=2,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD,Adam
                # patience=0, # close earlystop
                resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='SDM',
               #freeze=10
                )
