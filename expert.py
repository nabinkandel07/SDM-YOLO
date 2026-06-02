import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('best.pt')
    model.export(format='onnx', simplify=True, opset=13)

    onnx_model = YOLO("best.onnx")




    results = onnx_model("E:\NEU-DETSSDD\valid\images\crazing_20.jpg")
