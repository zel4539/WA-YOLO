import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\pycharm\project\ultralytics-main\ultralytics\cfg\models\11\WA-YOLO.yaml')
    #model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'D:\pycharm\project\datasets\try123.v4i.yolov11\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                close_mosaic=0,
                device='0',
                # amp=False,
                optimizer='SGD', # using SGD
                project='runs/train',
                name='exp',
                )