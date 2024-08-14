import cv2
from PIL import Image
import chardet
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from utils.utility import rotate_and_crop
import numpy as np

config = Cfg.load_config_from_name('vgg_seq2seq')

config['cnn']['pretrained'] = True
config['device'] = 'cpu'  # Hoặc 'cpu' nếu không có GPU

detector = Predictor(config)
test_image_path = r"D:\ocr_receipts\data\rotated_images\mcocr_public_145013idjei.jpg"
test_txt_path = r"D:\ocr_receipts\data\txt_det_rotated\mcocr_public_145013idjei.txt"

with open(test_txt_path, 'r') as f:
    boxes = []
    lines = f.readlines()
    for line in lines:
        points = line.rstrip('\n').split(',')
        box = np.array([
            [float(points[0]), float(points[1])],
            [float(points[2]), float(points[3])],
            [float(points[4]), float(points[5])],
            [float(points[6]), float(points[7])]
        ], dtype=np.float32)
        boxes.append(box)

img = cv2.imread(test_image_path)
for box in boxes:
    img_crop = rotate_and_crop(img, box, debug=False, extend=True,
                               extend_x_ratio=0.0001,
                               extend_y_ratio=0.0001,
                               min_extend_y=2, min_extend_x=1)
    # convert numpy array to image
    img_crop = Image.fromarray(img_crop)
    text = detector.predict(img_crop)