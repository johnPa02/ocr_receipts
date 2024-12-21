import os
import cv2
import numpy as np
from PIL import Image
from config import filtered_train_img_dir
from rotation_corrector.inference import ImageRotationCorrector
from text_detector.predict import TextDetectorAPI
from utils.utility import get_list_file_in_folder
from paddleocr.tools.infer.utility import draw_ocr


def test_cls():
    imrc = ImageRotationCorrector()
    text_dectector = TextDetectorAPI(use_gpu=False)
    img_paths = get_list_file_in_folder(filtered_train_img_dir)

    for idx, img_path in enumerate(img_paths):
        if idx == 10:
            break
        det_res = text_dectector.predict(img_path)
        boxes = det_res[0]

        print(f"Processing image {img_path}", end=" ")
        image = cv2.imread(img_path)
        imrc.process_image(image, boxes)


def draw_boxes_from_txt(txt_path, img_path):
    with open(txt_path, 'r') as f:
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

        image = Image.open(img_path).convert('RGB')
        im_show = draw_ocr(image, boxes=boxes)
        im_show = Image.fromarray(im_show)
        im_show.save(f'./draw_boxes/{os.path.basename(img_path)}')


if __name__ == '__main__':
    draw_boxes_from_txt(r"D:\ocr_receipts\data\txt_det_rotated\mcocr_public_145013idjei.txt",
                        r"D:\ocr_receipts\data\rotated_images\mcocr_public_145013idjei.jpg")