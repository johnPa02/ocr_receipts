from config import filtered_train_img_dir
from utils.utility import get_random_img_path

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os


def detect_text(img_path):
    ocr = PaddleOCR()  # need to run only once to download and load model into memory
    if img_path is None:
        image_paths = get_random_img_path(filtered_train_img_dir, 10)
    else:
        image_paths = [img_path]
    for img_path in image_paths:
        result = ocr.ocr(img_path, cls=False, rec=False)
        # draw result
        result = result[0]
        image = Image.open(img_path).convert('RGB')
        im_show = draw_ocr(image, boxes=result)
        im_show = Image.fromarray(im_show)
        im_show.save(f'./text_detector_results/{os.path.basename(img_path)}')


if __name__ == '__main__':
    detect_text("D:/ocr_receipts/data/train_images/mcocr_public_145013ahfqj.jpg")

