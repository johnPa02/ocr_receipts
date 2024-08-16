import os
from utils.utility import get_list_file_in_folder
import cv2
from PIL import Image
from paddleocr.tools.infer.utility import draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from rotation_corrector.utils.utility import get_boxes_data

config = Cfg.load_config_from_name('vgg_seq2seq')

config['cnn']['pretrained'] = True
config['device'] = 'cpu'  # Hoặc 'cpu' nếu không có GPU

detector = Predictor(config)
image_dir = r"D:\ocr_receipts\data\rotated_images"
txt_dir = r"D:\ocr_receipts\data\txt_det_rotated"


def get_list_boxes_from_icdar(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    for anno in anno_txt:
        anno = anno.rstrip('\n').split(',')
        coors = [anno[i:i + 2] for i in range(0, len(anno), 2)]
        list_boxes.append(coors)
    return list_boxes


def main():
    list_img_path = get_list_file_in_folder(image_dir)
    for idx, img_path in enumerate(list_img_path):
        img = cv2.imread(img_path)
        txt_path = os.path.join(txt_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        boxes = get_list_boxes_from_icdar(txt_path)
        boxes_data = get_boxes_data(img, boxes, img_type='pil')
        txts, scores = detector.predict_batch(boxes_data, return_prob=True)
        im_show = draw_ocr(img, boxes,txts, scores, font_path='./Arial.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save(f'./ocr_visualize/{os.path.basename(img_path)}')
        print(f"Processed {idx + 1}/{len(list_img_path)} {img_path}")


if __name__ == '__main__':
    main()
