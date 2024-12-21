import os

import cv2
from PIL import Image
from utils.utility import get_random_img_path, get_list_file_in_folder
import paddleocr.tools.infer.utility as utility
from paddleocr.tools.infer.predict_cls import TextClassifier
import config


def make_rotated_images(img_paths, output_dir, angle=180):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in img_paths:
        img = Image.open(img_path)
        img = img.rotate(angle, expand=True)
        img.save(os.path.join(output_dir, os.path.basename(img_path)))


if __name__ == '__main__':
    # output_dir = "/data/sample_train_images_180"
    # img_paths = get_random_img_path("D:/ocr_receipts/data/train_images", 10)
    # make_rotated_images(img_paths, output_dir)

    output_dir = "D:/ocr_receipts/data/train_images_sample_180"
    args = utility.parse_args()
    args.image_dir = output_dir
    args.cls_model_dir = config.rot_model_dir
    args.use_gpu = False

    image_file_list = get_list_file_in_folder(args.image_dir)
    text_classifier = TextClassifier(args)
    img_list = []
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        img_list.append(img)

    try:
        img_list, cls_res, predict_time = text_classifier(img_list)
    except Exception as E:
        print(E)
        exit()
    for ino in range(len(img_list)):
        print(f"Predicts of {image_file_list[ino]}: {cls_res[ino]}")
