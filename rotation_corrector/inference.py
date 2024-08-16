import os.path

import cv2
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.tools.infer.utility import parse_args

import config
from config import rot_drop_thresh, rot_model_path, rot_img_dir, rot_txt_dir, filtered_train_img_dir
from rotation_corrector.predict import init_box_rectify_model
from rotation_corrector.utils.utility import rotate_image_bbox_angle, get_boxes_data, drop_box, filter_90_box
from rotation_corrector.utils.utility import get_mean_horizontal_angle
from utils.utility import get_list_file_in_folder, load_det_model


def write_boxes_to_txt(boxes, txt_path):
    with open(txt_path, 'w') as f:
        for box in boxes:
            box_str = ','.join([str(int(x)) for x in box])
            f.write(box_str + '\n')


class ImageRotationCorrector:
    def __init__(self, weight_path=rot_model_path):
        self.rotation_model = init_box_rectify_model(weight_path)

    def calculate_page_orient(self, img_rotated, boxes_list):
        boxes_data = get_boxes_data(img_rotated, boxes_list)
        rotation_state = {'0': 0, '180': 0}
        for it, img in enumerate(boxes_data):
            _, degr = self.rotation_model.inference(img, debug=False)
            rotation_state[degr[0]] += 1
            if degr[0] == '180':
                print('180')
        if rotation_state['0'] >= rotation_state['180']:
            ret = 0
        else:
            ret = 180
        return ret

    def process_image(self, image, boxes):
        boxes_list = drop_box(boxes, drop_gap=rot_drop_thresh)
        rotation = get_mean_horizontal_angle(boxes_list, False)
        img_rotated, boxes_list = rotate_image_bbox_angle(image, boxes_list, rotation)
        deg = self.calculate_page_orient(img_rotated, boxes_list)
        print(deg)
        img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, deg)
        boxes_list = filter_90_box(boxes_list)
        return img_rotated, boxes_list


if __name__ == '__main__':
    if os.path.exists(rot_img_dir) or os.path.exists(rot_txt_dir):
        raise Exception("Output folder already exists, it will be overwritten")
    os.makedirs(rot_img_dir, exist_ok=True)
    os.makedirs(rot_txt_dir, exist_ok=True)

    imrc = ImageRotationCorrector()

    text_dectector = load_det_model(
        det_model_dir=config.det_model_dir,
        use_gpu=False
    )
    img_paths = get_list_file_in_folder(filtered_train_img_dir)
    for idx, img_path in enumerate(img_paths):
        det_res = text_dectector.predict(img_path)
        boxes = det_res[0]
        image = cv2.imread(img_path)
        img_rotated, boxes_list = imrc.process_image(image, boxes)

        out_img_path = os.path.join(rot_img_dir, os.path.basename(img_path))
        out_txt_path = os.path.join(rot_txt_dir, os.path.basename(img_path).split('.')[0] + '.txt')

        write_boxes_to_txt(boxes_list, out_txt_path)
        cv2.imwrite(out_img_path, img_rotated)

        print(f"[{idx+1}/{len(img_paths)}] Processed {img_path}")

