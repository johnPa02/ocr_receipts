import cv2
from config import rot_drop_thresh, rot_model_path
from rotation_corrector.predict import init_box_rectify_model
from rotation_corrector.utils.utility import rotate_image_bbox_angle, get_boxes_data, drop_box, filter_90_box
from rotation_corrector.utils.utility import get_mean_horizontal_angle
from text_dectector.predict import TextDetectorAPI
from utils.utility import get_list_file_in_folder


class ImageRotationCorrector:
    def __init__(self, weight_path=rot_model_path):
        self.rotation_model = init_box_rectify_model(weight_path)

    def calculate_page_orient(self, img_rotated, boxes_list):
        boxes_data = get_boxes_data(img_rotated, boxes_list)
        rotation_state = {'0': 0, '180': 0}
        for it, img in enumerate(boxes_data):
            _, degr = self.rotation_model.inference(img, debug=False)
            rotation_state[degr[0]] += 1
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
    ir_corrector = ImageRotationCorrector()
    text_detector_api = TextDetectorAPI(use_gpu=False)

    test_image_paths = [
        "D:/ocr_receipts/data/train_images/mcocr_public_145013ahfqj.jpg",
        "D:/ocr_receipts/data/train_images/mcocr_public_145013aisfu.jpg"
    ]

    for idx, img_path in enumerate(test_image_paths):
        if idx > 9:
            break
        img = cv2.imread(img_path)
        boxes = text_detector_api.predict(img_path)
        print(img_path, end=' ')
        img_corrected, boxes_corrected = ir_corrector.process_image(img, boxes[0])
