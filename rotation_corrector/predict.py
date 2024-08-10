from config import rot_drop_thresh
from rotation_corrector.utils.utility import rotate_image_bbox_angle, get_boxes_data, drop_box, filter_90_box
from rotation_corrector.utils.utility import get_mean_horizontal_angle


class ImageRotation:
    def __init__(self, model_path: str):
        self.rotation_model = None

    def calculate_page_orient(self, box_rectify, img_rotated, boxes_list):
        boxes_data = get_boxes_data(img_rotated, boxes_list)
        rotation_state = {'0': 0, '180': 0}
        for it, img in enumerate(boxes_data):
            _, degr = box_rectify.inference(img, debug=False)
            rotation_state[degr[0]] += 1
        print(rotation_state)
        if rotation_state['0'] >= rotation_state['180']:
            ret = 0
        else:
            ret = 180
        return ret

    def process_image(self, image, boxes):
        boxes_list = drop_box(boxes, drop_gap=rot_drop_thresh)
        rotation = get_mean_horizontal_angle(boxes_list, False)
        img_rotated, boxes_list = rotate_image_bbox_angle(image, boxes_list, rotation)

        deg = self.calculate_page_orient(self.rotation_model, img_rotated, boxes_list)
        img_rotated, boxes_list = rotate_image_bbox_angle(img_rotated, boxes_list, deg)
        boxes_list = filter_90_box(boxes_list)
        return img_rotated, boxes_list
