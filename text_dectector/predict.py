import cv2
from paddleocr.tools.infer.utility import parse_args
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.ppocr.utils.utility import check_and_read
import config


class TextDetectorAPI:
    def __init__(self, **kwargs):
        self.args = parse_args()
        self.args.det_model_dir = kwargs.get('det_model_dir', config.det_model_dir)
        for key, value in kwargs.items():
            setattr(self.args, key, value)
        self.text_detector = TextDetector(self.args)

    def predict(self, image_file):
        det_res = []
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            imgs = [img]
        else:
            page_num = self.args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            dt_boxes, _ = self.text_detector(img)
            det_res.append(dt_boxes)
        return det_res


if __name__ == '__main__':
    text_detector_api = TextDetectorAPI(use_gpu=False)
    result = text_detector_api.predict("D:/ocr_receipts/data/train_images/mcocr_public_145013ahfqj.jpg")
