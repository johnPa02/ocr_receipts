import os
import cv2
from PIL import Image
from paddleocr.tools.infer.utility import parse_args, draw_ocr, draw_text_det_res
from paddleocr.tools.infer.predict_det import TextDetector
from paddleocr.ppocr.utils.utility import check_and_read
import config


# class TextDetectorAPI:
#     def __init__(self, **kwargs):
#         self.args = parse_args()
#         self.args.det_model_dir = kwargs.get('det_model_dir', config.det_model_dir)
#         for key, value in kwargs.items():
#             setattr(self.args, key, value)
#         self.text_detector = TextDetector(self.args)
#
#     def predict(self, image):
#         det_res = []
#         img, flag_gif, flag_pdf = check_and_read(image_file)
#         if not flag_gif and not flag_pdf:
#             img = cv2.imread(image_file)
#         if not flag_pdf:
#             imgs = [img]
#         else:
#             page_num = self.args.page_num
#             if page_num > len(img) or page_num == 0:
#                 page_num = len(img)
#             imgs = img[:page_num]
#         for index, img in enumerate(imgs):
#             dt_boxes, _ = self.text_detector(img)
#             det_res.append(dt_boxes)
#         return det_res

class CustomTextDetector:
    def __init__(self, **kwargs):
        args = parse_args()
        for key, value in kwargs.items():
            setattr(args, key, value)
        self.text_detector = TextDetector(args)

    def __call__(self, image):
        return self.text_detector(image)


if __name__ == '__main__':
    from utils.utility import get_list_file_in_folder
    text_detector = CustomTextDetector(
        det_model_dir=config.det_model_dir,
        use_gpu=False
    )

    image_file_list = get_list_file_in_folder(config.filtered_train_img_dir)
    for image_file in image_file_list:
        # case img is jpg, png
        img = cv2.imread(image_file)
        dt_boxes, _ = text_detector(img)
        # draw boxes to image
        # src_im = draw_text_det_res(dt_boxes, img)
        # img_path = os.path.join(
        #     'paddle_results', "det_res_{}".format(os.path.basename(image_file))
        # )
        # cv2.imwrite(img_path, src_im)

        # save boxes to txt
        txt_path = os.path.join(
            'txt_det', str(os.path.basename(image_file).replace('.jpg', '.txt'))
        )
        result_txt = ''
        for box in dt_boxes:
            line = ','.join(map(str, [item for sublist in box for item in sublist]))
            result_txt += line + ',\n'
        result_txt = result_txt.rstrip(',\n')
        with open(txt_path, 'w') as f:
            f.write(result_txt)


