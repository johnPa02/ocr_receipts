from config import filtered_train_img_dir
from utils.utility import get_random_img_path

from paddleocr import PaddleOCR,draw_ocr
from PIL import Image
import os



ocr = PaddleOCR() # need to run only once to download and load model into memory

for idx, img_path in enumerate(get_random_img_path(filtered_train_img_dir, 10)):
    result = ocr.ocr(img_path, cls=False, rec=False)

    # draw result
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, boxes=result)
    im_show = Image.fromarray(im_show)

    im_show.save(f'./text_detector_results/{os.path.basename(img_path)}')
