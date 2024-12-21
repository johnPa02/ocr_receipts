import os
from config import rec_thresh, rec_out_txt_dir, rot_txt_dir, rot_img_dir
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from rotation_corrector.utils.utility import get_boxes_data


class TextRecognizer:
    def __init__(self):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu'  # Hoặc 'cpu' nếu không có GPU
        self.detector = Predictor(config)

    def recognize_text(self, img, boxes):
        boxes_data = get_boxes_data(img, boxes, img_type='pil')
        txts, scores = self.detector.predict_batch(boxes_data, return_prob=True)
        return txts, scores


def write_result_to_txt(txt_path, boxes, txts, scores, prob_thresh):
    results = ''
    for idx, box in enumerate(boxes):
        str_box = ','.join([','.join(b) for b in box])
        if scores[idx] > prob_thresh:
            results += f"{str_box}, {txts[idx]}"
        else:
            results += f"{str_box},"
        results += '\n'
    results = results.rstrip('\n')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(results)


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
    from utils.utility import get_list_file_in_folder
    detector = TextRecognizer()
    list_img_path = get_list_file_in_folder(rot_img_dir)
    for idx, img_path in enumerate(list_img_path):
        img = cv2.imread(img_path)
        txt_path = os.path.join(rot_txt_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        txt_out_path = os.path.join(rec_out_txt_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        boxes = get_list_boxes_from_icdar(txt_path)
        txts, scores = detector.recognize_text(img, boxes)

        # draw result
        # im_show = draw_ocr(img, boxes,txts, scores, font_path='./Arial.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save(f'./ocr_visualize/{os.path.basename(img_path)}')

        # save result to txt
        write_result_to_txt(txt_out_path, boxes, txts, scores, prob_thresh=rec_thresh)
        print(f"Processed {idx + 1}/{len(list_img_path)} {img_path}")


if __name__ == '__main__':
    main()
