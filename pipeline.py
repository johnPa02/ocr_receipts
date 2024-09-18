import os.path
import shutil
import cv2
import config
from text_detector.predict import CustomTextDetector
from rotation_corrector.inference import ImageRotationCorrector
from key_info_extraction.predict import KeyInfoExtractor
from text_recognitor.predict import TextRecognizer


class OCRPipeline:
    def __init__(self):
        self.text_detector = CustomTextDetector(
            det_model_dir=config.det_model_dir,
            det_db_thresh=config.det_db_thresh,
            det_db_box_thresh=config.det_db_box_thresh,
            use_gpu=False
        )
        self.rotation_corrector = ImageRotationCorrector()
        self.text_recognizer = TextRecognizer()
        self.key_info_extractor = KeyInfoExtractor()

    def process_image(self, img_path):
        if os.path.exists(config.kie_boxes_transcripts_temp):
            shutil.rmtree(config.kie_boxes_transcripts_temp)
        # Detect text
        img = cv2.imread(img_path)
        boxes, _ = self.text_detector(img)
        # Correct rotation
        img_rotated, boxes_list = self.rotation_corrector.process_image(img, boxes, self.text_detector)
        # Draw boxes for debugging
        out_path = os.path.join(r'D:\ocr_receipts\text_detector\test_results', os.path.basename(img_path))
        self.text_detector.draw_ocr(img_rotated, boxes_list, out_path)
        # Recognize text
        txts, scores = self.text_recognizer.recognize_text(img_rotated, boxes_list)
        # Extract key information
        images_folder = os.path.dirname(img_path)
        self.key_info_extractor.save_boxes_and_transcripts(img_path, boxes_list, txts, scores)
        entities = self.key_info_extractor.extract(images_folder, config.kie_boxes_transcripts_temp)
        return entities


if __name__ == '__main__':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    pipeline = OCRPipeline()
    entities = pipeline.process_image(r'D:\ocr_receipts\data\val_images\mcocr_val_145115gozuc.jpg')
    print(entities)
    # mcocr_val_145115gozuc
