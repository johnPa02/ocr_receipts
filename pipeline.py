import os.path
import cv2
import config
from text_detector.predict import CustomTextDetector
from key_info_extraction.predict import KeyInfoExtractor
from rotation_corrector.inference import ImageRotationCorrector
from text_recognitor.predict import TextRecognizer


class OCRPipeline:
    def __init__(self):
        self.text_detector = CustomTextDetector(
            det_model_dir=config.det_model_dir,
            use_gpu=False
        )
        self.rotation_corrector = ImageRotationCorrector()
        self.text_recognizer = TextRecognizer()
        self.key_info_extractor = KeyInfoExtractor()

    def process_image(self, img_path):
        if os.path.exists(config.kie_boxes_transcripts_temp):
            os.remove(config.kie_boxes_transcripts_temp)
        # Detect text
        boxes = self.text_detector(img_path)
        boxes = boxes[0]
        # Correct rotation
        img = cv2.imread(img_path)
        img_rotated, boxes_list = self.rotation_corrector.process_image(img, boxes)
        # Recognize text
        txts, scores = self.text_recognizer.recognize_text(img_rotated, boxes_list)
        # Extract key information
        images_folder = os.path.dirname(img_path)
        self.key_info_extractor.save_boxes_and_transcripts(img_path, boxes_list, txts, scores)
        self.key_info_extractor.extract(images_folder, config.kie_boxes_transcripts_temp)


if __name__ == '__main__':
    pipeline = OCRPipeline()
    pipeline.process_image(r'D:\ocr_receipts\data\rotated_images\mcocr_public_145013aagqw.jpg')