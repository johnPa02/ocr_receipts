import csv
import os

import numpy as np
from PIL import Image, ImageDraw
from rotation_corrector.utils.utility import get_boxes_data


# Load CSV and process each entry
def process_annotations(csv_file_path, image_dir, output_dir, output_txt):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile, open(output_txt, 'w',
                                                                            encoding='utf-8') as outfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_id = row['img_id']
            img_path = os.path.join(image_dir, img_id)
            img = Image.open(img_path)
            # Load bounding boxes and labels
            annotations  = eval(row['anno_polygons'])  # List of dictionaries
            labels = row['anno_texts'].split('|||')

            for idx, annotation in enumerate(annotations):
                label = labels[idx]
                segmentation = annotation['segmentation']

                # Flatten the segmentation list (if it's multi-part, merge into a single polygon)
                if isinstance(segmentation[0], list):
                    segmentation = [coord for segment in segmentation for coord in segment]

                # Create cropped file name
                crop_file_name = f"{os.path.splitext(img_id)[0]}_crop_{idx}.jpg"

                # Crop and save the image using segmentation
                boxes_data = get_boxes_data(np.array(img), [segmentation], img_type='pil')
                cropped_img = boxes_data[0]
                cropped_img.save(os.path.join(output_dir, crop_file_name))
                # Write to output file
                outfile.write(f"{crop_file_name}\t{label}\n")


# Paths
csv_file_path = r'D:\ocr_receipts\data\mcocr_train_df_filtered_rotated.csv'
image_dir = r'D:\ocr_receipts\data\rotated_images'
output_dir = 'cropped_images/'
output_txt = 'cropped_images_labels.txt'

# Process the annotations and crop images
process_annotations(csv_file_path, image_dir, output_dir, output_txt)
