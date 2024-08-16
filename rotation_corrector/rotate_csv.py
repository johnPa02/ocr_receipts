import ast, cv2, os
from utils.utility import rotate_image_bbox_angle,drop_box, get_mean_horizontal_angle
from inference import get_list_boxes_from_icdar, ImageRotationCorrector
from config import det_out_txt_dir

import pandas as pd


def rotate_polygon_in_csv(csv_file, output_csv_file, img_dir):
    anno_dir = det_out_txt_dir
    imrc = ImageRotationCorrector()

    # Đọc CSV bằng pandas
    df = pd.read_csv(csv_file)

    output_rows = []

    for n, row in df.iterrows():
        img_name = row[0]
        print(n, img_name)

        boxes = ast.literal_eval(row[1])

        test_img = cv2.imread(os.path.join(img_dir, img_name))
        tem_img = test_img.copy()

        anno_path = os.path.join(anno_dir, img_name.replace('.jpg', '.txt'))
        boxes_list = get_list_boxes_from_icdar(anno_path)
        boxes_list = drop_box(boxes_list)
        tem_boxes_list = [temp['segmentation'] for temp in boxes]

        rotation = get_mean_horizontal_angle(boxes_list, False)
        img_rotated, boxes_list = rotate_image_bbox_angle(test_img, boxes_list, rotation)
        tem_img_rotated, tem_boxes_list = rotate_image_bbox_angle(tem_img, tem_boxes_list, rotation)

        degre = imrc.calculate_page_orient(img_rotated, boxes_list)
        tem_img_rotated, tem_boxes_list = rotate_image_bbox_angle(tem_img_rotated, tem_boxes_list, degre)

        for idx, temp in enumerate(boxes):
            boxes[idx]['segmentation'] = tem_boxes_list[idx]

        # Cập nhật row
        row[1] = str(boxes)
        output_rows.append(row)

    # Ghi kết quả vào file CSV mới
    df_output = pd.DataFrame(output_rows)
    df_output.to_csv(output_csv_file, index=False)
    print('Done')


if __name__ == '__main__':
    from config import filtered_train_img_dir, filtered_csv, rotate_filtered_csv
    rotate_polygon_in_csv(csv_file=filtered_csv,
                          output_csv_file=rotate_filtered_csv,
                          img_dir=filtered_train_img_dir)