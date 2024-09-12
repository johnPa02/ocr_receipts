import csv
import os
import random

import cv2
import numpy as np

from config import filtered_csv, rec_out_txt_dir, kie_out_txt_dir, rot_img_dir, filterd_rotated_csv, \
    kie_boxes_transcripts, kie_boxes_transcripts_test
from utils.utility import cer_loss_one_image, get_list_file_in_folder
from utils.utility import color_map, type_map, get_list_gt_poly, get_list_icdar_poly, IoU


def parse_anno_from_csv_to_icdar_result(csv_file, icdar_dir, output_dir, img_dir=None, debug=False):
    with open(csv_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line = True

        total_boxes_not_match = 0
        total_boxes = 0
        for n, row in enumerate(csv_reader):
            if first_line:
                first_line = False
                continue
            if n < 0:
                continue
            img_name = row[0]
            print('\n' + str(n), img_name)
            # if 'mcocr_public_145014smasw' not in img_name:
            #     continue
            src_img = cv2.imread(os.path.join(img_dir, img_name))
            # src_img = cv2.imread(os.path.join(img_dir, 'viz_' + img_name))

            # Read all poly from training data
            list_gt_poly = get_list_gt_poly(row)

            # Read all poly from icdar
            icdar_path = os.path.join(icdar_dir, img_name.replace('.jpg', '.txt'))
            list_icdar_poly = get_list_icdar_poly(icdar_path)

            # Compare iou and parse text from training data
            for pol in list_gt_poly:
                total_boxes += 1
                match = False
                if debug:
                    gt_img = src_img.copy()
                    gt_box = np.array(pol.list_pts).astype(np.int32)
                    cv2.polylines(gt_img, [gt_box], True, color=color_map[pol.type], thickness=2)
                max_iou = 0
                for icdar_pol in list_icdar_poly:
                    iou = IoU(pol, icdar_pol, False)
                    if iou > max_iou:
                        max_iou = iou
                    cer = cer_loss_one_image(pol.value, icdar_pol.value)
                    if debug:
                        pred_img = src_img.copy()
                        pred_box = np.array(icdar_pol.list_pts).astype(np.int32)
                        cv2.polylines(pred_img, [pred_box], True, color=color_map[pol.type], thickness=2)
                    if iou > 0.3:
                        match = True
                        print('gt  :', pol.value)
                        print('pred:', icdar_pol.value)
                        print('cer', round(cer, 3), ',iou', iou)
                        icdar_pol.type = pol.type

                if not match:
                    total_boxes_not_match += 1
                    print(' not match gt  :', pol.value)
                    print('Max_iou', max_iou)
                    if debug:
                        gt_img_res = cv2.resize(gt_img, (int(gt_img.shape[1]/2),int(gt_img.shape[0]/2)))
                        cv2.imshow('not match gt box', gt_img_res)
                        cv2.waitKey(0)

            # save to output file
            output_icdar_path = os.path.join(output_dir, img_name.replace('.jpg', '.txt'))
            output_icdar_txt = ''
            for icdar_pol in list_icdar_poly:
                output_icdar_txt += icdar_pol.to_icdar_line(map_type=type_map) + '\n'

            output_icdar_txt = output_icdar_txt.rstrip('\n')
            with open(output_icdar_path, 'w', encoding='utf-8') as f:
                f.write(output_icdar_txt)
            if total_boxes > 0:
                print('Total not match', total_boxes_not_match, 'total boxes', total_boxes, 'not match ratio',
                      round(total_boxes_not_match / total_boxes, 3))


def create_data_pick_boxes_and_transcripts(icdar_dir, output_dir):
    list_file = get_list_file_in_folder(icdar_dir, ext=['txt'])
    for idx, anno in enumerate(list_file):
        print(idx, anno)
        with open(os.path.join(icdar_dir, anno), mode='r', encoding='utf-8') as f:
            list_bboxes = f.readlines()
        for idx, line in enumerate(list_bboxes):
            list_bboxes[idx] = str(idx + 1) + ',' + line
        with open(os.path.join(output_dir, anno.replace('.txt', '.tsv')), mode='wt', encoding='utf-8') as f:
            f.writelines(list_bboxes)


def create_data_pick_boxes_and_transcripts(icdar_dir, output_dir):
    list_file = get_list_file_in_folder(icdar_dir, endswith=['txt'])
    for idx, anno in enumerate(list_file):
        print(idx, anno)
        with open(os.path.join(icdar_dir, anno), mode='r', encoding='utf-8') as f:
            list_bboxes = f.readlines()
        for idx, line in enumerate(list_bboxes):
            list_bboxes[idx] = str(idx + 1) + ',' + line
        out_file = os.path.join(output_dir, os.path.basename(anno).replace('txt','tsv'))
        with open(os.path.join(out_file), mode='wt', encoding='utf-8') as f:
            f.writelines(list_bboxes)


def create_data_pick_csv_train_val(train_dir, train_ratio=0.92):
    list_files = get_list_file_in_folder(os.path.join(train_dir, 'images'))
    num_total = len(list_files)
    num_train = int(num_total * train_ratio)
    num_val = num_total - num_train

    random.shuffle(list_files)
    list_train = list_files[:num_train]
    list_val = list_files[num_train + 1:]

    train_txt_list = []
    for idx, f in enumerate(list_train):
        name = os.path.basename(f).split('.')[0]
        line = ','.join([str(idx + 1), 'receipts', name])
        train_txt_list.append(line + '\n')

    with open(os.path.join(train_dir, 'train_list.csv'), mode='w', encoding='utf-8') as f:
        f.writelines(train_txt_list)

    val_txt_list = []
    for idx, f in enumerate(list_val):
        name = os.path.basename(f).split('.')[0]
        line = ','.join([str(idx + 1), 'receipts', name])
        val_txt_list.append(line + '\n')

    with open(os.path.join(train_dir, 'val_list.csv'), mode='w', encoding='utf-8') as f:
        f.writelines(val_txt_list)
    print('Done')


if __name__ == '__main__':
    # parse_anno_from_csv_to_icdar_result(csv_file=filterd_rotated_csv,
    #                                     icdar_dir=rec_out_txt_dir,
    #                                     output_dir=kie_out_txt_dir,
    #                                     img_dir=rot_img_dir,
    #                                     debug=False)
    # os.symlink(rot_img_dir, os.path.join(kie_train_dir, 'images'))

    # create_data_pick_boxes_and_transcripts(icdar_dir=kie_out_txt_dir,
    #                                        output_dir=kie_boxes_transcripts)

    # Create train data
    # kie_train_dir = os.path.dirname(kie_out_txt_dir)
    # create_data_pick_csv_train_val(kie_train_dir, train_ratio=0.92)

    # Create test data
    create_data_pick_boxes_and_transcripts(icdar_dir=rec_out_txt_dir,
                                           output_dir=kie_boxes_transcripts_test)

