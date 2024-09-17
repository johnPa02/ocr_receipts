import math
import os

import cv2
import numpy as np
from PIL import Image

from rotation_corrector.utils.line_angle_correction import rotate_and_crop
from scipy.cluster.vq import kmeans, vq


def get_list_file_in_folder(folder_path, endswith=['.jpg', '.png', '.JPG', '.PNG']):
    list_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(tuple(endswith)):
                list_files.append(os.path.join(root, file))
    return list_files


def rotate_image_bbox_angle(img, bboxes, angle):
    def rotate_points(box):
        box_np = np.array(box).astype(float)
        box_np = np.rint(box_np).astype(np.int32)
        # print(box_np.shape)
        box_np = box_np.reshape(-1, 2)
        # add ones
        ones = np.ones(shape=(len(box_np), 1))
        points_ones = np.hstack([box_np, ones])
        # transform points
        transformed_points = Mat_rotation.dot(points_ones.T).T
        # print(transformed_points)
        transformed_points2 = transformed_points.reshape(-1)
        transformed_points2 = np.rint(transformed_points2)
        transformed_points2 = transformed_points2.astype(int)
        # print(transformed_points2)
        return transformed_points2

    if not isinstance(img, np.ndarray):
        img = np.array(img)
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # print(h_org, w_org)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    bound_w = (h_org * abs(sin)) + (w_org * abs(cos))
    bound_h = (h_org * abs(cos)) + (w_org * abs(sin))
    # print((bound_w / 2) - w_org / 2, ((bound_h / 2) - h_org / 2))
    Mat_rotation[0, 2] += ((bound_w / 2) - w_org / 2) - 1
    Mat_rotation[1, 2] += ((bound_h / 2) - h_org / 2) - 1
    Mat_rotation[1, 2] = 0 if Mat_rotation[1, 2] < 0 else Mat_rotation[1, 2]
    Mat_rotation[0, 2] = 0 if Mat_rotation[0, 2] < 0 else Mat_rotation[0, 2]
    # print(Mat_rotation)
    # Mat_rotation = Mat_rotation.round()
    bound_w, bound_h = int(bound_w), int(bound_h)
    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))

    ret_boxes = []
    for box_data in bboxes:
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        if isinstance(box, list) and isinstance(box[0], list):
            transformed_points = []
            for b in box:
                transformed_points.append(list(rotate_points(b)))
        else:
            transformed_points = list(rotate_points(box))

        if isinstance(box_data, dict):
            box_data['coors'] = transformed_points
        else:
            box_data = transformed_points
        ret_boxes.append(box_data)

    return img_result, ret_boxes


def rotate_image_angle(img, angle):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    shape_ = img.shape
    h_org = shape_[0]
    w_org = shape_[1]
    Mat_rotation = cv2.getRotationMatrix2D((w_org / 2, h_org / 2), 360 - angle, 1)
    # print(h_org, w_org)
    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    bound_w = (h_org * abs(sin)) + (w_org * abs(cos))
    bound_h = (h_org * abs(cos)) + (w_org * abs(sin))
    # print((bound_w / 2) - w_org / 2, ((bound_h / 2) - h_org / 2))
    Mat_rotation[0, 2] += ((bound_w / 2) - w_org / 2) - 1
    Mat_rotation[1, 2] += ((bound_h / 2) - h_org / 2) - 1
    Mat_rotation[1, 2] = 0 if Mat_rotation[1, 2] < 0 else Mat_rotation[1, 2]
    Mat_rotation[0, 2] = 0 if Mat_rotation[0, 2] < 0 else Mat_rotation[0, 2]
    # print(Mat_rotation)
    # Mat_rotation = Mat_rotation.round()
    bound_w, bound_h = int(bound_w), int(bound_h)
    img_result = cv2.warpAffine(img, Mat_rotation, (bound_w, bound_h))
    return img_result


def get_boxes_data(img_data, boxes, img_type='numpy'):
    boxes_data = []
    for box_data in boxes:
        if isinstance(box_data, dict):
            box_loc = box_data['coors']
        else:
            box_loc = box_data
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(img_data, box_loc, debug=False, extend=True,
                                   extend_x_ratio=0.0001,
                                   extend_y_ratio=0.0001,
                                   min_extend_y=2, min_extend_x=1)
        if img_type == 'pil':
            box_data = Image.fromarray(box_data)

        boxes_data.append(box_data)
    return boxes_data


def drop_box(boxlist, drop_gap=(.5, 2), debug=False):
    new_boxlist = []
    for ide, box_data in enumerate(boxlist):
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        box_np = np.array(box).astype(np.int32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(box_np)
        w, h = rect[1]
        if min(drop_gap) < w / h < max(drop_gap):
            continue
        new_boxlist.append(box_data)
    return new_boxlist


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))


class poly():
    def __init__(self, segment_pts, type=1, value=''):
        if isinstance(segment_pts, str):
            segment_pts = [int(f) for f in segment_pts.split(',')]
        elif isinstance(segment_pts, list):
            segment_pts = [round(f) for f in segment_pts]
            num_pts = int(len(segment_pts) / 2)
            # print('num_pts', num_pts)
            first_pts = [segment_pts[0], segment_pts[1]]
            self.list_pts = [first_pts]
            for i in range(1, num_pts):
                self.list_pts.append([segment_pts[2 * i], segment_pts[2 * i + 1]])
        else:
            # custom modify here
            self.list_pts = segment_pts
        self.type = type
        self.value = value

    def reduce_pts(self, dist_thres=7):  # reduce nearly duplicate points
        last_pts = self.list_pts[0]
        filter_pts = []
        for i in range(1, len(self.list_pts)):
            curr_pts = self.list_pts[i]
            dist = euclidean_distance(last_pts, curr_pts)
            # print('distance between', i - 1, i, ':', dist)
            if dist > dist_thres:
                filter_pts.append(last_pts)
                print('Keep point', i - 1)
            last_pts = curr_pts

        # print('distance between', len(self.list_pts) - 1, 0, ':', euclidean_distance(last_pts, self.list_pts[0]))
        if euclidean_distance(last_pts, self.list_pts[0]) > dist_thres:
            filter_pts.append(last_pts)
            # print('Keep last point')

        self.list_pts = filter_pts

    def check_max_wh_ratio(self):
        max_ratio = 0
        if len(self.list_pts) == 4:
            first_edge = euclidean_distance(self.list_pts[0], self.list_pts[1])
            second_edge = euclidean_distance(self.list_pts[1], self.list_pts[2])
            if first_edge / second_edge > 1:
                long_edge = (self.list_pts[0][0] - self.list_pts[1][0], self.list_pts[0][1] - self.list_pts[1][1])
            else:
                long_edge = (self.list_pts[1][0] - self.list_pts[2][0], self.list_pts[1][1] - self.list_pts[2][1])
            max_ratio = max(first_edge / second_edge, second_edge / first_edge)
        else:
            print('check_max_wh_ratio. Polygon is not qualitareal')
        return max_ratio, long_edge

    def check_horizontal_box(self):
        if len(self.list_pts) == 4:
            max_ratio, long_edge = self.check_max_wh_ratio()
            if long_edge[0] == 0:
                angle_with_horizontal_line = 90
            else:
                angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        else:
            print('check_horizontal_box. Polygon is not qualitareal')
        print('Angle', angle_with_horizontal_line)
        if 45 < math.fabs(angle_with_horizontal_line) < 135:
            return False
        else:
            return True

    def get_horizontal_angle(self):
        assert len(self.list_pts) == 4
        max_ratio, long_edge = self.check_max_wh_ratio()
        if long_edge[0] == 0:
            if long_edge[1] < 0:
                angle_with_horizontal_line = -90
            else:
                angle_with_horizontal_line = 90
        else:
            angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        return angle_with_horizontal_line

    def to_icdar_line(self, map_type=None):
        line_str = ''
        if len(self.list_pts) == 4:
            for pts in self.list_pts:
                line_str += '{},{},'.format(pts[0], pts[1])
            if map_type is not None:
                line_str += self.value + ',' + str(map_type[self.type])
            else:
                line_str += self.value + ',' + str(self.type)

        else:
            print('to_icdar_line. Polygon is not qualitareal')
        return line_str


def get_mean_horizontal_angle(boxlist, debug=False, cluster=True):
    if not boxlist:
        return 0
    all_box_angle = []
    for ide, box_data in enumerate(boxlist):
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        pol = poly(box)
        angle_with_horizontal_line = pol.get_horizontal_angle()
        # if 45 < abs(angle_with_horizontal_line) < 135:
        #     continue
        # print(angle_with_horizontal_line)
        if angle_with_horizontal_line >= 0:
            angle_with_horizontal_line = 180 - angle_with_horizontal_line + 90
        else:
            angle_with_horizontal_line = math.fabs(angle_with_horizontal_line) - 90
        all_box_angle.append(angle_with_horizontal_line)

    # all_box_angle
    mean_angle = np.array(all_box_angle).mean()
    mean_angle = mean_angle - 90
    return mean_angle


def filter_90_box(boxlist, debug=False, thresh=45):
    if not boxlist:
        return 0
    all_box_angle = []
    for ide, box_data in enumerate(boxlist):
        if isinstance(box_data, dict):
            box = box_data['coors']
        else:
            box = box_data
        pol = poly(box)
        angle_with_horizontal_line = pol.get_horizontal_angle()
        # if 45 < abs(angle_with_horizontal_line) < 135:
        #     continue
        # print(angle_with_horizontal_line)
        # if angle_with_horizontal_line >= 0:
        #     angle_with_horizontal_line = 180 - angle_with_horizontal_line + 90
        # else:
        #     angle_with_horizontal_line = math.fabs(angle_with_horizontal_line) - 90
        all_box_angle.append(angle_with_horizontal_line)

    # if cluster:
    all_box_angle = np.array(all_box_angle)
    all_box_angle_abs = np.absolute(all_box_angle)
    print(all_box_angle_abs.max() - all_box_angle_abs.min())
    if all_box_angle_abs.max() - all_box_angle_abs.min() > thresh:
        codebook, _ = kmeans(all_box_angle_abs, 2)  # three clusters
        cluster_indices, _ = vq(all_box_angle_abs, codebook)
        clas = set(cluster_indices)
        ret = {c: [] for c in clas}
        for idx, v in enumerate(all_box_angle_abs):
            ret[cluster_indices[idx]].append([v, boxlist[idx]])
        ret = list(ret.values())
        ret = sorted(ret, key=lambda e: len(e))
        list_angle_box = ret[-1]

        boxlist = []
        for ide, box_data in enumerate(list_angle_box):
            boxlist.append(box_data[1])
    return boxlist