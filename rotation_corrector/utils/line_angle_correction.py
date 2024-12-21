import cv2
import numpy as np


def rotate_and_crop(img, points, debug=False, rotate=True, extend=True,
                    extend_x_ratio=1, extend_y_ratio=0.01,
                    min_extend_y=1, min_extend_x=2):
    rect = cv2.minAreaRect(points)

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if debug:
        print("shape of cnt: {}".format(points.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    height = int(rect[1][0])
    width = int(rect[1][1])

    if extend:
        if width > height:
            w, h = width, height
        else:
            h, w = width, height
        ex = min_extend_x if (extend_x_ratio * w) < min_extend_x else (extend_x_ratio * w)
        ey = min_extend_y if (extend_y_ratio * h) < min_extend_y else (extend_y_ratio * h)
        ex = int(round(ex))
        ey = int(round(ey))
        if width < height:
            ex, ey = ey, ex
    else:
        ex, ey = 0, 0
    src_pts = box.astype("float32")
    # width = width + 10
    # height = height + 10
    dst_pts = np.array([
        [width - 1 + ex, height - 1 + ey],
        [ex, height - 1 + ey],
        [ex, ey],
        [width - 1 + ex, ey]
    ], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    # print(M)
    warped = cv2.warpPerspective(img, M, (width + 2 * ex, height + 2 * ey))
    h, w, c = warped.shape
    rotate_warped = warped

    # custom modification
    if (rect[2]) > 45:
        rotate_warped = cv2.rotate(rotate_warped, cv2.ROTATE_180)
    # =================

    if w < h and rotate:
        rotate_warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    if debug:
        print('ex, ey', ex, ey)
        cv2.imshow('before rotated', warped)
        cv2.imshow('rotated', rotate_warped)
        cv2.waitKey(0)
    return rotate_warped