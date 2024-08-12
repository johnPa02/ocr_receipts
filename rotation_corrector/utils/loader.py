import numbers
import os
from torchvision.transforms.functional import pad


def get_img_paths(dir_, extensions=('.jpg', '.png', '.jpeg', '.PNG', '.JPG', '.JPEG')):
    img_paths = []
    if type(dir_) is list:
        for d in dir_:
            for root, dirs, files in os.walk(d):
                for file in files:
                    for e in extensions:
                        if file.endswith(e):
                            p = os.path.join(root, file)
                            img_paths.append(p)
    else:
        for root, dirs, files in os.walk(dir_):
            for file in files:
                for e in extensions:
                    if file.endswith(e):
                        p = os.path.join(root, file)
                        img_paths.append(p)
    return img_paths


class NewPad(object):
    def __init__(self, t_size=(64, 192), fill=(255, 255, 255), padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.t_size = t_size

    def __call__(self, img):
        # def __call__(self, img, t_size):
        target_h, target_w = self.t_size
        # h, w, c = img.shape
        w, h = img.size

        im_scale = h / w
        target_scale = target_h / target_w
        # print(im_scale, target_scale)
        if im_scale < target_scale:
            # keep w, add padding h
            new_w = int(round(target_h / im_scale))
            # new_w =
            out_im = img.resize((new_w, target_h))
            # out_im = img
        else:
            # keep h, add padding w
            new_w = h / target_scale
            _pad = (new_w - w) / 2
            _pad = int(round(_pad))
            padding = (_pad, 0, _pad, 0)  # left, top, right and bottom
            # padding = (0, _pad, 0, _pad)  # left, top, right and bottom
            out_im = pad(img, padding, self.fill, self.padding_mode)
            out_im = out_im.resize((self.t_size[1], self.t_size[0]))
        # print(img.size, out_im.size)
        return out_im

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.fill, self.padding_mode)