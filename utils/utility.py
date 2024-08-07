import os
import random
type_map = {1: 'OTHER', 15: 'SELLER', 16: 'ADDRESS', 17: 'TIMESTAMP', 18: 'TOTAL_COST'}


def get_list_file_in_folder(folder_path, endswith=['.jpg', '.png', '.JPG', '.PNG']):
    list_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(tuple(endswith)):
                list_files.append(os.path.join(root, file))
    return list_files


def get_random_img_path(folder_path, n):
    list_files = get_list_file_in_folder(folder_path)
    random_image_paths = random.sample(list_files, min(n, len(list_files)))
    return random_image_paths
