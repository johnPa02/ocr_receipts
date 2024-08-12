import os

CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))

# organizer
train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images')
raw_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df.csv')

# EDA
filtered_train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images_filtered')
filtered_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df_filtered.csv')

# text_detector
det_model_dir = os.path.join(CONFIG_ROOT, 'models', 'ch_PP-OCRv3_det_infer')

# rotation_corrector
rot_drop_thresh = [.5, 2]
rot_model_dir = os.path.join(CONFIG_ROOT, 'models', 'ch_ppocr_mobile_v2.0_cls_infer')
rot_model_path = os.path.join(CONFIG_ROOT,'rotation_corrector/weights/mobilenetv3-Epoch-487-Loss-0.03-Acc-0.99.pth')