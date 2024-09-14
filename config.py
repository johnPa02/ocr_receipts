import os

CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))

# organizer
train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images')
raw_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df.csv')

# EDA
filtered_train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images_filtered')
filtered_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df_filtered.csv')
filterd_rotated_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df_filtered_rotated.csv')

# text_detector
det_model_dir = os.path.join(CONFIG_ROOT, 'models', 'ch_PP-OCRv3_det_infer')
det_out_txt_dir = os.path.join(CONFIG_ROOT, 'text_detector', 'txt_det')

# rotation_corrector
rot_drop_thresh = [.5, 2]
rot_model_dir = os.path.join(CONFIG_ROOT, 'models', 'ch_ppocr_mobile_v2.0_cls_infer')
rot_model_path = os.path.join(CONFIG_ROOT, 'rotation_corrector/weights/mobilenetv3-Epoch-487-Loss-0.03-Acc-0.99.pth')
rot_img_dir = os.path.join(CONFIG_ROOT, 'data', 'rotated_images')
rot_txt_dir = os.path.join(CONFIG_ROOT, 'data', 'txt_det_rotated')
rotate_filtered_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df_filtered_rotated.csv')

# text_recognitor
rec_thresh = 0.65
rec_out_txt_dir = os.path.join(CONFIG_ROOT, 'text_recognitor', 'txts')

# key_info_extraction
kie_out_txt_dir = os.path.join(CONFIG_ROOT, 'key_info_extraction', 'txts')
kie_boxes_transcripts = os.path.join(CONFIG_ROOT, 'key_info_extraction', 'boxes_and_transcripts')
kie_boxes_transcripts_test = os.path.join(CONFIG_ROOT, 'key_info_extraction', 'boxes_and_transcripts_test')
kie_model_dir = os.path.join(CONFIG_ROOT, 'models', 'pick', 'model_best.pth')
kie_result_dir = os.path.join(CONFIG_ROOT, 'key_info_extraction', 'test_results')
kie_boxes_transcripts_temp = os.path.join(CONFIG_ROOT, 'key_info_extraction', 'boxes_and_transcripts_temp')