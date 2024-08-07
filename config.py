import os

CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))

# organizer
train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images')
raw_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df.csv')

# EDA
filtered_train_img_dir = os.path.join(CONFIG_ROOT, 'data', 'train_images_filtered')
filtered_csv = os.path.join(CONFIG_ROOT, 'data', 'mcocr_train_df_filtered.csv')