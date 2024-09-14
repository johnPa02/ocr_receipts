import os

CONFIG_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(CONFIG_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}