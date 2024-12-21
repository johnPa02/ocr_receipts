import os

from flask import Flask, request, jsonify, url_for
from werkzeug.utils import secure_filename

import app_config
from pipeline import OCRPipeline

app = Flask(__name__)

app.config.from_object(app_config)

pipeline = OCRPipeline()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)

            out_path, entities = pipeline.process_image(file_path)
            # Generate a URL for the image that Flask can serve
            relative_path = os.path.relpath(out_path, app.config['UPLOAD_FOLDER'])
            img_url = url_for('static', filename=f'uploads/{relative_path}')
            return jsonify({
                'img_path': img_url,
                'entities': entities
            })
    return app.send_static_file('index.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)