import os

from flask import Flask, request, redirect, render_template
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
            redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            redirect(request.url)
        if file and allowed_file(file.filename):
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)

            result = pipeline.process_image(file_path)
            entities = result[0]
            return render_template('result.html', entities=entities)
    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)