from flask import Flask, request, jsonify,render_template, send_file
from flask_ngrok import run_with_ngrok
import os
from torch_utils import get_prediction

app = Flask(__name__,static_folder='./static')
run_with_ngrok(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','jfif'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',file_url='', flag = True)

@app.route('/image', methods=['GET'])
def get_image():
    filename = './testimages/1.jpg'
    return send_file(filename, mimetype='image/gif')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.files)
        file = request.files.get('leaf')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        try:
            file.save("./testimages/1.jpg")
            prediction = get_prediction()
            return render_template('index.html',predict=prediction, flag = False)
        except:
            return jsonify({'error': 'error during prediction'})