from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import os
import re
import uuid
from io import BytesIO
from PIL import Image

label_map = {0:"Segar", 1:"Tidak Segar"}
model = load_model('model.h5')


load_dotenv()  # Load file .env

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:5000")

@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/scan')
def scan():    
    return render_template('scan.html', header_show=True)    
@app.route('/result', methods=['POST'])
def result():
    try:
        # Handle uploaded file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                return process_image(file)

        # Handle camera image (base64)
        elif 'camera_image' in request.form and request.form['camera_image']:
            img_data = request.form['camera_image']
            return process_camera_image(img_data)

        else:
            data = {
                'label': 'Error',
                'accuracy': 0,
                'image_url': url_for('static', filename='error.png'),
                'status': False
            }
            return jsonify(data)

    except Exception as e:
        print(f"Error: {e}")
        # If error occurs, show error image
        data = {
            'label': 'Error',
            'accuracy': 0,
            'image_url': url_for('static', filename='error.png'),
            'status': False
        }
        return jsonify(data)

def process_image(file):
    """Process uploaded file"""
    # Generate unique filename
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the uploaded file
    file.save(filepath)

    # Process and predict
    return predict_and_render(filepath, filename)

def process_camera_image(img_data):
    """Process camera image (base64)"""
    try:
        # Extract base64 data
        if img_data.startswith('data:image'):
            img_data = re.sub('^data:image/.+;base64,', '', img_data)

        # Decode base64
        img_bytes = base64.b64decode(img_data)

        # Generate unique filename
        filename = f"camera_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save image
        with Image.open(BytesIO(img_bytes)) as img:
            img.save(filepath, 'JPEG')

        # Process and predict
        return predict_and_render(filepath, filename)

    except Exception as e:
        print(f"Camera image processing error: {e}")
        raise

def predict_and_render(filepath, filename):
    """Common prediction and rendering logic"""
    # Process the image for prediction
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using your model
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    label = label_map[class_index]    
    accuracy = 100 - (float(prediction[0][class_index]) * 100)
    

    # Prepare the data for rendering in result.html
    data = {
        'label': label,
        'accuracy': accuracy,
        'image_url': BASE_URL + url_for('uploaded_file', filename=filename),
        'status': label == 'Segar'
    }
    return jsonify(data)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/about')
def about():
    return render_template('about.html', header_show=True)



if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, port=port)
