import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Muat model Anda
model = load_model('batik_final.keras')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

def predict_image(image_path, model, threshold=0.5):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)
    predicted_class = np.argmax(predictions)

    if max_prob < threshold:
        return 'random'
    elif predicted_class == 0:
        return 'kawung'  # Kawung
    elif predicted_class == 1:
        return 'megamendung'  # Megamendung
    elif predicted_class == 2:
        return 'parang'  # Parang
    else:
        return 'random'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Lakukan prediksi
        predicted_class = predict_image(file_path, model, threshold=0.5)
        
        return render_template('profile.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
