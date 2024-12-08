from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class_dict= {0: 'Bishop', 1: 'King',2: 'Knight',3: 'Pawn',4: 'Queen',5 : 'Rook'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load your trained model
model = load_model('model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predicted_bit = np.argmax(model.predict(img_array))
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('result.html', prediction=prediction, image_path=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
