from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the style transfer model once
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def preprocess_image(path, max_dim=512):
    img = Image.open(path).convert('RGB')
    width, height = img.size
    scale = max_dim / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    img = img.resize(new_size)
    img_np = np.array(img) / 255.0
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
    return tf.expand_dims(img_tensor, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    stylized_url = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and stylize
            content_image = preprocess_image(filepath)
            style_image = preprocess_image(filepath)  # You can replace this with a fixed style image
            stylized_image = hub_model(content_image, style_image)[0]

            # Save stylized image
            output_img = tf.squeeze(stylized_image) * 255
            output_img = tf.cast(output_img, tf.uint8).numpy()
            stylized_filename = f'stylized_{filename}'
            stylized_path = os.path.join(app.config['UPLOAD_FOLDER'], stylized_filename)
            Image.fromarray(output_img).save(stylized_path)

            # URLs for display
            image_url = url_for('static', filename=f'uploads/{filename}')
            stylized_url = url_for('static', filename=f'uploads/{stylized_filename}')

    return render_template('index.html', image_url=image_url, stylized_url=stylized_url)

if __name__ == '__main__':
    app.run(debug=True)
