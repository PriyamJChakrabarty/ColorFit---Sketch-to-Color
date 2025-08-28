from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
c = 0


model = tf.keras.models.load_model('Models/generator5.h5')

def preprocess_image(image_data):
    """
    Convert base64 image to a NumPy array and preprocess for the model.
    Adds batch and channel dimensions to the input.
    Saves the received image and adds a white background if necessary.
    """
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("RGBA")
    
    save_image(image, prefix='received')

    if image.mode == 'RGBA':
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert("L")
    else:
        image = image.convert("L")
    
    image = image.resize((256, 256))  
    image_array = np.array(image) / 255.0  
    
    image_array = image_array[np.newaxis, ..., np.newaxis] 
    return image_array

def postprocess_image(model_output):
    """
    Convert the model output to a base64-encoded image.
    Saves the generated colored image.
    """
    image = ((model_output[0] + 1) * 127.5).astype(np.uint8)  
    image = Image.fromarray(image)
    
    save_image(image, prefix='generated')
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def save_image(image, prefix='image'):
    """
    Save the PIL Image to the 'saved_images' directory with a timestamp.
    """
    if not os.path.exists('saved_images'):
        os.makedirs('saved_images')
    timestamp = int(tf.timestamp().numpy())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_sketch', methods=['POST'])
def process_sketch():
    global c
    c += 1
    print(c)
    """
    Receive sketch image, add white background, save it, process it through the GAN, and return the colored image.
    """
    try:
        data = request.get_json()
        image_data = data['image']

        sketch = preprocess_image(image_data)
        
        colored_image = model.predict(sketch)

        colored_image_base64 = postprocess_image(colored_image)

        return jsonify({'coloredImage': colored_image_base64})
    except Exception as e:
        print(f"Error processing sketch: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
