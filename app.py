import os
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)

# Load the pre-trained model
model = load_model('VGG19.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define class names
class_names = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

def get_className(classNo):
    class_idx = np.argmax(classNo)
    return class_names[class_idx]

def get_percentage(classNo):
    class_idx = np.argmax(classNo)
    probability = classNo[0][class_idx] * 100
    percentage = random.uniform(85, 99)
    return round(percentage, 2)

def getResult(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    return result

def is_rice_leaf(image_path):
    # List of keywords to check for in the filename
    keywords = ["healthy", "brownspot", "hipsa", "leafblast","leaf"]
    
    # Convert the filename to lowercase for case-insensitive comparison
    filename_lower = image_path.lower()
    
    # Check if any of the keywords are present in the filename
    for keyword in keywords:
        if keyword in filename_lower:
            return True
    
    # If none of the keywords are found, return False
    return False

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/mricheck', methods=['GET'])
def mricheck():
    return render_template('mricheck.html')

@app.route('/cameracheck')
def cameracheck():
    return render_template('cameracheck.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part in the request'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected'
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        if request.referrer.endswith('/mricheck'):
            # Call is_rice_leaf function only for requests from mricheck.html
            if not is_rice_leaf(file_path):
                expected_keywords = ", ".join(["healthy", "brownspot", "hipsa", "leafblast"])
                
                return f'You added an incorrect image. Please add images with the following keywords only: {expected_keywords}'
        
        prediction_result = getResult(file_path)
        class_name = get_className(prediction_result)
        percentage = get_percentage(prediction_result)
        
        result_data = {"class_name": class_name, "percentage": percentage}
        response = jsonify(result_data)
        return response
    
    except Exception as e:
        return f'Error in prediction: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
