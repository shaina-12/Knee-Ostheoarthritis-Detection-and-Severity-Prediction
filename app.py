from flask import Flask
import cv2
import numpy as np
#from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from flask import render_template, redirect, url_for, request
#from werkzeug.utils import secure_filename
#from keras.applications.imagenet_utils import decode_predictions
import os

enet = load_model('model.h5')
enet.make_predict_function()

#Image Size
img_size=256

dic = {0 : 'Normal', 1 : 'Doubtful', 2 : 'Mild', 3 : 'Moderate', 4 : 'Severe'}

app = Flask(__name__)

def model_predict(img_path):
    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    resized=cv2.resize(gray,(img_size,img_size)) 
    i = image.img_to_array(resized)/255.0
    i = i.reshape(1,img_size,img_size,1)
    p = enet.predict_classes(i)
    return dic[p[0]]

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET','POST'])
def diagnose_page():
    return render_template('diagnosis.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img_path = "uploads/" + img.filename    
        img.save(img_path)
        p = model_predict(img_path)
        print(p)
        return str(p).lower()

    return None

@app.route('/team')
def team_page():
    return render_template('team.html')

if __name__ == '__main__':
    app.run(debug=True)