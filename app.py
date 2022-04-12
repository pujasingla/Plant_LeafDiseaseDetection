
from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import keras


from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message


app = Flask(__name__)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'plantdiseasedetection09@gmail.com'
app.config['MAIL_PASSWORD'] = 'test#123@'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail= Mail(app)


MODEL_PATH ='mobilenet_multi.h5'

model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    x = image.img_to_array(img)
    
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased Apple___Apple_scab"
    elif preds==1:
        preds="The leaf is diseased Apple___Black_rot"
    elif preds==2:
        preds="The leaf is Apple___Cedar_apple_rust"
    elif preds==3:
        preds="The leaf is fresh Apple___healthy"
    elif preds==4:
        preds="The leaf is fresh Blueberry___healthy"
    elif preds==5:
        preds="The leaf is Cherry_(including_sour)___Powdery_mildew"
    elif preds==6:
        preds="The leaf is Cherry_(including_sour)___healthy" 
    elif preds==7:
        preds="The leaf is Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    elif preds==8:
        preds="The leaf is Corn_(maize)___Common_rust_"
    elif preds==9:
        preds="The leaf is Corn_(maize)___Northern_Leaf_Blight"
    elif preds==10:
        preds="The leaf is Corn_(maize)___healthy"
    elif preds==11:
        preds="The leaf is Grape___Black_rot"
    elif preds==12:
        preds="The leaf is Grape___Esca_(Black_Measles)"
    elif preds==13:
        preds="The leaf is Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    elif preds==14:
        preds="The leaf is Grape___healthy"
    elif preds==15:
        preds="The leaf is Orange___Haunglongbing_(Citrus_greening)"
    elif preds==16:
        preds="The leaf is Peach___Bacterial_spot"
    elif preds==17:
        preds="The leaf is Peach___healthy"
    elif preds==18:
        preds="The leaf is Pepper,_bell___Bacterial_spot"
    elif preds==19:
        preds="The leaf is Pepper,_bell___healthy"
    elif preds==20:
        preds="The leaf is Potato___Early_blight"
    elif preds==21:
        preds="The leaf is Potato___Late_blight"
    elif preds==22:
        preds="The leaf is Potato___healthy" 
    elif preds==23:
        preds="The leaf is Raspberry___healthy"
    elif preds==24:
        preds="The leaf is Soybean___healthy"
    elif preds==25:
        preds="The leaf is Squash___Powdery_mildew"
    elif preds==26:
        preds="The leaf is Strawberry___Leaf_scorch"
    elif preds==27:
        preds="The leaf is Strawberry___healthy"
    elif preds==28:
        preds="The leaf is Tomato___Bacterial_spot"
    elif preds==29:
        preds="The leaf is Tomato___Early_blight"
    elif preds==30:
        preds="The leaf is Tomato___Late_blight"
    elif preds==31:
        preds="The leaf is Tomato___Leaf_Mold"
    elif preds==32:
        preds="The leaf is Tomato___Septoria_leaf_spot" 
    elif preds==33:
        preds="The leaf is Tomato___Spider_mites Two-spotted_spider_mite" 
    elif preds==34:
        preds="The leaf is Tomato___Target_Spot" 
    elif preds==35:
        preds="The leaf is Tomato___Tomato_Yellow_Leaf_Curl_Virus" 
    elif preds==36:
        preds="The leaf is Tomato___Tomato_mosaic_virus" 
    elif preds==37:
        preds="The leaf is Tomato___healthy"  
    else:
        preds="out of reach. Connot recognise."
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

@app.route('/send_message',methods=['GET','POST'])
def send_message():
    if request.method =="POST":
        email = request.form['email']
        subject = request.form['subject']
        msg = request.form['message']

        message=Message(subject,sender="plantdiseasedetection09@gmail.com",recipients=[email])

        message.body = msg 

        mail.send(message)

        success ="Message send"

        return render_template("result.html",success=success)

if __name__ == '__main__':
    app.run(port=5001,debug=True)
