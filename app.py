from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras import models
import os

from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# ************************************************************

app = Flask(__name__)

UPLOAD_FOLDER = 'Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ******************************************************************

@app.route("/")
def connect():
	return "Connect"
# **********************************************************************

@app.route('/api', methods=['POST'])
def predict():
  if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path1)
        path_to_sample_documents="Images/"+file1.filename
 #************************************************************************************************************** 
        model = models.load_model('model.h5')
#*************************************************************************************************************** 
  test =tf.keras.preprocessing.image.load_img(path_to_sample_documents, target_size=(224, 224))  
  test = tf.keras.preprocessing.image.img_to_array(test)
  test = np.expand_dims(test, axis=0)
  result = model.predict(test)
# ************************************************************************************************************
  if result[0][0] == 1:
    prediction = 'cat'
  else:
    prediction = 'dog'

  return prediction

# *********************************************************************************************************

if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True,use_reloader=False)
