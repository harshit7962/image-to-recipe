from flask import Flask, render_template
from flask.globals import request

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

newmodel = tf.keras.models.load_model('./saved_model/my_model')

category = {
    0: ['tea', 'Tea'],
    1: ['chapati', 'Chapati'],
    2: ['chole_bhature', 'Chole Bhature'],
    3: ['fried_rice', 'Fried Rice'],
    4: ['momos', 'Momos'],
    5: ['pav_bhaji', 'Pav Bhaji'],
}

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helloe_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    pred = newmodel.predict(img_processed)
    index = np.argmax(pred)
    classification = category[index][1]

    if classification == 'Tea':
        return render_template('chai.html')
    elif classification == 'Chapati':
        return render_template('chapati.html')
    elif classification == 'Pav Bhaji':
        return render_template('pav-bhaji.html')
    elif classification == 'Momos':
        return render_template('momos.html')
    elif classification == 'Fried Rice':
        return render_template('fried-rice.html')
    elif classification == 'Chole Bhature':
        return render_template('chole-bhature.html')

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)