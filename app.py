import os
import numpy as np
from tensorflow import keras
import load_pdf_to_img
from flask import Flask, request, jsonify

model = keras.models.load_model('model_margin_identification.h5')

app = Flask(__name__)

@app.route('/identify_margin', methods=("POST", "GET"))
def identify_margin():
    
    cvs = os.listdir('cv')
             
        
    margin_dict = {}
    for cv in cvs:
        if (cv != ".DS_Store") and (cv!='images'):
            img = load_pdf_to_img.pdf_to_img('cv', cv)
            pred = np.round(model.predict(img)[0])
            if pred == 1:            
                margin_dict[cv[:-4]] = 'Has 1 inch margin'
            else:
                margin_dict[cv[:-4]] = "Doesn't have 1 inch margin" 
    
    
    
    return jsonify({'margin':margin_dict})
    


if __name__ == '__main__':
    app.run('0.0.0.0' , 5000 , debug=True , threaded=True)