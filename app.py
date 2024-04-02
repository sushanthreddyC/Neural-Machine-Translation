import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import tensorflow as tf
import tensorflow_text 

app=Flask(__name__)

model = pickle.load(open('translator.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
        for rendering results on HTML GUI
    '''
    def tokenize(s):
        v_ptr = r"sin|cos|\^|\/|exp|\d|\w|\(|\)|\+|-|\*+"
        out=' '
        return out.join(re.findall(v_ptr,s))

    function_ = str(request.form.values())
    tokenized_function = tokenize(function_)
    predictions= model.translate(tf.constant([tokenized_function]))
    predicted_value = predictions[0].numpy().decode()
    predicted_value = re.sub(r'\s+', '', predicted_value)

    return render_template('index.html', predicted__text= 'Derivative of the given function is ${}'.format(predicted_value))

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     ''' 
#         For direct API calls 
#     '''

if __name__=="__main__":
    app.run(debug=True)


