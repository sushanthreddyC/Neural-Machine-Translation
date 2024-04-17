import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import tensorflow as tf
import tensorflow_text

app=Flask(__name__)



model = tf.saved_model.load('translator')

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

    function_ = [str(x) for x in  request.form.values()]
    print("recieved function :", function_[0])
    tokenized_function = tokenize(function_[0])
    print("tokenized function :", tokenized_function)
    predictions= model.translate(tf.constant([tokenized_function]))
    predicted_value = predictions[0].numpy().decode()
    predicted_value = re.sub(r'\s+', '', predicted_value)
    print("predicted value: ", predicted_value)
    prediction_text = 'Derivative of ' + function_[0] + ' = ' + predicted_value
    return render_template('index.html', prediction_text =  prediction_text)

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     ''' 
#         For direct API calls 
#     '''

if __name__=="__main__":
    app.run(debug=True)


