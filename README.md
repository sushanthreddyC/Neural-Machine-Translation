# Neural-Machine-Translation

## Objective
- The objective of this project was to create a deep learning model capable of computing the derivative of a given function with respect to a specified variable. This challenge aims to showcase machine learning expertise, specifically avoiding heuristics or rule-based solutions. The project was initiated with a dataset provided in a Google Drive link, containing various examples of functions and their derivatives.

## About the Project
- I employed a deep learning framework using TensorFlow and TensorFlow Text for processing and model development. The approach included several key steps:

- Data Preprocessing: The train.txt file was parsed to separate input functions from their derivatives, which were then tokenized for the neural network.

 - Model Development: I designed a model comprising an encoder-decoder architecture, suitable for sequence prediction tasks. This architecture was chosen for its effectiveness in understanding the context and structure of input sequences, critical for differentiating mathematical functions.

- Training: The model was trained on the preprocessed dataset, with careful monitoring of loss and accuracy metrics to prevent overfitting. Validation and test splits were used to evaluate the model's performance and generalizability.

- Evaluation: I used a portion of the train.txt file to test the model's performance, employing a scoring function that compared the predicted derivatives with the ground truth.

- Deployment: The model, along with a Flask application (app.py), was prepared for deployment, allowing users to input functions and receive their derivatives.


## Result
- The model achieved a commendable accuracy on the test dataset, demonstrating its capability to generalize and compute derivatives of unseen functions. The project deliverables include:

- A trained model with an accuracy score of 90.017% on the test data.
- A detailed summary of the model architecture, showing the layers and the number of parameters at each stage.
- Fully functional train.py and main.py scripts for training the model and evaluating new data.
- A requirements.txt file ensuring reproducibility of the project environment.

## Installation and Usage
- Clone the repository.
- Install dependencies: pip install -r requirements.txt.
- Run the Flask app: python app.py.
- For training the model, execute: python train.py.
- To evaluate the model on new data, use: python main.py.
