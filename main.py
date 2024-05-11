from typing import Tuple

import numpy as np
MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"

''''
Please Evaluate the model in a linux environment. My model has few dependancies whose versions doesnt install in windows

Replace the dummy test.txt using the actual test.txt file

'''
def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
import re
import tensorflow as tf
import tensorflow_text

reloaded = tf.saved_model.load('translator')

def tokenize(s):
    v_ptr = r"sin|cos|\^|\/|exp|\d|\w|\(|\)|\+|-|\*+"
    out=' '
    return out.join(re.findall(v_ptr,s))

def predict(functions: str):
    tokenized_function = tokenize(functions)
    predictions= reloaded.translate(tf.constant([tokenized_function]))
    predicted_value = predictions[0].numpy().decode()
    predicted_value = re.sub(r'\s+', '', predicted_value)
    return predicted_value
# ----------------- END ----------------- #


def main(filepath: str = "./data/test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file(filepath)
    functions = functions[:10]
    true_derivatives = true_derivatives[:10]
    predicted_derivatives = [predict(f) for f in functions]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))
    test_score = 100*(np.mean(scores))
    # Write scores to a file
    with open("metrics.txt", 'w') as outfile:
            # outfile.write("Training accuracy : %2.1f%%\n" % train_score)
            outfile.write("Test accuracy : %2.1f%%\n" % test_score)


if __name__ == "__main__":
    main()

