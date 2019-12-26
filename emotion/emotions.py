import cv2
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

set_session(sess)

emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
emotion_offsets = (20, 40)
emotion_classifier = load_model(emotion_model_path)
emotion_target_size = emotion_classifier.input_shape[1:3]

#{0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}

def smiling_detection(faces):
    emotion_text = []
    emotion_probability = []
    gray_face = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (emotion_target_size))
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability.append(np.max(emotion_prediction))
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text.append(emotion_labels[emotion_label_arg])

    return emotion_text, emotion_probability
