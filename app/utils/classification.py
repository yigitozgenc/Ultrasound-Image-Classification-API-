# data can be kept as bytes in an in-memory buffer when we use the io module's Byte IO operations.
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras import backend as K
from .preprocess import convert_uploaded_file_to_image

# Constants
MODEL = load_model('app/models/model.h5',custom_objects={"tf":tf})
TOTAL_CLASSES = 3
IMAGE_SIZE = 240

def classify_image(uploaded_file):
    global model
    # convert uploaded file into bytes and then into numpy array
    image_batch = convert_uploaded_file_to_image(uploaded_file)
    CLASSES =  ['covid','normal','pneumonia']
    predictions = MODEL.predict(image_batch) 
    scores=list(predictions.tolist())
    predicted_index = np.argmax(scores)
    Probability = "{:.3f}".format(max(scores[0]))
    result = {"Prediction":CLASSES[predicted_index],
             "Probability":Probability}
    return result

    
