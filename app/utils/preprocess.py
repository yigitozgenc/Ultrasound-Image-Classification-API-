from io import BytesIO
from PIL import Image
import numpy as np

# Constants
IMAGE_SIZE = 240

def convert_uploaded_file_to_image(data):
    pil_image = Image.open(BytesIO(data))
    img_batch = resize_image(pil_image)
    return img_batch

def resize_image(image: Image.Image):
    image = image.resize([IMAGE_SIZE, IMAGE_SIZE])
    image = np.asfarray(image)
    image = image/255.0
    img_batch = np.expand_dims(image, 0) 
    return img_batch    
