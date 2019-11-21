# test_pattern_correction.py
# This module calibrates the luminance response of the TG270 test patterns

import numpy as np
from tensorflow import keras
from PIL import Image
import imageio
import os

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('model_1.h5')
model_2 = keras.models.load_model('model_2.h5')
model_3 = keras.models.load_model('model_3.h5')

# Create directory to save the images
folder_name = str(os.getcwd()) + "/TG270_corrected/"
os.makedirs(os.path.dirname(folder_name), exist_ok=True)

for n in range(0, 260, 5):
    n = f"{n:03d}"
    image = Image.open('./TG270 test patterns/52/TG270-ULN8-'+str(n)+'.tif')   # Load test patterns

    # Run the models
    image_flattened = np.array(image).ravel()
    jnd_predicted = model_1.predict(image_flattened).flatten()
    jnd_corrected = model_2.predict(jnd_predicted).flatten()
    gray_level_corrected = model_3.predict(jnd_corrected).flatten()
    gray_level_corrected = gray_level_corrected.astype(int)

    # Correct image
    for i in range(len(image_flattened)):
        image_flattened[i] = gray_level_corrected[i]

    # Save image
    image_calibrated = np.reshape(image_flattened, (np.array(image).shape[0], np.array(image).shape[1]))
    imageio.imwrite(folder_name + 'cor_TG270-ULN8-'+str(n)+'.tif', image_calibrated)
    print('TG270-ULN8-'+str(n)+' saved...')
