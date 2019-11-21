# image_calibration.py
# This module calibrates an input gray scale image according to the DICOM GSDF

import numpy as np
from tensorflow import keras
from PIL import Image
import imageio

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('model_1.h5')
model_2 = keras.models.load_model('model_2.h5')
model_3 = keras.models.load_model('model_3.h5')

# Name and path of the image
file = 'TG270-pQC.tif'
im = Image.open('./TG270 test patterns /'+file)

# Run the models
pixel = np.array(im).ravel()
jnd_predicted = model_1.predict(pixel).flatten()  # pixel to luminance
jnd_corrected = model_2.predict(jnd_predicted).flatten()  # jnd correction
pixel_corrected = model_3.predict(jnd_corrected).flatten()  # jnd to pixel
pixel_corrected = pixel_corrected.astype(int)

# Correct the image
for i in range(len(pixel)):
    pixel[i] = pixel_corrected[i]

# Save the image
image_calibrated = np.reshape(pixel, (np.array(im).shape[0], np.array(im).shape[1]))
imageio.imwrite('cor_'+file, image_calibrated)
print('cor_'+file+' saved...')
