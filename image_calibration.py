import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pydicom as dicom
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import imageio

start = time.time()

im = Image.open('C:/Users/Djalma/Google Drive/University/Doctorate - BME/Repository/display_calibration/Test patterns TG270/52 tests/TG270ULN80035.tif')
im.show()

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('ddl_to_jnd.h5')
model_2 = keras.models.load_model('jnd_correction.h5')
model_3 = keras.models.load_model('jnd_to_ddl.h5')

# Run the models
pixel = np.array(im).ravel()
# pixel = pixel.ravel()
print('Running the model...')
jnd_predicted = model_1.predict(pixel).flatten()  # pixel to luminance
# jnd_conv = jnd_conv.astype(int)

jnd_corrected = model_2.predict(jnd_predicted).flatten()  # jnd correction
pixel_corrected = model_3.predict(jnd_corrected).flatten()  # jnd to pixel
pixel_corrected = pixel_corrected.astype(int)

print('Finished model')
# image_corrected = np.reshape(pixel_corrected, (np.array(im).shape[0], np.array(im).shape[1]))

end = time.time()
print('Elapsed time:', end - start, 'seconds')

for i in range(len(pixel)):
    pixel[i] = pixel_corrected[i]

image_calibrated = np.reshape(pixel, (np.array(im).shape[0], np.array(im).shape[1]))
imageio.imwrite('filename.tif', image_calibrated)

Image.open('filename.tif').show()