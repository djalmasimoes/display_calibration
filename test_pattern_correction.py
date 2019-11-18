import numpy as np
from tensorflow import keras
from PIL import Image
import imageio

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('ddl_to_jnd.h5')
model_2 = keras.models.load_model('jnd_correction.h5')
model_3 = keras.models.load_model('jnd_to_ddl.h5')

for n in range(250, 260, 5):
    n = f"{n:03d}"
    # Path of the test patterns
    im = Image.open('./Test patterns TG270/52/TG270-ULN8-'+str(n)+'.tif')

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
    imageio.imwrite('cor_TG270-ULN8-'+str(n)+'.tif', image_calibrated)
    print('TG270-ULN8-'+str(n)+' saved...')



