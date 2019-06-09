import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Read jnd files with measured and expected JND
with open('jnd_measured.txt') as f:
    jnd_measured = f.read()
jnd_measured = jnd_measured.split()
jnd_measured = list(map(float, jnd_measured))
jnd_measured = np.array(jnd_measured, dtype=float)  # input data

with open('jnd_desired.txt') as f:
    jnd_desired = f.read()
jnd_desired = jnd_desired.split()
jnd_desired = list(map(float, jnd_desired))
jnd_desired = np.array(jnd_desired, dtype=float)  # output data

# Read file with luminance measurements (length = 18)
with open('luminance.txt') as f:
    luminance_data = f.read()
luminance_data = luminance_data.split()
luminance_data = list(map(float, luminance_data))
luminance_data = np.array(luminance_data, dtype=float)  # input data

# Interpolate luminance values (length = 256)
x = np.linspace(1, 256, 256)    # desired length
xp = np.linspace(1, 256, len(luminance_data))   # current length
luminance_measured = np.interp(x, xp, luminance_data)

# Read file with luminance expected
with open('luminance_expected.txt') as f:
    luminance_data = f.read()
luminance_data = luminance_data.split()
luminance_data = list(map(float, luminance_data))
luminance_data = np.array(luminance_data, dtype=float)  # input data
luminance_expected = luminance_data

# Read file with luminance corrected
with open('luminance_corrected.txt') as f:
    luminance_data = f.read()
luminance_data = luminance_data.split()
luminance_data = list(map(float, luminance_data))
luminance_data = np.array(luminance_data, dtype=float)  # input data
luminance_corrected = luminance_data

# Create array with pixel values, i.e., digital logical levels (1-256)
pixel = np.linspace(1, 256, 256)

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('pixel_to_luminance.h5')
model_2 = keras.models.load_model('jnd_correction.h5')
model_3 = keras.models.load_model('jnd_to_pixel.h5')

luminance_predicted = model_1.predict(pixel).flatten()  # pixel to luminance

# Convert from luminance to jnd
l = luminance_predicted

A = 71.498068
B = 94.593053
C = 41.912053
D = 9.8247004
E = 0.28175407
F = -1.1878455
G = -0.18014349
H = 0.14710899
I = -0.017046845

jnd_conv = A + B*np.log10(l) + C*(np.log10(l))**2 + D*(np.log10(l))**3 +\
           E*(np.log10(l))**4 + F*(np.log10(l))**5 + G*(np.log10(l))**6 + H*(np.log10(l))**7 +\
           I*(np.log10(l))**8
jnd_conv = jnd_conv.astype(int)

jnd_corrected = model_2.predict(jnd_conv).flatten()  # jnd correction
pixel_corrected = model_3.predict(jnd_corrected).flatten()  # jnd to pixel
pixel_corrected = pixel_corrected.astype(int)

# Plot luminance
plt.plot(luminance_measured, 'r', label='Measured')
plt.plot(luminance_corrected, 'b', label='Corrected')
plt.plot(luminance_expected, 'g', label='Expected')
plt.title('Luminance response')
plt.xlabel('Digital driving level (DDL)')
plt.ylabel('Luminance [cd/ $m^{2}$]')
plt.legend()
plt.grid()
plt.show()

# Plot JND values
plt.plot(jnd_measured, 'r', label='Measured')
plt.plot(jnd_corrected, 'b', label='Corrected')
plt.plot(jnd_desired, 'g', label='Expected')
plt.title('JND index')
plt.xlabel('Digital driving level (DDL)')
plt.ylabel('Just noticeable difference')
plt.legend()
plt.grid()
plt.show()
