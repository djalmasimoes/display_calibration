import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pydicom as dicom
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

start = time.time()

# Load the image
ds = dicom.dcmread('xr_skull.dcm')
img16 = np.array(ds.pixel_array, dtype=np.uint16)
img16 = np.invert(img16)

# Normalised [0,1]
norm = (img16 - np.min(img16))/np.ptp(img16)

# Normalised 16-bits
img16 = 65535*(norm - np.min(norm))/np.ptp(norm).astype(int)

plt.axis("off")
plt.imshow(img16, cmap='gray')
plt.title('16-bits')
plt.show()
print('16-bits image')
print('Max:', np.max(img16))
print('Min:', np.min(img16), '\n')

# Normalised 8-bits
img8 = np.array(img16, dtype=np.uint16)
norm = (img8 - np.min(img8))/np.ptp(img8)
img8 = 255*(norm - np.min(norm))/np.ptp(norm).astype(np.uint8)
print('8-bits image')
print('Max:', np.max(img8))
print('Min:', np.min(img8), '\n')

print('Size of the image:', img8.shape[0], img8.shape[1], '\n')

plt.axis("off")
plt.imshow(img8, cmap='gray')
plt.title('8-bits')
plt.show()

# Load the models, including weights and optimizer.
model_1 = keras.models.load_model('pixel_to_luminance.h5')
model_2 = keras.models.load_model('jnd_correction.h5')
model_3 = keras.models.load_model('jnd_to_pixel.h5')


# Run the models
pixel = img8.ravel()
print('Running the model...')
luminance_predicted = model_1.predict(pixel).flatten()   # pixel to luminance

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

jnd_corrected = model_2.predict(jnd_conv).flatten()    # jnd correction
pixel_corrected = model_3.predict(jnd_corrected).flatten()  # jnd to pixel
pixel_corrected = pixel_corrected.astype(int)

print('Finished model')
image_corrected = np.reshape(pixel
corrected, (img8.shape[0], img8.shape[1]))

end = time.time()
print('Elapsed time:', end - start, 'seconds')
 
# Plot original and corrected image
fig, (ax1, ax2) = plt.subplots(ncols=2)
img1 = ax1.imshow(img8, cmap='gray')
ax1.set_title('Original image')
ax1.axis("off")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)
tick = list(np.linspace(np.min(img8), np.max(img8), 3))
cbar = fig.colorbar(img1, cax=cax1, ticks=[0, 50, 100, 150, 200, 255])
cbar.ax.set_ylabel('DDL')

img2 = ax2.imshow(image_corrected, cmap='gray')
ax2.set_title('Corrected image')
ax2.axis("off")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(img2, cax=cax2, ticks=[0, 50, 100, 150, 200, 245])
cbar.ax.set_ylabel('DDL')
plt.show()
