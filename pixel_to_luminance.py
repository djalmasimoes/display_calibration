import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
# from keras.layers import InputLayer
from keras.models import Sequential
import time

"""This module trains a model to compute the jnd values from luminance measurements"""

start = time.time()

# Read file with luminance measurements (length = 18)
with open('luminance.txt') as f:
    luminance_data = f.read()
luminance_data = luminance_data.split()
luminance_data = list(map(float, luminance_data))
luminance_data = np.array(luminance_data, dtype=float)  # input data

# Interpolate luminance values (length = 256)
x = np.linspace(1, 256, 256)    # desired length
xp = np.linspace(1, 256, len(luminance_data))   # current length
luminance = np.interp(x, xp, luminance_data)

# Plot measured and interpolated values
plt.plot(xp, luminance_data, 'ro', label='Measured')
plt.plot(luminance, 'b--', label='Interpolated')
plt.xlabel('Digital driving level (DDL)')
plt.ylabel('Luminance [cd/ $m^{2}$]')
plt.title('Luminance interpolation')
plt.legend()
plt.grid()
plt.show()

# Create array with pixel values, i.e., digital logical levels (1-256)
pixel = np.linspace(1, 256, 256)

# Create a dataset with (input, output) pairs
dataset = []
for i in range(len(luminance)):
    dataset.append((pixel[i], luminance[i]))
dataset = np.array(dataset)

# Shuffle training data
# np.random.shuffle(dataset)

'''# Split training and testing sets (data, label)
train_data = dataset[:204, 0]
train_label = dataset[:204, 1]
test_data = dataset[204:, 0]
test_label = dataset[204:, 1]'''

# Split training and testing sets (data, label)
train_data = dataset[:, 0]
train_label = dataset[:, 1]
test_data = dataset[:, 0]
test_label = dataset[:, 1]

# Build model function
def build_model():
    model = Sequential()
    model.add(layers.Dense(1024, input_dim=1, activation='relu'))
    model.add(layers.Dense(1))

    optimizer = optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 1000 == 0: print('')
        print('.', end='')


# Create checkpoint callback
checkpoint_path = r"C:\Users\Djalma\Google Drive\University\Doctorate - BME\Repository\display_calibration\pixel_to_luminance.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=5000)

EPOCHS = 35000


# Train model
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0)

'''
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot(), cp_callback])
'''

# Plot history
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean absolute error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Validation')
    plt.title('Training and validating error')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Validation')
    plt.title('Training and validating error')
    plt.legend()
    plt.grid()
    plt.show()


plot_history(history)

# model = keras.models.load_model('pixel_to_luminance.h5')

# Make predictions
test_predictions = model.predict(test_data).flatten()

_ = plt.plot([np.amin(test_label), np.amax(test_label)],
             [np.amin(test_label), np.amax(test_label)], '--r', label='Perfect line')
plt.scatter(test_label, test_predictions, 2, label='Model output')
plt.xlabel('Actual luminance [cd/ $m^{2}$]')
plt.ylabel('Predicted luminance [cd/ $m^{2}$]')
plt.title('Predicted vs. actual values')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.legend()
plt.grid()
plt.show()

print(test_label, test_predictions)

# Save entire model to a HDF5 file
# model.save('pixel_to_luminance.h5')

end = time.time()
print('Elapsed time:', end - start, 'seconds')