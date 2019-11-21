# gray_level_to_jnd.py
# This module trains a model to predict the measured JND index of the display for each gray level

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from keras import optimizers
from keras.models import Sequential

# Read txt file with luminance measurements
with open('luminance_measured.txt') as f:
    lum_measured = f.read()
lum_measured = lum_measured .split()
lum_measured = list(map(float, lum_measured))
lum_measured = np.array(lum_measured , dtype=float)
lum_measured = lum_measured[:]

# Convert luminance values to JND index
A = 71.498068
B = 94.593053
C = 41.912053
D = 9.8247004
E = 0.28175407
F = -1.1878455
G = -0.18014349
H = 0.14710899
I = -0.017046845

jnd_measured = A + B * (np.log10(lum_measured))**1 + C * (np.log10(lum_measured))**2 + D * (np.log10(lum_measured))**3 + \
    E * (np.log10(lum_measured))**4 + F * (np.log10(lum_measured))**5 + G * (np.log10(lum_measured))**6 + \
    H * (np.log10(lum_measured))**7 + I * (np.log10(lum_measured))**8

# Save measured JND values
np.savetxt('jnd_measured.txt', jnd_measured, delimiter=',')

# Interpolate JND values (length = 256)
x = np.linspace(0, 255, 256)    # desired length
xp = np.linspace(0, 255, len(jnd_measured))   # current length
jnd_measured_interp = np.interp(x, xp, jnd_measured)

# Plot measured and interpolated values
plt.plot(xp, jnd_measured, 'ro', label='Measured')
plt.plot(jnd_measured_interp, 'b--', label='Interpolated')
plt.xlabel('Digital driving level (DDL)')
plt.ylabel('JND index')
plt.title('JND interpolation')
plt.legend()
plt.grid()
plt.show()

# Create array gray levels (1-256 for 8-bits)
gray_level = np.linspace(0, 255, 256)

# Create a dataset with (input, output) pairs
dataset = []
for i in range(len(jnd_measured_interp)):
    dataset.append((gray_level[i], jnd_measured_interp[i]))
dataset = np.array(dataset)

# Shuffle training data
np.random.shuffle(dataset)

# Split training and testing sets (data, label)
train_data = dataset[:, 0]
train_label = dataset[:, 1]
test_data = dataset[:, 0]
test_label = dataset[:, 1]

# Build model
model = Sequential()
model.add(layers.Dense(1024, input_dim=1, activation='relu'))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error',
              optimizer=optimizers.RMSprop(lr=0.001, rho=0.9),
              metrics=['mean_absolute_error', 'mean_squared_error'])
model.summary()

EPOCHS = 30000


# Train model
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0)


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
    plt.show()

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

# Make predictions
test_predictions = model.predict(test_data).flatten()

_ = plt.plot([np.amin(test_label), np.amax(test_label)],
             [np.amin(test_label), np.amax(test_label)], '--r', label='Perfect line')
plt.scatter(test_label, test_predictions, 2, label='Model output')
plt.xlabel('Expected JND index')
plt.ylabel('Predicted JND index')
plt.title('Predicted vs. expected values')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.legend()
plt.grid()
plt.show()

result = dict(zip(test_label, test_predictions))
print(result)

# Save model in a HDF5 file
# model.save('model_one.h5')



