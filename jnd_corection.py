# jnd_correction.py
# This module trains a model to correct the JND indexes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from keras import optimizers
from keras.models import Sequential

# Read txt file with measured JND
with open('jnd_measured.txt') as f:
    jnd_measured = f.read()
jnd_measured = jnd_measured.split()
jnd_measured = list(map(float, jnd_measured))
jnd_measured = np.array(jnd_measured, dtype=float)
jnd_measured = jnd_measured[:-5]

# Create array gray levels (1-256 for 8-bits)
gray_level = np.linspace(0, 255, 256)

# Calculate mean JND/GL
jnd_mean = (max(jnd_measured) - min(jnd_measured))/(max(gray_level) - min(gray_level))

# Calculate expected JND
jnd_expected = range(1, len(jnd_measured)+1)
jnd_expected = np.array(jnd_expected, dtype=float)
for n in range(1, len(jnd_measured)+1):
    jnd_expected[n-1] = jnd_measured[0] + (n-1)*(jnd_measured[-1] - jnd_measured[0])/(len(jnd_measured)-1)

# Plot measured and expected JND indexes
plt.plot(np.linspace(0, 255, len(jnd_expected)), jnd_expected, 'bs', label='Expected')
plt.plot(np.linspace(0, 255, len(jnd_expected)), jnd_measured, 'ro', label='Measured')
plt.xlabel('Digital driving level (DDL)')
plt.ylabel('JND index')
plt.title('JND response')
plt.legend()
plt.grid()
plt.show()

# Interpolate measured and expected JND (length = 256)
x = np.linspace(0, 255, 256)    # desired length
xp = np.linspace(0, 255, len(jnd_measured))   # current length
jnd_measured_interp = np.interp(x, xp, jnd_measured)
jnd_expected_interp = np.interp(x, xp, jnd_expected)

# Create training_data = (input, output) pairs
dataset = [] = []
for i in range(len(jnd_measured_interp)):
    dataset.append((jnd_measured_interp[i], jnd_expected_interp[i]))
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
plt.legend()
plt.grid()
plt.show()

result = dict(zip(test_label, test_predictions))
print(result)

# Save model in a HDF5 file
# model.save('model_2.h5')

