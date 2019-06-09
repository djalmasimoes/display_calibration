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

# Read jnd files with measured and expected data
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

# Create training_data = (input, output) pairs
my_dataset = []
for i in range(len(jnd_measured)):
    my_dataset.append((jnd_measured[i], jnd_desired[i]))

my_dataset = np.array(my_dataset)
dataset = np.array(my_dataset)

# Shuffle training data
# np.random.shuffle(dataset)

# Split training and testing sets (data, label)
train_data = dataset[:, 0]
train_label = dataset[:, 1]
test_data = dataset[:, 0]
test_label = dataset[:, 1]

# Build model function
def build_model():
    model = Sequential()
    model.add(layers.Dense(512, input_dim=1, activation='relu'))
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
checkpoint_path = r"C:\Users\Djalma\Google Drive\University\Doctorate - BME\Repository\display_calibration\jnd_correction.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1,
                                              period=1000)

EPOCHS = 10000

# Train model
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0)

'''

# Train model
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
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

# model = keras.models.load_model('jnd_correction.h5')

# Make predictions
test_predictions = model.predict(test_data).flatten()

_ = plt.plot([np.amin(test_label), np.amax(test_label)],
             [np.amin(test_label), np.amax(test_label)], '--r', label='Perfect line')
plt.scatter(test_label, test_predictions, 2, label='Model output')
plt.xlabel('Actual JND index')
plt.ylabel('Predicted JND index')
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
model.save('jnd_correction.h5')