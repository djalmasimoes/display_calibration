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


# Read files with measured and expected data
with open('x.txt') as f:
    x = f.read()
x = x.split()
x = list(map(float, x))
x = np.array(x, dtype=float)  # input data

with open('y.txt') as f:
    y = f.read()
y = y.split()
y = list(map(float, y))
y = np.array(y, dtype=float)  # output data

# Create training_data = (input, output) pairs
my_dataset = []
for i in range(len(x)):
    my_dataset.append((x[i], y[i]))

my_dataset = np.array(my_dataset)
dataset = np.array(my_dataset)

# Shuffle training data
np.random.shuffle(dataset)

# Split training and testing sets (data, label)
train_data = dataset[:204, 0]
train_label = dataset[:204, 1]
test_data = dataset[204:, 0]
test_label = dataset[204:, 1]

# Normalization function
def norm(x):
    return (x - np.amin(dataset)) / (np.amax(dataset) - np.amin(dataset))

"""
norm_train_data = norm(train_data)
norm_train_label = norm(train_label)
norm_test_data = norm(test_data)
norm_test_label = norm(test_label)
norm_my_dataset = norm(my_dataset)
"""

# Build model function
def build_model():
    model = Sequential()
    model.add(layers.Dense(256, input_dim=1, activation='relu'))
    model.add(layers.Dense(1))

    optimizer = optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
model.summary()

example_batch = dataset[:15, 0]
example_result = model.predict(example_batch)
print('input', example_batch)
print('result', example_result)


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


checkpoint_path = r"C:\Users\Djalma\Google Drive\University\Doctorate - BME\Repository\display_calibration\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                              save_weights_only=True,
                                              verbose=1)

EPOCHS = 5000

'''
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot(), cp_callback])
'''
history = model.fit(
    train_data, train_label,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [JND]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Validation Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$JND^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Validation Error')
    plt.legend()
    plt.show()


plot_history(history)

# result = model.evaluate(test_data, test_label, verbose=0)

test_predictions = model.predict(test_data).flatten()

# Make predictions
_ = plt.plot([np.amin(test_label), np.amax(test_label)],
             [np.amin(test_label), np.amax(test_label)], '--r')
plt.scatter(test_label, test_predictions)
plt.xlabel('True Values [JND]')
plt.ylabel('Predictions [JND]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.show()


mm = np.asarray(test_predictions)
mm = np.sort(mm, axis=0)
test_label = np.sort(test_label)
test_data = np.sort(test_data)
plt.plot(test_label, 'r')
# plt.plot(my_dataset[:, 1], 'r')
plt.plot(mm, 'g')
plt.plot(test_data, 'b')
# plt.xlim([0, 256])
plt.show()

# Create a new model
model_two = build_model()
result = model_two.evaluate(test_data, test_label)
print("Untrained model, accuracy: \n", result)

"""
# Load model
model_two.load_weights(checkpoint_path)
result = model.evaluate(test_data, test_label)
print("\n Restored model, accuracy", result)
"""

