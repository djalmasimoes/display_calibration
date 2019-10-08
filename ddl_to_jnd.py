import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from keras import optimizers
from keras.models import Sequential
import time

"""This module trains a model to compute the measured jnd values for the corresponding DDL"""

start = time.time()

# Read txt file with JND measurements (length = 18)
with open('jnd_meas.txt') as f:
    jnd_measured = f.read()
jnd_measured = jnd_measured.split()
jnd_measured = list(map(float, jnd_measured))
jnd_measured = np.array(jnd_measured, dtype=float)  # input data

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

# Build model
model = Sequential()
model.add(layers.Dense(1024, input_dim=1, activation='relu'))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error',
              optimizer=optimizers.RMSprop(lr=0.001, rho=0.9),
              metrics=['mean_absolute_error', 'mean_squared_error'])
model.summary()

EPOCHS = 10000


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

print(test_label, test_predictions)

# Save entire model to a HDF5 file
# model.save('ddl_to_jnd.h5')

end = time.time()
print('Elapsed time:', end - start, 'seconds')
print('end')

# save train mean squared error
# np.savetxt('model_1_train_MSE.txt', history.history['mean_squared_error'], delimiter=',')

# save validation mean squared error
# np.savetxt('model_1_val_MSE.txt', history.history['val_mean_squared_error'], delimiter=',')
