from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import argmax
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./data/zoo.csv', delimiter=',', skiprows=1, usecols=range(1,18))
x_data = data[0:, 0:-1]
y_data = to_categorical(data[0:, [-1]], 8)[:, 1:]

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')
print(f'y_data.shape: {y_data[0]}')

model = Sequential()
model.add(Flatten(input_shape=(x_data.shape[1],)))
model.add(Dense(y_data.shape[1], activation='softmax'))
# model.add(Dense(y_data.shape[1], input_shape=(x_data.shape[1],), activation='softmax'))

model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=200, validation_split=0.2, verbose=0)

predict_data = np.array([
    x_data[1],
    x_data[2],
])
result_data = [
    y_data[0],
    y_data[2],
]
result = model.predict(predict_data)

print('Result ==========================================================')
print(f'predict_data: {predict_data}')
print(f'result_data: {result_data}')
print(f'result: {result}')
print(f'result(One-Hot encoding): {argmax(result, axis=1)}')

print('Model info ======================================================')
print(model.input)
print(model.output)
print(model.weights)

print('Loss Trend ======================================================')
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='validation_loss')
plt.legend(loc='best')

plt.show()

print('Accuracy ======================================================')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train_accuracy')
plt.plot(hist.history['val_accuracy'], label='validation_accuracy')
plt.legend(loc='best')

plt.show()