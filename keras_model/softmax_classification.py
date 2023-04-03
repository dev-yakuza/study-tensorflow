from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import argmax
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([
    [1, 2, 1],
    [2, 1, 3],
    [3, 1, 3],
    [4, 1, 5],
    [1, 7, 5],
    [1, 2, 5],
    [1, 6, 6],
    [1, 7, 7],
])
# y_data: One-Hot encoding
y_data = np.array([
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
])

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')

model = Sequential()
model.add(Flatten(input_shape=(x_data.shape[1],)))
model.add(Dense(y_data.shape[1], activation='softmax'))
# model.add(Dense(y_data.shape[1], input_shape=(x_data.shape[1],), activation='softmax'))

model.compile(optimizer=SGD(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=2000, verbose=0)

predict_data = np.array([
    [1, 11, 7],
    [1, 3, 4],
    [1, 1, 0],
])
result_data = [
    [0],
    [1],
    [2],
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

plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train_loss')
plt.legend(loc='best')

plt.show()