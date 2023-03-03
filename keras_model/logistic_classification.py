from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

x_data = np.array([
    [1, 2],
    [3, 1],
    [4, 3],
    [6, 2],
])
y_data = np.array([
    [0],
    [0],
    [1],
    [1],
])

print('Data shape ======================================================')
print(f'x_data.shape: {x_data.shape}')
print(f'y_data.shape: {y_data.shape}')

model = Sequential()
model.add(Flatten(input_shape=(x_data.shape[1],)))
model.add(Dense(y_data.shape[1], activation='sigmoid'))
# model.add(Dense(y_data.shape[1], input_shape=(x_data.shape[1],), activation='sigmoid'))

model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_data, y_data, epochs=5000, verbose=0)

predict_data = np.array([
    [5, 3],
    [2, 3],
])
result_data = [
    [1],
    [0],
]
result = model.predict(predict_data)

print('Result ==========================================================')
print(f'predict_data: {predict_data}')
print(f'result_data: {result_data}')
print(f'result: {result}')

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